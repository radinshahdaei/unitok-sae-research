"""Stream layer activations into hashed shards, then read individual token rows.

Extraction does not train, download, choose splits, or require an SAE. Source
images are split before tokenization by the caller. Each source has an ID and
content hash, allowing extraction to reject duplicates across splits. Tensor
shards retain [image,token,width] shape and source correspondence; the training
dataset exposes [width] rows without loading the entire corpus into RAM.
"""

import bisect
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def file_sha256(path):
    """Hash file bytes in bounded chunks without reading the whole asset into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_activations(session, batches, destination, *, layers, provenance):
    """Write bounded batches to a new directory and return its manifest path.

    Each batch is a dict: images (float32 BCHW [0,1]), sample_ids (strings),
    sha256 (source-image hashes), split (discovery/calibration/evaluation).
    ``provenance`` must record model identity, preprocessing and code revision.
    Hashes are supplied by the image loader, since resized tensors are not the
    original files. One shard per layer per incoming batch; choose batch sizes
    deliberately. A failed run has no complete manifest and cannot be loaded.
    """
    layers = list(layers)
    if not layers or len(set(layers)) != len(layers) or not provenance:
        raise ValueError("Unique layers and explicit provenance required")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    records, seen_ids, seen_hashes, widths = [], set(), set(), {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            ids, hashes, split = (
                list(batch["sample_ids"]),
                list(batch["sha256"]),
                batch["split"],
            )
            if split not in {"discovery", "calibration", "evaluation"}:
                raise ValueError(
                    "Explicit discovery/calibration/evaluation split required"
                )
            if (
                len(ids) != batch["images"].shape[0]
                or len(hashes) != len(ids)
                or any(not isinstance(x, str) or not x for x in ids)
                or any(
                    not isinstance(x, str)
                    or len(x) != 64
                    or any(c not in "0123456789abcdef" for c in x)
                    for x in hashes
                )
                or len(set(ids)) != len(ids)
                or len(set(hashes)) != len(hashes)
                or seen_ids.intersection(ids)
                or seen_hashes.intersection(hashes)
            ):
                raise ValueError(
                    "Missing/duplicate source IDs or SHA-256; possible split leakage"
                )
            seen_ids.update(ids)
            seen_hashes.update(hashes)
            result = session.forward(batch["images"], layers=layers, decode=False)
            for layer in layers:
                activations = (
                    result.layers[layer].activation.detach().cpu().contiguous()
                )
                if not torch.isfinite(activations).all():
                    raise ValueError("Nonfinite extracted activations")
                if layer in widths and widths[layer] != activations.shape[-1]:
                    raise ValueError("Layer width changed during extraction")
                widths[layer] = activations.shape[-1]
                filename = f"block-{layer}-batch-{batch_index:06d}.pt"
                torch.save(activations, destination / filename)
                records.append(
                    {
                        "file": filename,
                        "sha256": file_sha256(destination / filename),
                        "layer": layer,
                        "shape": list(activations.shape),
                        "split": split,
                        "sample_ids": ids,
                        "source_sha256": hashes,
                    }
                )
            del result
    if not records:
        raise ValueError("No activation batches supplied")
    manifest = {
        "format": "unitok-activations-v1",
        "provenance": provenance,
        "shards": records,
    }
    path = destination / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return path


class ActivationDataset(Dataset):
    """Map-style lazy token dataset for exactly one layer and one explicit split.

    Caches only one shard per process; sequential/shard-local sampling is most
    efficient. Global random token sampling may repeatedly reload large shards.
    Hashes are checked on every cache miss. Each row is cloned to prevent callers
    from changing cached data. Dataset length counts tokens, not source images.
    """

    def __init__(self, manifest, *, layer, split):
        self.manifest = Path(manifest).resolve()
        payload = json.loads(self.manifest.read_text())
        if payload.get("format") != "unitok-activations-v1" or split not in {
            "discovery",
            "calibration",
            "evaluation",
        }:
            raise ValueError("Unsupported activation manifest/split")
        self.shards = [
            s for s in payload["shards"] if s["layer"] == layer and s["split"] == split
        ]
        if not self.shards:
            raise ValueError("No shards for requested layer and split")
        self.ends, total = [], 0
        widths = set()
        for shard in self.shards:
            path = (self.manifest.parent / shard["file"]).resolve()
            if not path.is_relative_to(self.manifest.parent):
                raise ValueError("Shard path escapes manifest directory")
            shape = shard["shape"]
            if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
                raise ValueError("Invalid shard shape")
            widths.add(shape[-1])
            total += shape[0] * shape[1]
            self.ends.append(total)
        if len(widths) != 1:
            raise ValueError("Inconsistent activation widths")
        self.input_dim = widths.pop()
        self._cached_index, self._cached = None, None

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.ends, index)
        if self._cached_index != shard_index:
            spec = self.shards[shard_index]
            path = self.manifest.parent / spec["file"]
            if file_sha256(path) != spec["sha256"]:
                raise ValueError("Activation shard SHA-256 mismatch")
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if (
                list(tensor.shape) != spec["shape"]
                or not tensor.is_floating_point()
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("Invalid activation shard tensor")
            self._cached = tensor.reshape(-1, self.input_dim)
            self._cached_index = shard_index
        start = self.ends[shard_index - 1] if shard_index else 0
        return self._cached[index - start].clone()
