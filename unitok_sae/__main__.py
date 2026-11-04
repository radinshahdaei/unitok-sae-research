"""Small independent commands; run ``python -m unitok_sae --help``.

All commands require explicit inputs and a new output directory. No downloads or
large default jobs. Python API functions remain independently usable.
"""

import argparse
import json
from pathlib import Path

import torch

from .data import ActivationDataset, extract_activations, file_sha256
from .features import edit_features, top_features
from .loading import checked_asset, load_pretrained_sae, load_unitok
from .model import SAEAttachment, preprocess_image
from .sae import SAEConfig, SparseAutoencoder
from .training import fit, load_sae, save_sae


def image_batches(manifest, *, image_size, device):
    """Yield one verified source image at a time from {samples:[...]} JSON.

    Records contain path, id, sha256 and split; attack additionally needs label.
    Relative image paths resolve against the manifest directory. Source hashes
    are checked before preprocessing. This generator chooses no split itself.
    """
    from PIL import Image

    path = Path(manifest).resolve()
    for sample in json.loads(path.read_text())["samples"]:
        image_path = Path(sample["path"])
        image_path = (
            image_path if image_path.is_absolute() else path.parent / image_path
        )
        if file_sha256(image_path) != sample["sha256"]:
            raise ValueError(f"Source image hash mismatch: {sample['id']}")
        with Image.open(image_path) as image:
            images = preprocess_image(image, image_size=image_size)[None].to(device)
        yield {
            "images": images,
            "sample_ids": [sample["id"]],
            "sha256": [sample["sha256"]],
            "split": sample["split"],
            "sample": sample,
        }


def parser():
    """Build the four independent CLI commands without importing model weights."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("extract", "inspect", "attack", "train"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
        if name == "train":
            cmd.add_argument("--manifest", type=Path, required=True)
            cmd.add_argument("--layer", type=int, required=True)
            cmd.add_argument(
                "--variant", choices=["relu", "topk", "batchtopk"], required=True
            )
            cmd.add_argument("--hidden-dim", type=int, required=True)
            cmd.add_argument("--k", type=int, default=64)
            cmd.add_argument("--l1", type=float, default=0)
            cmd.add_argument("--steps", type=int, required=True)
            cmd.add_argument("--batch-size", type=int, default=256)
            cmd.add_argument("--learning-rate", type=float, default=1e-4)
            cmd.add_argument("--seed", type=int, default=0)
            continue
        cmd.add_argument("--config", type=Path, required=True)
        cmd.add_argument("--images", type=Path, required=True)
        if name == "extract":
            cmd.add_argument("--layers", type=int, nargs="+", required=True)
        elif name == "inspect":
            cmd.add_argument("--layer", type=int, required=True)
            cmd.add_argument(
                "--sae",
                type=Path,
                help="Toolkit SAE checkpoint; otherwise use config's author SAE",
            )
            cmd.add_argument(
                "--features", type=int, nargs="+", help="Optional feature IDs to edit"
            )
            cmd.add_argument(
                "--operation", choices=["set", "add", "scale"], default="set"
            )
            cmd.add_argument("--value", type=float, default=0)
        else:
            cmd.add_argument(
                "--prototypes",
                type=Path,
                required=True,
                help="JSON asset spec for .pt containing normalized text_features and recorded vocabulary",
            )
            cmd.add_argument("--epsilon", type=float, required=True)
            cmd.add_argument("--step-size", type=float, required=True)
            cmd.add_argument("--steps", type=int, required=True)
            cmd.add_argument("--target", type=int)
            cmd.add_argument("--objective", choices=["ce", "margin"], default="ce")
            cmd.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    """Execute one explicitly requested command into a new output directory."""
    args = parser().parse_args(argv)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "command.json").write_text(json.dumps(settings, indent=2) + "\n")
    source_hashes = {
        path.name: file_sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    if args.command == "train":
        torch.manual_seed(args.seed)
        dataset = ActivationDataset(args.manifest, layer=args.layer, split="discovery")
        sae = SparseAutoencoder(
            SAEConfig(dataset.input_dim, args.hidden_dim, args.variant, args.k, args.l1)
        ).to(args.device)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False
        )
        batches = (x.to(args.device) for x in loader)
        history, optimizer = fit(
            sae, batches, max_steps=args.steps, learning_rate=args.learning_rate
        )
        # One finite pass only; no implicit epochs. Threshold remains uncalibrated
        # for BatchTopK until explicitly fit on a separate calibration panel.
        save_sae(
            args.output / "sae.pt",
            sae,
            block=args.layer,
            steps=len(history),
            optimizer=optimizer,
            provenance={
                "settings": settings,
                "manifest_sha256": file_sha256(args.manifest),
                "toolkit_source_sha256": source_hashes,
            },
        )
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        return
    config = json.loads(args.config.read_text())
    (args.output / "inputs.json").write_text(
        json.dumps(
            {
                "model_config": config,
                "images_manifest_sha256": file_sha256(args.images),
                "toolkit_source_sha256": source_hashes,
                "toolkit_sae_sha256": file_sha256(args.sae)
                if args.command == "inspect" and args.sae
                else None,
                "preprocessing": preprocess_image.__doc__,
            },
            indent=2,
        )
        + "\n"
    )
    session, _, audit = load_unitok(config, device=args.device)
    (args.output / "load-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    batches = image_batches(
        args.images, image_size=session.image_size, device=args.device
    )
    if args.command == "extract":
        extract_activations(
            session,
            batches,
            args.output / "activations",
            layers=args.layers,
            provenance={
                "config": config,
                "images_manifest_sha256": file_sha256(args.images),
                "preprocessing": preprocess_image.__doc__,
                "toolkit_source_sha256": source_hashes,
            },
        )
    elif args.command == "inspect":
        from .visualization import reconstruction_grid

        if args.sae:
            sae, _ = load_sae(args.sae, device=args.device)
        else:
            sae, _ = load_pretrained_sae(
                config["sae_checkpoint"],
                implementation_repo=config["implementation_repo"],
                block=args.layer,
                block_provenance=config.get("block_provenance"),
                device=args.device,
            )
        edit = (
            None
            if args.features is None
            else lambda z: edit_features(
                z, args.features, operation=args.operation, value=args.value
            )
        )
        with torch.inference_mode():
            for i, batch in enumerate(batches):
                native = session.forward(batch["images"])
                result = session.forward(
                    batch["images"], attachments={args.layer: SAEAttachment(sae, edit)}
                )
                state = result.layers[args.layer]
                torch.save(
                    {
                        "sample": batch["sample"],
                        "codes": state.codes.cpu(),
                        "edited_codes": state.edited_codes.cpu(),
                        "native_indices": native.indices.cpu(),
                        "native_image_features": native.image_features.cpu(),
                        "indices": result.indices.cpu(),
                        "image_features": result.image_features.cpu(),
                        "native_reconstruction": native.reconstruction.cpu(),
                        "edited_reconstruction": result.reconstruction.cpu(),
                        "top_feature_ids": top_features(
                            state.codes, count=min(10, sae.hidden_dim)
                        ).indices.cpu(),
                    },
                    args.output / f"sample-{i:04d}.pt",
                )
                reconstruction_grid(
                    batch["images"], native.reconstruction, result.reconstruction
                ).save(args.output / f"sample-{i:04d}.png")
    else:
        from .attacks import PGDConfig, pgd

        spec = json.loads(args.prototypes.read_text())
        if not spec.get("vocabulary") or not spec.get("templates"):
            raise ValueError(
                "Record prototype vocabulary/order and text templates in asset spec"
            )
        prototypes = torch.load(
            checked_asset(spec), map_location=args.device, weights_only=True
        )["text_features"]
        if prototypes.shape[0] != 1000 or len(spec["vocabulary"]) != 1000:
            raise ValueError("ImageNet CLI requires 1000 ordered prototypes/classes")
        config_attack = PGDConfig(
            epsilon=args.epsilon,
            step_size=args.step_size,
            steps=args.steps,
            seed=args.seed,
            objective=args.objective,
            targeted=args.target is not None,
        )
        for i, batch in enumerate(batches):
            label = batch["sample"]["label"] if args.target is None else args.target
            result = pgd(
                lambda x: session.logits(x, prototypes),
                batch["images"],
                torch.tensor([label], device=args.device),
                config_attack,
            )
            torch.save(
                {
                    "sample": batch["sample"],
                    "images": result.images.cpu(),
                    "delta": result.delta.cpu(),
                    "success": result.success.cpu(),
                    "trajectory": result.trajectory,
                    "settings": result.settings,
                    "prototypes": spec,
                },
                args.output / f"attack-{i:04d}.pt",
            )


if __name__ == "__main__":
    main()
