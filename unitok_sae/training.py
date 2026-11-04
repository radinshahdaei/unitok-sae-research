"""Bounded SAE training primitives; no model training starts on import.

Use a token DataLoader over a discovery-only activation dataset. The caller owns
seeds, shuffling, data splits, device and training budget. These baseline variants
use normalized decoder columns and no dead-feature auxiliary/revival loss. The
original pretrained SAE implementation remains available separately.
"""

import math
from pathlib import Path

import torch

from .sae import SAEConfig, SparseAutoencoder


def train_step(sae, activations, optimizer, *, max_grad_norm=1.0):
    """Take one optimizer step and return detached scalar training diagnostics.

    Input must be frozen activations, on the SAE device. This never backprops
    into UniTok. BatchTopK pooling includes every vector in this batch. Any
    previously calibrated inference threshold is invalidated by a weight update.
    """
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite")
    if not torch.isfinite(activations).all():
        raise ValueError("Nonfinite training activations")
    sae.train()
    optimizer.zero_grad(set_to_none=True)
    output = sae(activations.detach())
    if not torch.isfinite(output["loss"]):
        raise ValueError("Nonfinite training loss")
    output["loss"].backward()
    sae.remove_parallel_grad()
    norm = torch.nn.utils.clip_grad_norm_(
        sae.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    optimizer.step()
    sae.normalize_decoder()
    if hasattr(sae, "threshold_calibrated"):
        sae.threshold_calibrated.fill_(False)
    codes = output["latent"].detach()
    return {
        "loss": float(output["loss"].detach()),
        "mse": float(output["recon_loss"].detach()),
        "l0": float((codes > 0).float().sum(-1).mean()),
        "gradient_norm": float(norm),
    }


def fit(sae, batches, *, max_steps, learning_rate=1e-4, optimizer=None):
    """Train for at most max_steps from a finite or streaming iterable of tensors.

    Returns (history, optimizer); exhausting the iterable ends the call, rather
    than silently restarting an epoch. Pass the returned optimizer for continued
    training. This function does not choose a split, seed, epoch count or device.
    """
    if (
        type(max_steps) is not int
        or max_steps < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("Positive max_steps and learning_rate required")
    optimizer = (
        optimizer
        if optimizer is not None
        else torch.optim.Adam(sae.parameters(), lr=learning_rate)
    )
    history = []
    iterator = iter(batches)
    for _ in range(max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        history.append(train_step(sae, batch, optimizer))
    return history, optimizer


@torch.no_grad()
def fidelity(sae, activations):
    """Measure MSE, global explained variance, cosine and L0 on one bounded panel.

    Explained variance uses per-dimension centering over all supplied vectors;
    constant panels return None. Inactive fraction is conditional on this panel,
    not a global dead-feature claim. This does not measure downstream fidelity.
    Restores the caller's SAE mode, including on errors.
    """
    was_training = sae.training
    sae.eval()
    try:
        if not torch.isfinite(activations).all():
            raise ValueError("Nonfinite evaluation activations")
        codes = sae.encode(activations)[0]
        reconstruction = sae.decode(codes)
        x, y = (
            activations.float().reshape(-1, sae.input_dim),
            reconstruction.float().reshape(-1, sae.input_dim),
        )
        sse = (x - y).square().sum()
        variance = (x - x.mean(0)).square().sum()
        frequency = (codes.reshape(-1, sae.hidden_dim) > 0).float().mean(0)
        return {
            "n_vectors": x.shape[0],
            "mse": float(sse / x.numel()),
            "explained_variance": float(1 - sse / variance) if variance > 0 else None,
            "cosine": float(torch.nn.functional.cosine_similarity(x, y).mean()),
            "l0": float((codes > 0).float().sum(-1).mean()),
            "inactive_fraction": float((frequency == 0).float().mean()),
        }
    finally:
        sae.train(was_training)


def save_sae(path, sae, *, block, provenance, steps, optimizer=None):
    """Save a versioned toolkit checkpoint, optionally including optimizer state.

    ``provenance`` records model SHA/revision, activation manifest, preprocessing,
    split, seed and training settings. No artifact is overwritten. This checkpoint
    is distinct from the original author's format; use load_sae to restore it.
    """
    if (
        not provenance
        or type(block) is not int
        or block < 0
        or type(steps) is not int
        or steps < 0
    ):
        raise ValueError(
            "Explicit provenance, nonnegative block and step count required"
        )
    payload = {
        "format": "unitok-sae-toolkit-v1",
        "config": sae.metadata(),
        "block": block,
        "provenance": provenance,
        "steps": steps,
        "state_dict": sae.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_sae(path, *, device="cpu"):
    """Return (eval SAE, checkpoint) without changing RNG or restoring an optimizer.

    For continuation, instantiate an optimizer over the returned SAE, then call
    optimizer.load_state_dict(checkpoint['optimizer']). RNG state is provided;
    data-loader order, CUDA RNG and sampler state remain the caller's responsibility.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "unitok-sae-toolkit-v1":
        raise ValueError(
            "Not a toolkit SAE checkpoint; use the explicit upstream loader"
        )
    sae = SparseAutoencoder(SAEConfig(**payload["config"]))
    sae.load_state_dict(payload["state_dict"], strict=True)
    sae.checkpoint_block = payload["block"]
    sae.eval().to(device)
    return sae, payload
