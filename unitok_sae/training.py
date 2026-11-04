"""Bounded SAE training primitives; no model training starts on import.

Use a token DataLoader over a discovery-only activation dataset. The caller owns
seeds, shuffling, data splits, device and training budget. Families share normalized
decoder columns, with optional AuxK for nonnegative Top-K families. No automatic
dead-neuron resampling is performed. The original pretrained implementation stays
separate. See docs/sae-variants.md for exact objective/recipe differences.
"""

import math
from pathlib import Path

import torch

from .sae import SAEConfig, SparseAutoencoder


def train_step(
    sae,
    activations,
    optimizer,
    *,
    max_grad_norm=1.0,
    sparsity_scale=1.0,
    sparsity_p=1.0,
    training_k=None,
):
    """Take one optimizer step and return detached scalar training diagnostics.

    Input must be frozen activations, on the SAE device. This never backprops
    into UniTok. BatchTopK pooling includes every vector in this batch. Any
    previously calibrated inference threshold is invalidated by a weight update.
    sparsity_scale warms the L1/Lp/L0 penalty; sparsity_p<1 is a smoothed Lp^p
    penalty for ReLU/Gated. training_k overrides the current Top-K training budget.
    AuxK counters advance only on a successful step and count nonzero codes.
    """
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive and finite")
    if not torch.isfinite(activations).all():
        raise ValueError("Nonfinite training activations")
    sae.train()
    optimizer.zero_grad(set_to_none=True)
    output = sae(
        activations.detach(),
        sparsity_scale=sparsity_scale,
        sparsity_p=sparsity_p,
        training_k=training_k,
    )
    if not torch.isfinite(output["loss"]):
        raise ValueError("Nonfinite training loss")
    output["loss"].backward()
    sae.remove_parallel_grad()
    norm = torch.nn.utils.clip_grad_norm_(
        sae.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    optimizer.step()
    sae.normalize_decoder()
    sae.record_activity(output["latent"].detach())
    if hasattr(sae, "threshold_calibrated"):
        sae.threshold_calibrated.fill_(False)
    codes = output["latent"].detach()
    return {
        "loss": float(output["loss"].detach()),
        "mse": float(output["recon_loss"].detach()),
        "l0": float((codes != 0).float().sum(-1).mean()),
        "gradient_norm": float(norm),
        "sparsity_scale": sparsity_scale,
        "sparsity_p": sparsity_p,
        "training_k": training_k,
        "sparsity_loss": float(output["sparsity_loss"].detach()),
        "aux_loss": float(output["aux_loss"].detach()),
        "gated_aux_loss": float(output["gated_aux_loss"].detach()),
        "multitopk_loss": float(output["multitopk_loss"].detach()),
    }


def fit(
    sae,
    batches,
    *,
    max_steps,
    learning_rate=1e-4,
    optimizer=None,
    schedule=None,
    start_step=0,
):
    """Train for at most max_steps from a finite or streaming iterable of tensors.

    Returns (history, optimizer); exhausting the iterable ends the call, rather
    than silently restarting an epoch. Pass the returned optimizer for continued
    training. This function does not choose a split, seed, epoch count or device.
    schedule(step) may return train_step keyword options; start_step is the global
    optimizer-update cursor for continuation. Save schedule settings and cursor
    in checkpoint provenance explicitly; no schedule is inferred on restoration.
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
    if type(start_step) is not int or start_step < 0:
        raise ValueError("start_step must be a nonnegative integer")
    history = []
    iterator = iter(batches)
    for step in range(start_step, start_step + max_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        options = {} if schedule is None else schedule(step)
        history.append({"step": step, **train_step(sae, batch, optimizer, **options)})
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
        frequency = (codes.reshape(-1, sae.hidden_dim) != 0).float().mean(0)
        return {
            "n_vectors": x.shape[0],
            "mse": float(sse / x.numel()),
            "explained_variance": float(1 - sse / variance) if variance > 0 else None,
            "cosine": float(torch.nn.functional.cosine_similarity(x, y).mean()),
            "l0": float((codes != 0).float().sum(-1).mean()),
            "negative_l0": float((codes < 0).float().sum(-1).mean()),
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
