"""Paper-based tokenizer attacks using the installed, pinned upstream APGD.

Methods follow arXiv:2602.18252v1 §3.1 Eq.(1), §3.2 Eq.(2) and Appendix B.1.
No official code release was verified; these adapters are our implementation of
those objectives, not author code or an exact experimental reproduction. Inputs
are float32 RGB BCHW [0,1]. Optimization changes pixels, never token IDs directly.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .attacks import _check_images
from .external_attacks import (
    _budget,
    _Counter,
    _execution,
    _positive_int,
    _seed,
    backend_versions,
)


class UniTokEmbedding(nn.Module):
    """Registered encoder/SAE bridge returning actual quantizer-input [B,T,D].

    The codebooks, classifier and decoder do not execute. Supply optional SAE
    attachments to attack an edited encoder. Model/SAEs must already be eval-mode;
    the caller owns checkpoint, preprocessing and edit provenance. No chunking:
    each reference row remains paired with its source image through all steps.
    """

    def __init__(self, session, *, attachments=None):
        super().__init__()
        session._check_model()
        self.session, self.unitok = session, session.model
        self.attachments = {} if attachments is None else dict(attachments)
        if any(spec.sae.training for spec in self.attachments.values()):
            raise ValueError("Attached SAEs must already be in evaluation mode")
        self.saes = nn.ModuleDict(
            {str(k): spec.sae for k, spec in self.attachments.items()}
        )
        self.eval()

    def forward(self, images):
        """Return graph-connected pre-quantization vectors, with optional edits."""
        return self.session.pre_quantization(images, attachments=self.attachments)


@dataclass(frozen=True)
class EmbeddingAttackConfig:
    """Explicit APGD settings, independent of classification AutoAttack profiles.

    epsilon/norm constrain pixel changes; the embedding objective always uses a
    sum of squared L2 distances without normalizing features. steps=100 matches
    the paper's basic comparison, not every experiment (some use 2000/2500).
    Linf is the paper's evaluated pixel norm; L2 is a toolkit extension.
    """

    epsilon: float
    norm: str = "Linf"
    steps: int = 100
    restarts: int = 1
    seed: int = 0
    rho: float = 0.75

    def __post_init__(self):
        _budget(self.norm, self.epsilon)
        _positive_int(self.steps, "steps")
        _positive_int(self.restarts, "restarts")
        _seed(self.seed)
        if not math.isfinite(self.rho) or not 0 < self.rho < 1:
            raise ValueError("rho must be finite and between zero and one")


def squared_embedding_distance(features, reference):
    """Per-image sum over tokens/channels, requiring equal finite [B,T,D] tensors.

    No implicit broadcasting, unit normalization or per-token averaging. The
    caller explicitly expands a single target for multiple source images.
    """
    if (
        features.ndim != 3
        or features.numel() == 0
        or features.shape != reference.shape
        or features.device != reference.device
        or features.dtype != reference.dtype
        or not features.is_floating_point()
        or not torch.isfinite(features).all()
        or not torch.isfinite(reference).all()
    ):
        raise ValueError("Expected equal finite floating [B,T,D] feature tensors")
    return (features - reference).square().flatten(1).sum(1)


def embedding_consistency_loss(student_features, frozen_clean_features):
    """Eq.(2) outer loss: batch mean of summed squared distances to clean teacher.

    The reference is always detached; gradients flow only through the supplied
    student features. Obtain the teacher from a separately frozen ORIGINAL
    encoder on clean images, not the evolving student. This is one loss helper,
    not adversarial training: generate attacks against that same frozen reference,
    then recompute student features with training gradients and update only the
    explicitly chosen encoder parameters. It does not freeze/select modules or
    silently run a training pipeline.
    """
    return squared_embedding_distance(
        student_features, frozen_clean_features.detach()
    ).mean()


@dataclass
class EmbeddingAttackResult:
    """Measured objective and pixel bounds, with no invented classification success.

    distances_before/after are per-image SQUARED embedding distances. pixel_norms
    are Linf or L2 distances in raw [0,1] units. best candidates include the clean
    input, so match never increases and diverge never decreases the objective
    distance (up to numerical ties). references are detached, cloned and retained
    for replay; caller must save source/target/checkpoint IDs alongside the result.
    """

    images: torch.Tensor
    reference_features: torch.Tensor
    distances_before: torch.Tensor
    distances_after: torch.Tensor
    pixel_norms: torch.Tensor
    settings: dict


class _DistanceScores(nn.Module):
    """Two-column carrier for APGD's custom loss hook, never classifier logits."""

    def __init__(self, extractor, reference, mode):
        super().__init__()
        self.extractor, self.mode = extractor, mode
        self.register_buffer("reference", reference)

    def forward(self, images):
        distance = squared_embedding_distance(self.extractor(images), self.reference)
        score = distance if self.mode == "diverge" else -distance
        return torch.stack((score, torch.zeros_like(score)), dim=1)


def attack_embeddings(
    extractor,
    images,
    config,
    *,
    mode="diverge",
    reference_features=None,
    max_evaluations=None,
):
    """APGD on pre-quantization distance, with random starts and best-loss restarts.

    diverge defaults to this extractor's clean features (Eq.1); match requires
    explicit target-image features (Appendix B.1). For Eq.2's inner maximization,
    use diverge and pass original frozen teacher features explicitly. References
    may have batch 1 (explicitly expanded here) or B, with matching T,D/device/dtype.

    Upstream APGD has no public callable-loss argument. A tiny local subclass
    replaces its dlr_loss hook with the exact signed squared-distance objective.
    'dlr' is only its internal dispatch slot: NO DLR ratio/CE/softmax is applied.
    best_loss=True keeps every source in every restart and disables classification
    early stopping/subsetting. Upstream controls projection, random initialization,
    momentum, oscillation checks and adaptive steps. A random start avoids the
    identically zero gradient of Eq.1 at delta=0. This is not AutoAttack standard.
    """
    from autoattack.autopgd_base import APGDAttack

    _check_images(images)
    if torch.is_inference_mode_enabled():
        raise ValueError("Embedding attacks require autograd outside inference_mode")
    if not isinstance(extractor, nn.Module) or any(
        m.training for m in extractor.modules()
    ):
        raise ValueError("Supply an eval-mode nn.Module feature extractor")
    if any(
        t.device != images.device
        for t in [*extractor.parameters(), *extractor.buffers()]
    ):
        raise ValueError("Move the extractor to the image device explicitly")
    if mode not in {"diverge", "match"}:
        raise ValueError("mode must be 'diverge' or 'match'")
    if mode == "match" and reference_features is None:
        raise ValueError("match requires explicit target-image reference_features")
    if max_evaluations is not None:
        _positive_int(max_evaluations, "max_evaluations")
    versions = backend_versions()
    if versions["autoattack"]["commit"] != "a39220048b3c9f2cca9a4d3a54604793c68eca7e":
        raise ValueError(
            "Install the tested APGD pin with pip install -e '.[attacks]'"
        )
    counter = _Counter(extractor, max_evaluations).eval()

    class DistanceAPGD(APGDAttack):
        def dlr_loss(self, scores, _labels):
            # Exact objective, used for both gradients and adaptive step decisions.
            return scores[:, 0]

    with _execution(counter, config.seed):
        with torch.no_grad():
            clean_features = counter(images)
            reference = (
                clean_features if reference_features is None else reference_features
            )
            if reference.ndim == 3 and reference.shape[0] == 1:
                reference = reference.expand(len(images), -1, -1)
            reference = reference.detach().clone()
            before = squared_embedding_distance(clean_features, reference)
        carrier = _DistanceScores(counter, reference, mode).eval()
        optimizer = DistanceAPGD(
            carrier,
            n_iter=config.steps,
            n_restarts=config.restarts,
            eps=config.epsilon,
            norm=config.norm,
            seed=config.seed,
            rho=config.rho,
            loss="dlr",
            device=str(images.device),
            verbose=False,
        )
        candidates = optimizer.perturb(
            images.detach().clone(),
            torch.zeros(len(images), dtype=torch.int64, device=images.device),
            best_loss=True,
        )
        with torch.no_grad():
            after = squared_embedding_distance(counter(candidates), reference)
            improve = after > before if mode == "diverge" else after < before
            candidates = torch.where(
                improve[:, None, None, None], candidates, images
            ).detach()
            after = torch.where(improve, after, before)
            delta = (candidates - images).flatten(1)
            pixel_norms = (
                delta.abs().amax(1) if config.norm == "Linf" else delta.norm(dim=1)
            )
        if (
            not torch.isfinite(candidates).all()
            or candidates.min() < 0
            or candidates.max() > 1
            or (pixel_norms > config.epsilon + max(1e-6, config.epsilon * 1e-5)).any()
        ):
            raise ValueError(
                "APGD returned nonfinite pixels or exceeded the pixel budget"
            )
    return EmbeddingAttackResult(
        candidates,
        reference,
        before.detach(),
        after.detach(),
        pixel_norms.detach(),
        {
            "method": "paper-based embedding APGD adaptation",
            "paper": "arXiv:2602.18252v1",
            "mode": mode,
            "config": asdict(config),
            "reference": "same-extractor clean"
            if reference_features is None
            else "caller-supplied",
            "objective": "sum_token_channel_squared_distance",
            "selection": "best_loss_all_sources_all_restarts_including_clean",
            "backend_versions": versions,
            "extractor_forward_calls": counter.calls,
            "image_evaluations": counter.evaluations,
            "max_evaluations": counter.limit,
            "feature_shape": list(reference.shape),
        },
    )


def token_change_counts(clean_indices, attacked_indices):
    """Count changed codebook entries AND token positions, independently per image.

    Native indices are int64 [B,codebooks,T]. A position changes if any codebook
    entry differs there. Counts/fractions measure disagreement, not semantic damage
    or attack success. No fixed token/codebook counts and no numeric ID distances.
    """
    if (
        clean_indices.ndim != 3
        or clean_indices.numel() == 0
        or clean_indices.dtype != torch.int64
        or attacked_indices.dtype != torch.int64
        or clean_indices.shape != attacked_indices.shape
        or clean_indices.device != attacked_indices.device
    ):
        raise ValueError("Expected equal nonempty int64 [B,codebooks,T] indices")
    changed = clean_indices != attacked_indices
    return {
        "changed_entries": changed.flatten(1).sum(1),
        "entry_fraction": changed.float().flatten(1).mean(1),
        "changed_positions": changed.any(1).sum(1),
        "position_fraction": changed.any(1).float().mean(1),
        "entries_per_image": clean_indices.shape[1] * clean_indices.shape[2],
        "positions_per_image": clean_indices.shape[2],
    }
