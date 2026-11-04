"""L-infinity pixel attacks as independent functions over a logits callable.

Inputs/epsilon/step size are always in raw [0,1] units. Model and SAE parameters
are never optimized, and parameter .grad buffers are not populated. Attack a
``lambda x: session.logits(x, prototypes, attachments=...)`` to include edits.
UniTok's native quantizer uses hard forward values and surrogate STE gradients;
these finite attacks are diagnostics, not certificates or gradient-masking audits.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PGDConfig:
    """Optimization settings. Targeted attacks minimize loss for supplied targets.

    Shared mode fits one persistent [1,C,H,W] delta on the supplied discovery set.
    It selects one iterate/restart for the entire set, never per-image deltas.
    Apply the returned delta to new images with apply_delta; never refit on final
    evaluation inputs. Effective perturbations differ after pixel clipping.
    """

    epsilon: float = 4 / 255
    step_size: float = 1 / 255
    steps: int = 20
    restarts: int = 1
    seed: int = 0
    targeted: bool = False
    shared: bool = False
    objective: str = "ce"
    random_start: bool = True

    def __post_init__(self):
        if any(
            not math.isfinite(v) or not 0 < v <= 1
            for v in (self.epsilon, self.step_size)
        ):
            raise ValueError("epsilon and step_size must be finite in (0,1]")
        if (
            any(type(v) is not int or v < 1 for v in (self.steps, self.restarts))
            or type(self.seed) is not int
        ):
            raise ValueError("Integer seed and positive steps/restarts required")
        if self.objective not in {"ce", "margin"}:
            raise ValueError("objective must be ce or margin")


@dataclass
class AttackResult:
    """Frozen attack output plus CPU trajectory; no optimization graphs retained.

    success is target prediction for targeted mode, misclassification otherwise.
    For shared attacks delta is the common underlying pattern; images-original
    is the clipped effective perturbation. Individual mode stores effective deltas.
    """

    images: torch.Tensor
    delta: torch.Tensor
    success: torch.Tensor
    trajectory: list[dict]
    settings: dict


def _check_images(images):
    if (
        images.ndim != 4
        or images.shape[0] == 0
        or not images.is_floating_point()
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
    ):
        raise ValueError("Expected nonempty finite floating BCHW images in [0,1]")


def apply_delta(images, delta):
    """Apply a saved [1,C,H,W] shared or [B,C,H,W] individual pixel perturbation."""
    _check_images(images)
    if (
        delta.ndim != 4
        or delta.shape[1:] != images.shape[1:]
        or delta.shape[0] not in (1, images.shape[0])
        or delta.device != images.device
        or delta.dtype != images.dtype
        or not torch.isfinite(delta).all()
    ):
        raise ValueError("Delta must match image spatial shape, dtype and device")
    return (images + delta).clamp(0, 1)


def _loss(logits, labels, config):
    if (
        logits.ndim != 2
        or logits.shape[0] != labels.shape[0]
        or logits.shape[1] < 2
        or not torch.isfinite(logits).all()
        or labels.min() < 0
        or labels.max() >= logits.shape[1]
    ):
        raise ValueError("Expected finite [B,classes] logits and valid labels")
    if config.objective == "ce":
        loss = F.cross_entropy(logits, labels, reduction="none")
        return loss if config.targeted else -loss
    chosen = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.scatter(1, labels[:, None], -torch.inf).amax(1)
    return other - chosen if config.targeted else chosen - other


def pgd(logits_fn, images, labels, config=None):
    """Optimize bounded targeted/untargeted, individual/shared PGD on native logits.

    Both CE and margin objectives use gradient descent. Retain successful
    candidates ahead of unsuccessful ones, then lower objective. Individual mode
    chooses per image; shared mode chooses by success fraction then mean loss.
    Every iterate, including initialization, is scored on the original forward.
    A shared update takes sign(mean gradient), not mean(sign gradients).
    Caller must use eval-mode deterministic logits and release external graphs.
    """
    config = PGDConfig() if config is None else config
    _check_images(images)
    if torch.is_inference_mode_enabled():
        raise ValueError("Attacks need autograd; do not use torch.inference_mode")
    clean = images.detach()
    labels = torch.as_tensor(labels, device=images.device)
    if labels.dtype != torch.int64 or labels.shape != (images.shape[0],):
        raise ValueError("One int64 label/target per input image required")
    if config.shared and config.targeted and not bool((labels == labels[0]).all()):
        raise ValueError("This shared targeted attack requires one fixed target class")
    generator = torch.Generator(device=images.device).manual_seed(config.seed)
    shape = (1, *images.shape[1:]) if config.shared else images.shape
    best_delta = clean.new_zeros(shape)
    best_images = clean.clone()
    with torch.no_grad():
        native = logits_fn(clean)
        best_loss = _loss(native, labels, config)
        best_success = (
            native.argmax(1).eq(labels)
            if config.targeted
            else native.argmax(1).ne(labels)
        )
    best_key = (float(best_success.float().mean()), -float(best_loss.mean()))
    trajectory = []
    for restart in range(config.restarts):
        delta = (
            (
                torch.rand(
                    shape, device=images.device, dtype=images.dtype, generator=generator
                )
                * 2
                - 1
            )
            * config.epsilon
            if config.random_start
            else clean.new_zeros(shape)
        )
        for step in range(config.steps + 1):
            with torch.enable_grad():
                delta = delta.detach().requires_grad_(step < config.steps)
                candidate = apply_delta(clean, delta)
                logits = logits_fn(candidate)
                losses = _loss(logits, labels, config)
                success = (
                    logits.argmax(1).eq(labels)
                    if config.targeted
                    else logits.argmax(1).ne(labels)
                )
                with torch.no_grad():
                    if config.shared:
                        key = (float(success.float().mean()), -float(losses.mean()))
                        if key > best_key:
                            best_key = key
                            best_delta = delta.detach().clone()
                            best_images = candidate.detach().clone()
                            best_loss, best_success = (
                                losses.detach().clone(),
                                success.detach().clone(),
                            )
                    else:
                        better = (success & ~best_success) | (
                            (success == best_success) & (losses < best_loss)
                        )
                        best_images = torch.where(
                            better[:, None, None, None], candidate.detach(), best_images
                        )
                        best_loss = torch.where(better, losses.detach(), best_loss)
                        best_success = torch.where(better, success, best_success)
                    item = {
                        "restart": restart,
                        "step": step,
                        "mean_loss": float(losses.mean()),
                        "success_fraction": float(success.float().mean()),
                    }
                if step < config.steps:
                    grad = torch.autograd.grad(losses.mean(), delta)[0]
                    if not torch.isfinite(grad).all():
                        raise ValueError("Nonfinite attack gradient")
                    item["gradient_linf"] = float(grad.detach().abs().max())
                    item["zero_gradient"] = bool((grad == 0).all())
                    delta = (delta.detach() - config.step_size * grad.sign()).clamp(
                        -config.epsilon, config.epsilon
                    )
                trajectory.append(item)
    if not config.shared:
        best_delta = best_images - clean
    if (best_images - clean).abs().max() > config.epsilon + 1e-6:
        raise AssertionError("Attack exceeded pixel budget")
    return AttackResult(
        best_images.detach(),
        best_delta.detach(),
        best_success.detach(),
        trajectory,
        asdict(config),
    )


def fgsm(logits_fn, images, labels, *, epsilon=4 / 255, targeted=False):
    """Score one gradient-sign candidate, retaining clean if it is better.

    Uses the same best-candidate rule, pixel units and result type as PGD.
    """
    return pgd(
        logits_fn,
        images,
        labels,
        PGDConfig(
            epsilon=epsilon,
            step_size=epsilon,
            steps=1,
            targeted=targeted,
            random_start=False,
        ),
    )


def autoattack(
    logits_fn, images, labels, *, epsilon, seed=0, batch_size=1, version="standard"
):
    """Optional official AutoAttack integration; never installs the dependency.

    Requires the upstream ``autoattack`` package. This wrapper is untargeted Linf
    evaluation; record its installed revision and configuration. It is separate
    from the PGD implementation above. Do not claim GPU/AutoAttack validation
    from the toolkit's synthetic PGD tests.
    """
    _check_images(images)
    if (
        not math.isfinite(epsilon)
        or not 0 < epsilon <= 1
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("Positive pixel epsilon and batch_size required")
    from autoattack import AutoAttack

    attacker = AutoAttack(
        logits_fn,
        norm="Linf",
        eps=epsilon,
        seed=seed,
        version=version,
        device=str(images.device),
        verbose=False,
    )
    return attacker.run_standard_evaluation(images, labels, bs=batch_size)
