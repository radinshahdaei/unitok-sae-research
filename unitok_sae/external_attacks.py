"""Pinned external attack backends over eval-mode PyTorch logits models.

Install the optional dependencies with pip install -e '.[attacks]'. Imports here do not install
or download anything. Images and all perturbation budgets use raw [0,1] pixels.
L2 epsilon is a whole-image Euclidean radius, not a per-pixel bound. No ImageNet
normalization is inserted. Use UniTokClassifier for native hard-forward/STE logits.
Checkpoint identity, prototypes/class order and preprocessing remain caller-owned
provenance; algorithm configuration and actual backend versions are returned.
"""

import importlib.metadata
import json
import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .attacks import _check_images


class UniTokClassifier(nn.Module):
    """Expose session.logits as an nn.Module for external libraries.

    Registers the actual UniTok, attached SAEs and text prototypes so .to() moves
    their tensors together. Model and attached SAEs must already be eval-mode.
    SAE codes/quantization stay in the attacked computation if attachments are
    supplied. No classifier head or convenience detached encode_image is used.
    max_batch_size optionally chunks forwards, useful for black-box query batches;
    use it only with batch-independent edit callbacks (not batch-shaped donors).
    For ImageNet, supply all 1000 prototypes in the recorded class order.
    """

    def __init__(
        self,
        session,
        text_features,
        *,
        attachments=None,
        use_logit_scale=True,
        max_batch_size=None,
    ):
        super().__init__()
        session._check_model()
        if max_batch_size is not None and (
            type(max_batch_size) is not int or max_batch_size < 1
        ):
            raise ValueError("max_batch_size must be a positive integer or None")
        self.session = session
        self.unitok = session.model
        self.attachments = {} if attachments is None else dict(attachments)
        if any(spec.sae.training for spec in self.attachments.values()):
            raise ValueError("Attached SAEs must already be in evaluation mode")
        self.saes = nn.ModuleDict(
            {str(k): spec.sae for k, spec in self.attachments.items()}
        )
        self.register_buffer("text_features", text_features.detach().clone())
        self.use_logit_scale, self.max_batch_size = use_logit_scale, max_batch_size
        self.eval()

    def forward(self, images):
        """Return differentiable [B,classes] logits through the native model path."""
        size = images.shape[0] if self.max_batch_size is None else self.max_batch_size
        return torch.cat(
            [
                self.session.logits(
                    x,
                    self.text_features,
                    attachments=self.attachments,
                    use_logit_scale=self.use_logit_scale,
                )
                for x in images.split(size)
            ],
            dim=0,
        )


@dataclass(frozen=True)
class AutoAttackConfig:
    """Official profile or explicit custom profile; reduced budgets require custom.

    standard/plus/rand retain upstream defaults. custom requires an explicit
    component list; optional iterations/restarts/queries/target count/EOT settings
    are applied to upstream attack objects and reported. Targeted suite components
    search internal target classes to find *any* error, not a caller's fixed target.
    """

    epsilon: float
    norm: str = "Linf"
    version: str = "standard"
    attacks: tuple[str, ...] = ()
    seed: int = 0
    batch_size: int = 1
    iterations: int | None = None
    restarts: int | None = None
    queries: int | None = None
    target_classes: int | None = None
    eot_iterations: int | None = None

    def __post_init__(self):
        _budget(self.norm, self.epsilon)
        if self.version not in {"standard", "plus", "rand", "custom"}:
            raise ValueError("Unknown official AutoAttack version")
        _positive_int(self.batch_size, "batch_size")
        _seed(self.seed)
        object.__setattr__(self, "attacks", tuple(self.attacks))
        allowed = {"apgd-ce", "apgd-dlr", "apgd-t", "fab", "fab-t", "square"}
        if (
            len(set(self.attacks)) != len(self.attacks)
            or not set(self.attacks) <= allowed
        ):
            raise ValueError("Unknown or duplicated AutoAttack components")
        overrides = [
            self.iterations,
            self.restarts,
            self.queries,
            self.target_classes,
            self.eot_iterations,
        ]
        if self.version != "custom" and (
            self.attacks or any(v is not None for v in overrides)
        ):
            raise ValueError(
                "Attack selection/budget overrides require version='custom'"
            )
        if self.version == "custom" and not self.attacks:
            raise ValueError("A custom profile requires an explicit component list")
        for value in overrides:
            if value is not None:
                _positive_int(value, "custom budget")


@dataclass
class ExternalAttackResult:
    """Detached native-forward measurements, with full-cohort denominators intact.

    success is misclassification against labels, or equality to supplied targets.
    clean_correct is separate: already wrong clean images are not attack-induced
    failures. No robust-accuracy or successful-only aggregate is inferred here.
    distances are measured on the returned, valid-pixel, epsilon-bounded images.
    Backend counters include checks and final scoring, not only optimizer steps.
    """

    images: torch.Tensor
    success: torch.Tensor
    clean_correct: torch.Tensor
    clean_predictions: torch.Tensor
    predictions: torch.Tensor
    distances: torch.Tensor
    labels: torch.Tensor
    targets: torch.Tensor | None
    settings: dict


class QueryBudgetExceeded(RuntimeError):
    """The explicit cap on total image-forward evaluations was reached."""


class _Counter(nn.Module):
    def __init__(self, model, limit):
        super().__init__()
        self.model, self.limit = model, limit
        self.calls, self.evaluations = 0, 0

    def forward(self, images):
        if self.limit is not None and self.evaluations + len(images) > self.limit:
            raise QueryBudgetExceeded("External attack image-evaluation cap reached")
        self.calls += 1
        self.evaluations += len(images)
        return self.model(images)


def backend_versions():
    """Return installed distribution identities, including AutoAttack's VCS commit."""
    versions = {}
    for name in ("autoattack", "foolbox", "eagerpy"):
        try:
            distribution = importlib.metadata.distribution(name)
            info = {"version": distribution.version}
            if name == "autoattack":
                direct = distribution.read_text("direct_url.json")
                info["commit"] = (
                    json.loads(direct).get("vcs_info", {}).get("commit_id")
                    if direct
                    else None
                )
            versions[name] = info
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError("seed must be an integer in [0,2**32)")


def _budget(norm, epsilon):
    if norm not in {"Linf", "L2"} or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite epsilon and norm Linf/L2 required")
    if norm == "Linf" and epsilon > 1:
        raise ValueError("Linf radius exceeds the entire [0,1] pixel range")


@contextmanager
def _execution(model, seed):
    # Upstream attacks seed global Torch/NumPy RNGs. Restore caller RNGs and
    # parameter flags/grad buffers even on a backend error. FAB uses backward().
    import numpy as np

    numpy_state, python_state = np.random.get_state(), random.getstate()
    parameters = [(p, p.requires_grad, p.grad) for p in model.parameters()]
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            for p, _, _ in parameters:
                p.requires_grad_(False)
            with torch.enable_grad():
                yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        for p, requires_grad, gradient in parameters:
            p.requires_grad_(requires_grad)
            p.grad = gradient


def _prepare(model, images, labels, targets, max_evaluations):
    _check_images(images)
    if torch.is_inference_mode_enabled():
        raise ValueError("External attacks need autograd; leave torch.inference_mode")
    if not isinstance(model, nn.Module) or any(m.training for m in model.modules()):
        raise ValueError("Supply an eval-mode nn.Module, e.g. UniTokClassifier")
    if any(t.device != images.device for t in [*model.parameters(), *model.buffers()]):
        raise ValueError("Move model parameters/buffers to the image device explicitly")
    if max_evaluations is not None:
        _positive_int(max_evaluations, "max_evaluations")
    if (
        labels.dtype != torch.int64
        or labels.shape != (len(images),)
        or labels.device != images.device
    ):
        raise ValueError("labels must be int64 [B] on the image device")
    if targets is not None and (
        targets.dtype != torch.int64
        or targets.shape != labels.shape
        or targets.device != images.device
    ):
        raise ValueError("targets must be int64 [B] on the image device")
    counter = _Counter(model, max_evaluations).eval()
    with torch.no_grad():
        logits = counter(images)
    if (
        logits.ndim != 2
        or logits.shape[0] != len(images)
        or logits.shape[1] < 2
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("Model must return finite [B,classes] logits")
    for values in (labels, targets):
        if values is not None and (values.min() < 0 or values.max() >= logits.shape[1]):
            raise ValueError("Label/target outside the class vocabulary")
    return counter, logits.argmax(1), logits.shape[1]


def _result(
    counter,
    clean,
    adversarial,
    labels,
    targets,
    clean_predictions,
    norm,
    epsilon,
    settings,
):
    if (
        adversarial.shape != clean.shape
        or adversarial.device != clean.device
        or adversarial.dtype != clean.dtype
        or not torch.isfinite(adversarial).all()
    ):
        raise ValueError("Backend returned invalid image tensors")
    if adversarial.min() < -1e-6 or adversarial.max() > 1 + 1e-6:
        raise ValueError("Backend returned images outside [0,1]")
    adversarial = adversarial.detach().clamp(0, 1)
    delta = (adversarial - clean).flatten(1)
    distances = delta.abs().amax(1) if norm == "Linf" else delta.norm(p=2, dim=1)
    if (distances > epsilon + max(1e-6, epsilon * 1e-5)).any():
        raise ValueError("Backend exceeded the declared norm budget")
    with torch.no_grad():
        logits = counter(adversarial)
        if (
            logits.shape != (len(clean), settings["num_classes"])
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("Invalid final hard-forward logits")
        predictions = logits.argmax(1)
    success = predictions.ne(labels) if targets is None else predictions.eq(targets)
    settings.update(
        backend_versions=backend_versions(),
        classifier_forward_calls=counter.calls,
        image_evaluations=counter.evaluations,
        n_source_images=len(clean),
        max_evaluations=counter.limit,
    )
    return ExternalAttackResult(
        adversarial,
        success,
        clean_predictions.eq(labels),
        clean_predictions.detach(),
        predictions.detach(),
        distances.detach(),
        labels.detach().clone(),
        None if targets is None else targets.detach().clone(),
        settings,
    )


def run_autoattack(model, images, labels, config, *, max_evaluations=None):
    """Run an official ensemble or explicitly configured component subset.

    Custom profiles are recorded as custom, even when they contain all components.
    This API is an untargeted evaluation; apgd-t/fab-t are internal search routes,
    not fixed-source/target discovery attacks. Backend warnings remain visible.
    max_evaluations caps image forwards including backend checks/final scoring.
    """
    from autoattack import AutoAttack

    counter, clean_predictions, classes = _prepare(
        model, images, labels, None, max_evaluations
    )
    attack = AutoAttack(
        counter,
        norm=config.norm,
        eps=config.epsilon,
        seed=config.seed,
        version=config.version,
        attacks_to_run=list(config.attacks),
        device=str(images.device),
        verbose=False,
    )
    if config.iterations is not None:
        attack.apgd.n_iter = attack.apgd_targeted.n_iter = attack.fab.n_iter = (
            config.iterations
        )
    if config.restarts is not None:
        attack.apgd.n_restarts = attack.apgd_targeted.n_restarts = (
            attack.fab.n_restarts
        ) = attack.square.n_restarts = config.restarts
    if config.queries is not None:
        attack.square.n_queries = config.queries
    if config.target_classes is not None:
        attack.apgd_targeted.n_target_classes = attack.fab.n_target_classes = (
            config.target_classes
        )
    if config.eot_iterations is not None:
        attack.apgd.eot_iter = attack.apgd_targeted.eot_iter = config.eot_iterations
    if "apgd-dlr" in attack.attacks_to_run and classes < 3:
        raise ValueError("Untargeted DLR requires at least three class logits")
    if "apgd-t" in attack.attacks_to_run and classes < 4:
        raise ValueError("Targeted DLR requires at least four class logits")
    if (
        "apgd-t" in attack.attacks_to_run
        and attack.apgd_targeted.n_target_classes >= classes
    ):
        raise ValueError("APGD target-class count must be smaller than the vocabulary")
    if "fab-t" in attack.attacks_to_run and attack.fab.n_target_classes >= classes:
        raise ValueError("FAB target-class count must be smaller than the vocabulary")
    settings = {
        "backend": "autoattack",
        "config": asdict(config),
        "num_classes": classes,
        "attacks_to_run": list(attack.attacks_to_run),
        "resolved": {
            "apgd_iterations": attack.apgd.n_iter,
            "apgd_restarts": attack.apgd.n_restarts,
            "targeted_iterations": attack.apgd_targeted.n_iter,
            "target_classes": attack.apgd_targeted.n_target_classes,
            "fab_iterations": attack.fab.n_iter,
            "fab_restarts": attack.fab.n_restarts,
            "square_queries": attack.square.n_queries,
            "square_restarts": attack.square.n_restarts,
        },
    }
    with _execution(counter, config.seed):
        adversarial = attack.run_standard_evaluation(
            images.detach().clone(), labels, bs=config.batch_size
        )
    return _result(
        counter,
        images,
        adversarial,
        labels,
        None,
        clean_predictions,
        config.norm,
        config.epsilon,
        settings,
    )


# Only deliberately exposed mappings, not an unrestricted getattr/eval dispatcher.
# Constructor/run kwargs are forwarded unchanged and recorded for each run.
FOOLBOX_ATTACKS = {
    "fgsm": ("Linf", "LinfFastGradientAttack", True),
    "pgd_linf": ("Linf", "LinfProjectedGradientDescentAttack", True),
    "pgd_l2": ("L2", "L2ProjectedGradientDescentAttack", True),
    "cw_l2": ("L2", "L2CarliniWagnerAttack", True),
    "deepfool_l2": ("L2", "L2DeepFoolAttack", False),
    "deepfool_linf": ("Linf", "LinfDeepFoolAttack", False),
    "hopskipjump_l2": ("L2", "HopSkipJumpAttack", True),
    "hopskipjump_linf": ("Linf", "HopSkipJumpAttack", True),
}


def run_foolbox(
    model,
    images,
    labels,
    *,
    attack,
    epsilon,
    targets=None,
    attack_kwargs=None,
    run_kwargs=None,
    seed=0,
    max_evaluations=None,
):
    """Run a selected Foolbox attack with a measured epsilon-bounded return value.

    labels are true/source labels; targets optionally request fixed-target success.
    DeepFool supports untargeted mode only. HopSkipJump accepts explicit adversarial
    starting_points in run_kwargs; its initialization and binary searches consume
    extra queries. For minimization attacks, Foolbox's epsilon-clipped candidate
    can fail even when its unconstrained candidate succeeded; we score the actual
    returned clipped image. No backend success flag is trusted as a substitute.
    """
    import foolbox as fb

    _seed(seed)
    if attack not in FOOLBOX_ATTACKS:
        raise ValueError(f"attack must be one of {tuple(FOOLBOX_ATTACKS)}")
    norm, class_name, supports_targets = FOOLBOX_ATTACKS[attack]
    _budget(norm, epsilon)
    if targets is not None and not supports_targets:
        raise ValueError("This attack does not support fixed targets")
    options = {} if attack_kwargs is None else dict(attack_kwargs)
    run_options = {} if run_kwargs is None else dict(run_kwargs)
    if attack.startswith("hopskipjump"):
        expected = "l2" if norm == "L2" else "linf"
        if "constraint" in options and options["constraint"] != expected:
            raise ValueError(
                "HopSkipJump constraint disagrees with requested attack/norm"
            )
        options["constraint"] = expected
    counter, clean_predictions, classes = _prepare(
        model, images, labels, targets, max_evaluations
    )
    fmodel = fb.PyTorchModel(counter, bounds=(0, 1), device=images.device)
    criterion = (
        fb.criteria.Misclassification(labels)
        if targets is None
        else fb.criteria.TargetedMisclassification(targets)
    )
    algorithm = getattr(fb.attacks, class_name)(**options)
    # Tensor starting points may be large: settings name their shapes; caller
    # should save/hash their actual bytes alongside the result when used.
    recorded_run = {
        k: {"tensor_shape": list(v.shape)} if isinstance(v, torch.Tensor) else v
        for k, v in run_options.items()
    }
    try:
        json.dumps(
            {"attack_kwargs": options, "run_kwargs": recorded_run}, allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Use JSON-serializable options; run_kwargs may additionally contain tensor starting points"
        ) from error
    settings = {
        "backend": "foolbox",
        "attack": attack,
        "norm": norm,
        "epsilon": epsilon,
        "num_classes": classes,
        "seed": seed,
        "targeted": targets is not None,
        "attack_kwargs": options,
        "run_kwargs": recorded_run,
    }
    with _execution(counter, seed):
        _, adversarial, _ = algorithm(
            fmodel, images.detach().clone(), criterion, epsilons=epsilon, **run_options
        )
    return _result(
        counter,
        images,
        adversarial,
        labels,
        targets,
        clean_predictions,
        norm,
        epsilon,
        settings,
    )
