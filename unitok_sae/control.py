"""Finite support evaluation and distortion-aware fitting inside frozen spans.

Callbacks keep model paths, task definitions and batching explicit. These are
small reusable kernels, not the private campaign controller or its full solver.
"""

import math

import torch

from .steering import _ids, _matrix, project_edit, svd_span


def select_support(dictionary, proposals, evaluate, *, max_groups,
                   max_image_evaluations, images_per_group):
    """Maximize a caller's hard-forward score over finitely many proposed groups.

    ``evaluate(ids, basis)`` returns a finite scalar, larger is better. It must
    use discovery images only. ``images_per_group`` charges *all* image forwards
    inside a callback, including dose probes; it is a declared cost, not measured
    instrumentation. The caller must verify it. No callback runs after either
    ceiling. Duplicate supports are skipped; zero-rank proposals are recorded.

    Candidate generation and scoring must be matched across dictionaries. A raw
    activation ranking alone is not the causal evaluation provided here. Rank is
    recorded, not silently completed; match capacity before comparing results.
    """
    _matrix(dictionary, "dictionary")
    for name, value in (("max_groups", max_groups),
                        ("max_image_evaluations", max_image_evaluations),
                        ("images_per_group", images_per_group)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    seen, trace, best = set(), [], None
    used = 0
    # Bound consumption too: even an infinite stream of duplicate proposals stops.
    iterator = iter(proposals)
    for _ in range(max_groups):
        if used + images_per_group > max_image_evaluations:
            break
        try:
            ids = _ids(next(iterator), dictionary.shape[1])
        except StopIteration:
            break
        key = tuple(sorted(ids))
        if key in seen:
            continue
        seen.add(key)
        basis, info = svd_span(dictionary[:, ids])
        row = {"ids": ids, "rank": info["rank"], "score": None}
        if info["rank"]:
            with torch.no_grad():
                score = float(evaluate(ids, basis))
            used += images_per_group
            if not math.isfinite(score):
                raise ValueError("Support score must be finite")
            row["score"] = score
            if best is None or score > best["score"]:
                best = {**row, "basis": basis}
        trace.append(row)
    if best is None:
        raise ValueError("No nonzero support evaluated within the declared budget")
    return {**best, "trace": trace, "groups_evaluated": sum(r["score"] is not None for r in trace),
            "image_evaluations": used}


def _measure(success, mse):
    if (success.ndim != 1 or mse.shape != success.shape or success.numel() == 0
            or success.dtype != torch.bool or not mse.is_floating_point()
            or not torch.isfinite(mse).all() or (mse < 0).any()):
        raise ValueError("measure must return bool [N] success and finite nonnegative [N] mse")


def fit_span_edit(basis, initial_edit, objective, measure, *, distortion_cap,
                  max_norm, steps, learning_rate=0.02, magnitude_learning_rate=0.05,
                  penalty_weight=1.0, initial_magnitude=None, check_every=5):
    """Fit one spatial edit with a finite Adam trajectory and hard-path checks.

    ``objective(edit) -> (task_loss, per_image_mse)`` must return a differentiable
    scalar task loss and [N] raw-scale reconstruction MSE. The *same* penalty
    weight/cap is used for every compared head or decoded-image objective.
    ``measure(edit) -> (success_bool, hard_mse)`` must evaluate the original hard
    forward on the fitting images. Never use evaluation images in either callback.

    Unit spatial coefficients and log magnitude parameterize E=AQ^T. All iterates
    stay within max_norm; the soft distortion penalty does not ensure feasibility.
    Select a measured iterate with every fitting-image mse <= cap if one exists,
    then highest success rate, then lowest maximum violation. Otherwise return
    the least-violating iterate with feasible=False. That flag says nothing about
    feasibility on new images. Full-resolution model callbacks may need batching.

    Only edit parameters receive accumulated gradients. Freeze model weights and
    use eval mode yourself; native quantizer gradients may be straight-through.
    Returned tensors are detached. Each call is one optimization trajectory.
    """
    _matrix(initial_edit, "initial_edit")
    for name, value in (("distortion_cap", distortion_cap), ("max_norm", max_norm),
                        ("learning_rate", learning_rate),
                        ("magnitude_learning_rate", magnitude_learning_rate)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(penalty_weight) or penalty_weight < 0:
        raise ValueError("penalty_weight must be finite and nonnegative")
    if type(steps) is not int or steps < 1 or type(check_every) is not int or check_every < 1:
        raise ValueError("steps and check_every must be positive integers")
    _, coef = project_edit(initial_edit, basis)
    if coef.numel() == 0 or float(coef.norm()) == 0:
        raise ValueError("Initial edit must have a nonzero projection in the span")
    magnitude = min(float(coef.norm()), max_norm) if initial_magnitude is None else initial_magnitude
    if not math.isfinite(magnitude) or not 0 < magnitude <= max_norm:
        raise ValueError("initial_magnitude must lie in (0,max_norm]")
    basis = basis.detach()
    direction = (coef.detach() / coef.detach().norm()).clone().requires_grad_(True)
    log_magnitude = initial_edit.new_tensor(math.log(magnitude / max_norm), requires_grad=True)
    optimizer = torch.optim.Adam([{"params": [direction], "lr": learning_rate},
                                  {"params": [log_magnitude], "lr": magnitude_learning_rate}])

    def edit_tensor():
        unit = direction / direction.norm().clamp_min(torch.finfo(direction.dtype).tiny)
        # Project after each optimizer step instead of clamping in this graph:
        # a boundary clamp can suppress the magnitude gradient at max_norm.
        scale = max_norm * log_magnitude.exp()
        return (scale * unit) @ basis.T

    trace, best = [], None
    for step in range(steps + 1):
        edit = edit_tensor()
        if step % check_every == 0 or step == steps:
            with torch.no_grad():
                success, mse = measure(edit.detach())
                _measure(success, mse)
                violation = float((mse / distortion_cap - 1).clamp_min(0).max())
                feasible = bool((mse <= distortion_cap).all())
                rate = float(success.float().mean())
                row = {"step": step, "feasible": feasible, "success_rate": rate,
                       "max_violation": violation, "norm": float(edit.norm())}
                trace.append(row)
                key = (int(feasible), rate if feasible else -violation,
                       -violation if feasible else rate)
                if best is None or key > best[0]:
                    best = (key, edit.detach().clone(), dict(row))
        if step == steps:
            break
        task_loss, mse = objective(edit)
        if (task_loss.ndim != 0 or not torch.isfinite(task_loss)
                or mse.ndim != 1 or mse.numel() == 0 or not mse.is_floating_point()
                or not torch.isfinite(mse).all() or (mse < 0).any()):
            raise ValueError("objective must return finite scalar loss and nonnegative [N] mse")
        penalty = (mse / distortion_cap - 1).clamp_min(0).square().mean()
        loss = task_loss + penalty_weight * penalty
        gradients = torch.autograd.grad(loss, (direction, log_magnitude))
        if any(not torch.isfinite(g).all() for g in gradients):
            raise ValueError("Nonfinite edit gradient")
        optimizer.zero_grad(set_to_none=True)
        direction.grad, log_magnitude.grad = gradients
        optimizer.step()
        with torch.no_grad():
            log_magnitude.clamp_(max=0)
    return {"edit": best[1], **best[2], "trace": trace, "steps": steps,
            "distortion_cap": distortion_cap, "penalty_weight": penalty_weight,
            "max_norm": max_norm}
