"""Explicit sparsity curricula, independent of optimizers, epochs and datasets.

The caller's step is the optimizer-update index. Save both schedule settings and
step when checkpointing. Warmup applies to L1/Lp or JumpReLU L0, not auxiliary
reconstruction losses. k curricula affect training only; config.k remains the
declared inference target. p<1 uses the toolkit's smoothed Lp^p penalty: this is
inspired by p-annealing, without its adaptive coefficient matching/resampling.
"""

import math
from dataclasses import dataclass


def linear_value(start, end, step, duration, *, start_step=0):
    """Clamped linear ramp, including both endpoints; duration=0 switches at start."""
    if (
        type(step) is not int
        or step < 0
        or type(duration) is not int
        or duration < 0
        or type(start_step) is not int
        or start_step < 0
        or not math.isfinite(start)
        or not math.isfinite(end)
    ):
        raise ValueError(
            "Finite endpoints and nonnegative integer steps/duration required"
        )
    fraction = (
        float(step >= start_step)
        if duration == 0
        else min(1.0, max(0.0, (step - start_step) / duration))
    )
    return start + (end - start) * fraction


@dataclass(frozen=True)
class SparsitySchedule:
    """Callable accepted by fit(schedule=..., start_step=...).

    Set k_start/k_end together for a dense-to-sparse Top-K curriculum. Set
    p_start/p_end for ReLU/Gated Lp annealing. Combine only options supported by
    the chosen family; SAE.forward rejects incompatible choices. This is a
    collection of explicit controls, not a recommended training recipe.
    """

    warmup_steps: int = 0
    k_start: int | None = None
    k_end: int | None = None
    k_steps: int = 0
    p_start: float = 1.0
    p_end: float = 1.0
    p_steps: int = 0
    anneal_start: int = 0

    def __post_init__(self):
        for name in ("warmup_steps", "k_steps", "p_steps", "anneal_start"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (self.k_start is None) != (self.k_end is None):
            raise ValueError("Supply both k endpoints or neither")
        if self.k_start is not None and any(
            type(n) is not int or n < 1 for n in (self.k_start, self.k_end)
        ):
            raise ValueError("k endpoints must be positive integers")
        if any(
            not math.isfinite(p) or not 0 < p <= 1 for p in (self.p_start, self.p_end)
        ):
            raise ValueError("p endpoints must lie in (0,1]")

    def __call__(self, step):
        options = {
            "sparsity_scale": linear_value(0, 1, step, self.warmup_steps),
            "sparsity_p": linear_value(
                self.p_start,
                self.p_end,
                step,
                self.p_steps,
                start_step=self.anneal_start,
            ),
        }
        if self.k_start is not None:
            options["training_k"] = round(
                linear_value(
                    self.k_start,
                    self.k_end,
                    step,
                    self.k_steps,
                    start_step=self.anneal_start,
                )
            )
        return options
