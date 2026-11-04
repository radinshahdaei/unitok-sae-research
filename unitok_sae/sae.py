"""Small trainable SAE families with a shared encode/decode interface.

These are toolkit implementations, not reproductions of published training
recipes. They use upstream-compatible ``pre_bias``, ``encoder`` and bias-free
``decoder`` conventions. Existing author checkpoints retain their original class
and auxiliary loss via ``loading.load_pretrained_sae``.

BatchTopK selects at most N*k positive entries across all N input vectors during
training (tokens count as vectors). Evaluation uses a separately calibrated fixed
threshold, so changing the inference batch cannot change an image's codes.
Calibrate again after training or changing weights. No inference-time k guarantee
is made. See Bussmann et al., arXiv:2412.06410, training/inference formulation.
"""

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .activations import jump_count, jump_relu

SAE_VARIANTS = (
    "relu",
    "topk",
    "batchtopk",
    "jumprelu",
    "gated",
    "matryoshka",
    "multitopk",
    "abstopk",
)


@dataclass(frozen=True)
class SAEConfig:
    """Architecture/loss settings; L1 is mean per-vector sum of codes.

    Set a positive l1_coefficient for a sparse ReLU/L1 baseline. Zero deliberately
    permits an unregularized ReLU autoencoder control; it has no sparse budget.

    k is a per-vector budget for topk/multitopk/abstopk and an average training
    budget for batchtopk/matryoshka. Other families ignore k. JumpReLU uses
    l0_coefficient, learned log thresholds initialized at initial_threshold, and
    a rectangular surrogate of width bandwidth, in *activation units*. Tune these
    on discovery/calibration data; LM-normalized defaults do not transfer to raw
    UniTok activations automatically. Gated uses l1_coefficient on gate activations
    and gated_aux_coefficient on the detached-decoder reconstruction.

    prefixes are fixed ordered Matryoshka widths, ending at hidden_dim. The
    stop-gradient flag is the paper's explicit group-isolation ablation. Multi-
    TopK adds multitopk_weight times the larger-budget reconstruction, using
    min(multitopk_multiplier*k, hidden_dim) features. AuxK is disabled by default;
    when enabled it uses up to aux_k dead latents after dead_after_vectors input
    vectors without a main-path activation, tracked at batch resolution.
    """

    input_dim: int
    hidden_dim: int
    variant: str = "topk"
    k: int = 64
    l1_coefficient: float = 0.0
    l0_coefficient: float = 0.0
    initial_threshold: float = 0.001
    bandwidth: float = 0.001
    gated_aux_coefficient: float = 1.0
    prefixes: tuple[int, ...] = ()
    matryoshka_stop_gradient: bool = False
    multitopk_multiplier: int = 4
    multitopk_weight: float = 0.125
    aux_coefficient: float = 0.0
    aux_k: int = 512
    dead_after_vectors: int = 10_000_000

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.variant not in SAE_VARIANTS:
            raise ValueError(f"variant must be one of {SAE_VARIANTS}")
        if self.k > self.hidden_dim and self.variant not in {
            "relu",
            "gated",
            "jumprelu",
        }:
            raise ValueError("k cannot exceed hidden_dim")
        for name in (
            "l1_coefficient",
            "l0_coefficient",
            "gated_aux_coefficient",
            "multitopk_weight",
            "aux_coefficient",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("initial_threshold", "bandwidth"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("multitopk_multiplier", "aux_k", "dead_after_vectors"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.matryoshka_stop_gradient) is not bool:
            raise ValueError("matryoshka_stop_gradient must be boolean")
        object.__setattr__(self, "prefixes", tuple(self.prefixes))
        if self.variant == "matryoshka":
            if (
                not self.prefixes
                or self.prefixes[-1] != self.hidden_dim
                or any(type(n) is not int or n < 1 for n in self.prefixes)
                or any(a >= b for a, b in zip(self.prefixes, self.prefixes[1:]))
            ):
                raise ValueError(
                    "Matryoshka prefixes must increase strictly and end at hidden_dim"
                )
        elif self.prefixes or self.matryoshka_stop_gradient:
            raise ValueError("Prefix options require variant='matryoshka'")
        if self.l0_coefficient and self.variant != "jumprelu":
            raise ValueError("l0_coefficient is for JumpReLU's threshold surrogate")
        if self.variant == "jumprelu" and self.l1_coefficient:
            raise ValueError(
                "JumpReLU uses l0_coefficient; l1_coefficient would be ignored"
            )
        if self.aux_coefficient and self.variant not in {
            "topk",
            "batchtopk",
            "multitopk",
            "matryoshka",
        }:
            raise ValueError("AuxK currently supports nonnegative Top-K families only")


class SparseAutoencoder(nn.Module):
    """Operate on any ``[..., input_dim]`` tensor, without input normalization.

    ``encode`` returns (codes, None, None, preactivations), matching the existing
    adapter's tuple interface. ``decode`` includes pre_bias; ``decoder(delta)``
    does not, which is essential for residual-preserving edits.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.nonnegative_codes = config.variant != "abstopk"
        self.input_dim, self.hidden_dim, self.k = (
            config.input_dim,
            config.hidden_dim,
            config.k,
        )
        self.pre_bias = nn.Parameter(torch.zeros(self.input_dim))
        self.encoder = nn.Linear(self.input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("threshold_calibrated", torch.tensor(False))
        if config.variant == "jumprelu":
            self.log_threshold = nn.Parameter(
                torch.full((self.hidden_dim,), math.log(config.initial_threshold))
            )
        if config.variant == "gated":
            self.log_magnitude_scale = nn.Parameter(torch.zeros(self.hidden_dim))
            self.magnitude_bias = nn.Parameter(torch.zeros(self.hidden_dim))
        if config.aux_coefficient:
            self.register_buffer(
                "vectors_since_active", torch.zeros(self.hidden_dim, dtype=torch.int64)
            )
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder()

    def _preactivations(self, x):
        if x.ndim < 2 or x.shape[-1] != self.input_dim or x.numel() == 0:
            raise ValueError("Expected nonempty [..., input_dim] activations")
        return self.encoder(x - self.pre_bias)

    def encode(self, x, *, k=None):
        """Return exact codes and preactivations; hard supports can jump.

        Optional k changes a Top-K family's budget explicitly. BatchTopK and
        Matryoshka accept this override during training only, since inference
        uses the separately calibrated threshold. It does not mutate config.k.
        AbsTopK selects by magnitude and preserves signs; other codes are >=0.
        """
        pre = self._preactivations(x)
        positive = pre.relu()
        budget = self.k if k is None else k
        if self.config.variant in {
            "topk",
            "multitopk",
            "abstopk",
            "batchtopk",
            "matryoshka",
        }:
            if type(budget) is not int or not 1 <= budget <= self.hidden_dim:
                raise ValueError("k must be an integer in [1,hidden_dim]")
        elif k is not None:
            raise ValueError("k override applies only to Top-K families")
        if self.config.variant == "relu":
            codes = positive
        elif self.config.variant == "jumprelu":
            codes = jump_relu(
                positive, self.log_threshold.exp(), bandwidth=self.config.bandwidth
            )
        elif self.config.variant == "gated":
            raw = F.linear(x - self.pre_bias, self.encoder.weight)
            magnitude = (
                raw * self.log_magnitude_scale.exp() + self.magnitude_bias
            ).relu()
            codes = magnitude * (pre > 0)
        elif self.config.variant == "abstopk":
            indices = pre.abs().topk(budget, dim=-1).indices
            codes = torch.zeros_like(pre).scatter(-1, indices, pre.gather(-1, indices))
        elif self.config.variant in {"batchtopk", "matryoshka"} and not self.training:
            if k is not None:
                raise ValueError(
                    "BatchTopK inference uses calibrated thresholds, not a k override"
                )
            if not bool(self.threshold_calibrated):
                raise RuntimeError("Calibrate BatchTopK threshold before evaluation")
            codes = positive * (positive > self.threshold)
        elif self.config.variant in {"batchtopk", "matryoshka"}:
            flat = positive.flatten()
            count = positive.numel() // self.hidden_dim * budget
            values, indices = flat.topk(count)
            codes = (
                torch.zeros_like(flat).scatter(0, indices, values).reshape_as(positive)
            )
        else:
            values, indices = pre.topk(budget, dim=-1)
            codes = torch.zeros_like(pre).scatter(-1, indices, values.relu())
        return codes, None, None, pre

    def decode(self, codes):
        """Map sparse codes to activation space, including the learned bias."""
        return self.decoder(codes) + self.pre_bias

    def forward(self, x, *, sparsity_scale=1.0, sparsity_p=1.0, training_k=None):
        """Return codes and separately named reconstruction/sparsity/auxiliary losses.

        MSE averages over vectors *and dimensions*, unlike papers using mean
        squared vector norm; sparsity coefficients therefore need calibration.
        Gated uses a detached-decoder auxiliary reconstruction and L1 on gate
        activations, not magnitudes. JumpReLU uses learned-threshold L0. Matryoshka
        averages prefix MSEs (paper Appendix B sums; inspected trainer averages).
        Multi-TopK adds weight*MSE(multiplier*k), capped at dictionary width.
        p<1 is a smoothed Lp^p objective for ReLU/Gated only, not an exact
        reproduction of the adaptive-coefficient p-annealing training recipe.
        """
        if not math.isfinite(sparsity_scale) or sparsity_scale < 0:
            raise ValueError("sparsity_scale must be finite and nonnegative")
        if not math.isfinite(sparsity_p) or not 0 < sparsity_p <= 1:
            raise ValueError("sparsity_p must be in (0,1]")
        if sparsity_p != 1 and self.config.variant not in {"relu", "gated"}:
            raise ValueError("Lp objectives apply to ReLU/Gated families")
        codes, _, _, pre = self.encode(x, k=training_k)
        reconstruction = self.decode(codes)
        mse = F.mse_loss(reconstruction.float(), x.float())
        l1 = codes.float().abs().sum(-1).mean()
        objective = mse
        zero = mse.new_zeros(())
        gate_aux, extra_recon = zero, zero
        sparse_values = pre.relu() if self.config.variant == "gated" else codes.abs()
        if sparsity_p == 1:
            penalty = sparse_values.float().sum(-1).mean()
        else:
            penalty = (
                ((sparse_values.float() + 1e-6).pow(sparsity_p) - 1e-6**sparsity_p)
                .sum(-1)
                .mean()
            )
        if self.config.variant == "jumprelu":
            penalty = (
                jump_count(
                    pre.relu(),
                    self.log_threshold.exp(),
                    bandwidth=self.config.bandwidth,
                )
                .sum(-1)
                .mean()
            )
            coefficient = self.config.l0_coefficient
        else:
            coefficient = self.config.l1_coefficient
        if self.config.variant == "gated":
            via_gate = F.linear(
                pre.relu(), self.decoder.weight.detach(), self.pre_bias.detach()
            )
            gate_aux = F.mse_loss(via_gate.float(), x.float())
            objective = objective + self.config.gated_aux_coefficient * gate_aux
        prefix_losses = []
        if self.config.variant == "matryoshka":
            current = self.pre_bias.expand_as(x)
            start = 0
            for end in self.config.prefixes:
                current = current + F.linear(
                    codes[..., start:end], self.decoder.weight[:, start:end]
                )
                prefix_losses.append(F.mse_loss(current.float(), x.float()))
                current = (
                    current.detach()
                    if self.config.matryoshka_stop_gradient
                    else current
                )
                start = end
            objective = torch.stack(prefix_losses).mean()
        if self.config.variant == "multitopk":
            active_k = self.k if training_k is None else training_k
            larger = min(self.hidden_dim, self.config.multitopk_multiplier * active_k)
            extra_recon = F.mse_loss(
                self.decode(self.encode(x, k=larger)[0]).float(), x.float()
            )
            objective = objective + self.config.multitopk_weight * extra_recon
        aux_loss = (
            self.auxiliary_loss(x, reconstruction, codes, pre)
            if self.config.aux_coefficient
            else zero
        )
        return {
            "reconstruction": reconstruction,
            "latent": codes,
            "recon_loss": mse,
            "l1": l1,
            "sparsity_loss": penalty,
            "gated_aux_loss": gate_aux,
            "multitopk_loss": extra_recon,
            "prefix_losses": tuple(prefix_losses),
            "aux_loss": aux_loss,
            "loss": objective
            + sparsity_scale * coefficient * penalty
            + self.config.aux_coefficient * aux_loss,
        }

    def decode_prefix(self, codes, prefix):
        """Decode a Matryoshka prefix, keeping the same codes/global selection."""
        if (
            self.config.variant != "matryoshka"
            or type(prefix) is not int
            or prefix not in self.config.prefixes
        ):
            raise ValueError("Choose a configured Matryoshka prefix")
        return F.linear(
            codes[..., :prefix], self.decoder.weight[:, :prefix], self.pre_bias
        )

    def auxiliary_loss(self, x, reconstruction, codes, pre):
        """AuxK reconstructs detached residual with currently inactive dead features.

        Uses raw MSE and a vector-count dead window, not a variance-normalized
        reference-trainer loss. Negative dead preactivations still ReLU to zero;
        AuxK is not a guarantee of revival. No persistent counters change here.
        """
        active = (codes.detach().reshape(-1, self.hidden_dim) != 0).any(0)
        dead = (self.vectors_since_active >= self.config.dead_after_vectors) & ~active
        n_dead = int(dead.sum())
        if not n_dead:
            return reconstruction.new_zeros(())
        values, indices = pre.masked_fill(~dead, -torch.inf).topk(
            min(self.config.aux_k, n_dead), dim=-1
        )
        auxiliary = torch.zeros_like(pre).scatter(-1, indices, values.relu())
        return F.mse_loss(
            self.decoder(auxiliary).float(), (x - reconstruction).detach().float()
        )

    @torch.no_grad()
    def record_activity(self, codes):
        """Advance optional AuxK bookkeeping only after a successful training step."""
        if self.config.aux_coefficient:
            vectors = codes.numel() // self.hidden_dim
            self.vectors_since_active.add_(vectors)
            self.vectors_since_active[
                (codes.reshape(-1, self.hidden_dim) != 0).any(0)
            ] = 0

    @torch.no_grad()
    def normalize_decoder(self):
        """Unit-normalize columns after each optimizer step."""
        self.decoder.weight.div_(
            self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8)
        )

    @torch.no_grad()
    def remove_parallel_grad(self):
        """Remove radial decoder gradients before taking the optimizer step."""
        w, g = self.decoder.weight, self.decoder.weight.grad
        if g is not None:
            direction = F.normalize(w, dim=0)
            g.sub_((g * direction).sum(0, keepdim=True) * direction)

    @torch.no_grad()
    def calibrate_threshold(self, calibration_activations):
        """Fit a global inference threshold on a bounded, non-final data panel.

        Select the midpoint between the N*k-th and next largest positive values.
        Ties and zeros can yield fewer active entries; report measured L0. This
        explicit calibration is not the reference implementation's training EMA.
        Returns the calibrated scalar without changing train/eval mode.
        """
        if self.config.variant not in {"batchtopk", "matryoshka"}:
            raise ValueError("Threshold calibration applies only to BatchTopK")
        values = self._preactivations(calibration_activations).relu().flatten()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite calibration activations")
        count = values.numel() // self.hidden_dim * self.k
        if count == values.numel():
            threshold = values.new_zeros(())
        else:
            boundary = values.topk(count + 1).values
            threshold = (boundary[-2] + boundary[-1]) / 2
        self.threshold.copy_(threshold)
        self.threshold_calibrated.fill_(True)
        return float(self.threshold)

    def metadata(self):
        """JSON-compatible architecture settings for toolkit checkpoints."""
        return asdict(self.config)
