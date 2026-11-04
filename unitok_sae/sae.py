"""Small trainable ReLU/L1, Top-K and BatchTopK SAE baselines.

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


@dataclass(frozen=True)
class SAEConfig:
    """Architecture/loss settings; L1 is mean per-vector sum of codes.

    Set a positive l1_coefficient for a sparse ReLU/L1 baseline. Zero deliberately
    permits an unregularized ReLU autoencoder control; it has no sparse budget.
    """

    input_dim: int
    hidden_dim: int
    variant: str = "topk"
    k: int = 64
    l1_coefficient: float = 0.0

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.variant not in {"relu", "topk", "batchtopk"}:
            raise ValueError("variant must be relu, topk or batchtopk")
        if self.k > self.hidden_dim:
            raise ValueError("k cannot exceed hidden_dim")
        if not math.isfinite(self.l1_coefficient) or self.l1_coefficient < 0:
            raise ValueError("l1_coefficient must be finite and nonnegative")


class SparseAutoencoder(nn.Module):
    """Operate on any ``[..., input_dim]`` tensor, without input normalization.

    ``encode`` returns (codes, None, None, preactivations), matching the existing
    adapter's tuple interface. ``decode`` includes pre_bias; ``decoder(delta)``
    does not, which is essential for residual-preserving edits.
    """

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
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
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder()

    def _preactivations(self, x):
        if x.ndim < 2 or x.shape[-1] != self.input_dim or x.numel() == 0:
            raise ValueError("Expected nonempty [..., input_dim] activations")
        return self.encoder(x - self.pre_bias)

    def encode(self, x):
        """Return differentiable exact sparse codes; Top-K support can jump."""
        pre = self._preactivations(x)
        positive = pre.relu()
        if self.config.variant == "relu":
            codes = positive
        elif self.config.variant == "batchtopk" and not self.training:
            if not bool(self.threshold_calibrated):
                raise RuntimeError("Calibrate BatchTopK threshold before evaluation")
            codes = positive * (positive > self.threshold)
        elif self.config.variant == "batchtopk":
            flat = positive.flatten()
            count = positive.numel() // self.hidden_dim * self.k
            values, indices = flat.topk(count)
            codes = (
                torch.zeros_like(flat).scatter(0, indices, values).reshape_as(positive)
            )
        else:
            values, indices = pre.topk(self.k, dim=-1)
            codes = torch.zeros_like(pre).scatter(-1, indices, values.relu())
        return codes, None, None, pre

    def decode(self, codes):
        """Map sparse codes to activation space, including the learned bias."""
        return self.decoder(codes) + self.pre_bias

    def forward(self, x):
        """Return reconstruction, codes, MSE, L1 magnitude and combined loss."""
        codes = self.encode(x)[0]
        reconstruction = self.decode(codes)
        mse = F.mse_loss(reconstruction.float(), x.float())
        l1 = codes.float().sum(-1).mean()
        return {
            "reconstruction": reconstruction,
            "latent": codes,
            "recon_loss": mse,
            "l1": l1,
            "loss": mse + self.config.l1_coefficient * l1,
        }

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
        if self.config.variant != "batchtopk":
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
