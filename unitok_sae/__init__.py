"""Composable UniTok/SAE tools; importing this package never downloads assets.

Start with ``UniTokSession`` for an already loaded eval-mode UniTok. Model and
SAE loading are explicit functions in :mod:`unitok_sae.loading`; dataset writing,
training, attacks and plotting are independent modules. Images at this API's
boundary are float RGB BCHW tensors in [0, 1], unlike upstream's [-1, 1].
"""

from .features import edit_features, feature_map, top_features
from .model import ForwardResult, LayerState, SAEAttachment, UniTokSession
from .sae import SAEConfig, SparseAutoencoder

__all__ = [
    "UniTokSession",
    "SAEAttachment",
    "ForwardResult",
    "LayerState",
    "SparseAutoencoder",
    "SAEConfig",
    "edit_features",
    "feature_map",
    "top_features",
]
