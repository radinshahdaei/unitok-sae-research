"""Independent steering recipes using explicit discovery data and frozen edits.

No model, data, target, feature IDs, distortion budget or compute is inferred.
These functions expose experiment mechanics, not a benchmark or semantic claim.
"""

import torch

from unitok_sae.steering import (mean_feature_edit, mean_preserving_span,
                                 rotated_dictionary, svd_span)


def discovery_template(clean_codes, attacked_codes, decoder_columns, feature_ids):
    """Fit one [T,D] template from discovery donors, never evaluation donors."""
    if clean_codes.shape != attacked_codes.shape:
        raise ValueError("Paired discovery codes must have identical shapes")
    return mean_feature_edit(attacked_codes - clean_codes, decoder_columns, feature_ids)


def comparison_spans(discovery_deltas, decoder_columns, feature_ids, *, seed):
    """Build equal-rank SAE, rotated, Gaussian and mean-preserving PCA spans.

    IDs are supplied, not selected here. A fair selection comparison must search
    each dictionary with equal resources; this alone compares fixed spans.
    """
    selected, _ = svd_span(decoder_columns[:, feature_ids])
    rotated, _ = svd_span(rotated_dictionary(decoder_columns, seed=seed)[:, feature_ids])
    generator = torch.Generator(device=decoder_columns.device).manual_seed(seed)
    random = torch.randn(decoder_columns.shape[0], len(feature_ids),
                         generator=generator, device=decoder_columns.device,
                         dtype=decoder_columns.dtype)
    gaussian, _ = svd_span(random)
    data, _ = mean_preserving_span(discovery_deltas, len(feature_ids))
    spans = {"sae": selected, "rotated": rotated, "gaussian": gaussian, "data": data}
    rank = min(basis.shape[1] for basis in spans.values())
    if rank == 0:
        raise ValueError("At least one comparator has zero rank")
    return {name: basis[:, :rank] for name, basis in spans.items()}


def apply_template(session, images, *, block, edit):
    """Run a frozen [T,D] template through the native model, retaining gradients.

    Caller chooses inference mode for inspection or autograd for fitting/attacks.
    This additive intervention retains the recipient activation and SAE residual.
    """
    if edit.ndim != 2:
        raise ValueError("Expected a fixed [T,D] spatial template")
    return session.forward(images, activation_edits={block: lambda h: h + edit})
