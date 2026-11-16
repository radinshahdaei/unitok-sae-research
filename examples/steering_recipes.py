"""Independent steering recipes using explicit discovery data and frozen edits.

No model, data, target, feature IDs, distortion budget or compute is inferred.
These functions expose experiment mechanics, not a benchmark or semantic claim.
"""

import torch
from torch.nn import functional as F

from unitok_sae.control import fit_span_edit
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


def fit_template(session, discovery_images, prototypes, *, block, basis,
                 neutral_edit, target, distortion_cap, max_norm, steps,
                 penalty_weight=1.0, logit_scale=1.0):
    """Fit on discovery images with a common native-reconstruction penalty.

    prototypes are normalized [classes,embedding_dim] in recorded class order.
    logit_scale is an explicit multiplier, held fixed across comparisons. The
    small recipe runs all discovery images together; profile/batch larger jobs
    in the caller. Freeze model parameters, use eval mode and enable autograd.
    Native reconstruction MSE uses raw decoder units, not [0,1] display pixels.
    """
    with torch.no_grad():
        clean = session.forward(discovery_images, decode=False)
        clean_recon = session.decode_tokens(clean.indices)
        clean_pred = (clean.image_features @ prototypes.T).argmax(1)
    labels = torch.full((len(discovery_images),), target, dtype=torch.long,
                        device=discovery_images.device)

    def objective(edit):
        result = apply_template(session, discovery_images, block=block, edit=edit)
        logits = (result.image_features @ prototypes.T) * logit_scale
        mse = (result.reconstruction - clean_recon).flatten(1).square().mean(1)
        return F.cross_entropy(logits, labels), mse

    def measure(edit):
        result = apply_template(session, discovery_images, block=block, edit=edit)
        hard_recon = session.decode_tokens(result.indices)
        pred = (result.image_features @ prototypes.T).argmax(1)
        success = pred.eq(target) & clean_pred.ne(target)
        mse = (hard_recon - clean_recon).flatten(1).square().mean(1)
        return success, mse

    return fit_span_edit(basis, neutral_edit, objective, measure,
                         distortion_cap=distortion_cap, max_norm=max_norm, steps=steps,
                         penalty_weight=penalty_weight)
