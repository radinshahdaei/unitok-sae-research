"""Spatial activation edits and matched span controls, independent of a model.

Feature IDs refer to a particular decoder and layer. A q-direction span permits
T*q spatial coefficients, not just q scalar values. Re-encoding an edited state
need not confine its SAE response to the selected coordinates.
"""

import math

import torch


def _matrix(value, name):
    if (value.ndim != 2 or min(value.shape) < 1
            or value.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite nonempty float32/float64 matrix")


def _ids(ids, width):
    ids = list(ids)
    if (not ids or len(set(ids)) != len(ids)
            or any(type(j) is not int or not 0 <= j < width for j in ids)):
        raise ValueError("Feature IDs must be unique valid integer decoder columns")
    return ids


@torch.no_grad()
def svd_span(columns, *, rtol=None):
    """Return (orthonormal basis, rank metadata) for actual [D,q] columns.

    Factorization uses float64; output preserves the input dtype and device.
    The default relative threshold is max(shape)*eps(input dtype). A deficient
    group loses rank rather than receiving arbitrary QR-completion directions.
    An all-zero matrix returns a [D,0] basis. Record rtol for reproducibility.
    """
    _matrix(columns, "columns")
    rtol = max(columns.shape) * torch.finfo(columns.dtype).eps if rtol is None else rtol
    if not math.isfinite(rtol) or not 0 <= rtol < 1:
        raise ValueError("rtol must be finite and in [0,1)")
    u, singular, _ = torch.linalg.svd(columns.double(), full_matrices=False)
    keep = singular > singular[0] * rtol
    basis = u[:, keep].to(columns.dtype)
    return basis, {
        "rank": int(keep.sum()), "requested": columns.shape[1], "rtol": float(rtol),
        "condition": float(singular[0] / singular[keep][-1]) if keep.any() else None,
    }


@torch.no_grad()
def mean_preserving_span(deltas, q, *, rtol=None):
    """Mean direction plus up to q-1 centered PCs of discovery [N,T,D] deltas.

    A redundant or zero mean can reduce rank. No additional directions are
    supplied to fill that capacity. q=1 uses only the mean. Fit on discovery
    images and freeze before evaluation; PCA does not certify causal usefulness.
    """
    if deltas.ndim != 3 or min(deltas.shape) < 1:
        raise ValueError("Expected nonempty [N,T,D] discovery deltas")
    flat = deltas.reshape(-1, deltas.shape[-1])
    _matrix(flat, "deltas")
    if type(q) is not int or not 1 <= q <= flat.shape[1]:
        raise ValueError("q must be between 1 and D")
    rtol = max(flat.shape) * torch.finfo(flat.dtype).eps if rtol is None else rtol
    mean = flat.double().mean(0)
    _, singular, vh = torch.linalg.svd(flat.double() - mean, full_matrices=False)
    keep = singular > singular[0] * rtol
    pcs = vh[keep][:q - 1].T
    mean = mean / mean.norm().clamp_min(torch.finfo(torch.float64).tiny)
    columns = torch.cat([mean[:, None], pcs], dim=1).to(flat.dtype)
    return svd_span(columns, rtol=rtol)


def mean_feature_edit(code_delta, decoder_columns, feature_ids):
    """Decode selected discovery code changes and average images, preserving tokens.

    Inputs are [N,T,F] clean-to-attacked changes and [D,F] linear decoder
    weights, with matching dtype/device. No decoder bias is added to a change.
    Returns a reusable [T,D] edit; no evaluation-image donor is consulted.
    """
    _matrix(decoder_columns, "decoder_columns")
    if (code_delta.ndim != 3 or min(code_delta.shape) < 1
            or code_delta.shape[-1] != decoder_columns.shape[1]
            or code_delta.dtype != decoder_columns.dtype
            or code_delta.device != decoder_columns.device
            or not torch.isfinite(code_delta).all()):
        raise ValueError("Expected finite matching [N,T,F] code changes")
    ids = _ids(feature_ids, decoder_columns.shape[1])
    return code_delta[..., ids].mean(0) @ decoder_columns[:, ids].T


def project_edit(edit, basis):
    """Least-squares projection of [T,D] into an orthonormal [D,r] span.

    Returns (projected [T,D] edit, [T,r] coefficients). Zero-rank bases are
    accepted. Use the same neutral edit to initialize all compared geometries.
    """
    _matrix(edit, "edit")
    if (basis.ndim != 2 or basis.shape[0] != edit.shape[1]
            or basis.dtype != edit.dtype or basis.device != edit.device
            or not torch.isfinite(basis).all()):
        raise ValueError("basis must be finite [D,r] with matching dtype/device")
    identity = torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device)
    if not torch.allclose(basis.T @ basis, identity, atol=1e-5, rtol=1e-5):
        raise ValueError("basis must be orthonormal")
    coefficients = edit @ basis
    return coefficients @ basis.T, coefficients


@torch.no_grad()
def rotated_dictionary(columns, *, seed):
    """Rotate [D,F] decoder directions, preserving their Gram matrix and norms.

    Uses a local generator; does not alter the global RNG. Column normalization,
    if desired, must be applied to both the original and rotated dictionaries.
    """
    _matrix(columns, "columns")
    generator = torch.Generator(device=columns.device).manual_seed(seed)
    noise = torch.randn(columns.shape[0], columns.shape[0], generator=generator,
                        device=columns.device, dtype=columns.dtype)
    rotation, _ = torch.linalg.qr(noise)
    return rotation @ columns
