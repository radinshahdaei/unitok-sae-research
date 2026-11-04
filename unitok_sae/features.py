"""Feature edits in SAE coordinate space and inspectable spatial summaries.

No feature label is inferred here. A high activation or a visible decoded edit
does not establish a semantic concept. All edits are out-of-place and preserve
autograd; the model wrapper decides whether to preserve reconstruction residuals.
"""

import math

import torch


def edit_features(
    codes,
    feature_ids,
    *,
    operation="set",
    value=0.0,
    token_mask=None,
    allow_signed=False,
):
    """Set/add/scale selected codes, optionally at selected tokens.

    Args:
        codes: ``[B,T,F]`` SAE codes. No in-place mutation occurs.
        feature_ids: Unique integer IDs in this dictionary, not decoder dimensions.
        operation: ``set``, ``add`` or ``scale``. Ablation is set to zero.
        value: Scalar or tensor broadcastable to ``[B,T,len(feature_ids)]``.
            Setting accepts donor values, including per-image/per-token values.
        token_mask: Optional bool tensor broadcastable to ``[B,T,1]``. For a
            length-T mask pass ``mask[None,:,None]``. True selects edited tokens.

    Values must stay nonnegative unless allow_signed=True for a signed dictionary
    such as AbsTopK. The session independently checks the attached SAE's contract.
    """
    if codes.ndim != 3 or not torch.isfinite(codes).all():
        raise ValueError("Expected finite [B,T,F] codes")
    ids = list(feature_ids)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(type(j) is not int or not 0 <= j < codes.shape[-1] for j in ids)
    ):
        raise ValueError("feature_ids must be unique valid integer coordinates")
    previous = codes[..., ids]
    value = torch.as_tensor(value, dtype=codes.dtype, device=codes.device)
    if operation == "set":
        selected = torch.broadcast_to(value, previous.shape)
    elif operation == "add":
        selected = previous + value
    elif operation == "scale":
        selected = previous * value
    else:
        raise ValueError("operation must be set, add or scale")
    if token_mask is not None:
        if token_mask.dtype != torch.bool:
            raise ValueError("token_mask must be boolean")
        selected = torch.where(token_mask.to(codes.device), selected, previous)
    if not torch.isfinite(selected).all() or (
        not allow_signed and ((selected < 0).any() or (codes < 0).any())
    ):
        raise ValueError(
            "Signed values require allow_signed=True; all values must be finite"
        )
    result = codes.clone()
    result[..., ids] = selected
    return result


def top_features(codes, *, count=10, reduction="mean", absolute=False):
    """Rank by signed values or absolute magnitudes over tokens; IDs are layer-local.

    absolute=True avoids missing large negative AbsTopK features. Returned ranking
    values then describe magnitudes; inspect original codes for each feature's sign.
    """
    if codes.ndim != 3 or not 1 <= count <= codes.shape[-1]:
        raise ValueError("Expected [B,T,F] and a valid feature count")
    if reduction not in {"mean", "max"}:
        raise ValueError("reduction must be mean or max")
    values = codes.abs() if absolute else codes
    aggregate = values.mean(1) if reduction == "mean" else values.amax(1)
    return aggregate.topk(count, dim=-1)


def feature_map(codes, feature_id, *, grid_shape):
    """Reshape one coordinate into ``[B,H,W]`` using an explicit measured grid.

    Do not include special tokens or assume the report's 64-patch grid. An equal
    token count alone does not establish the correct spatial ordering.
    """
    if (
        codes.ndim != 3
        or type(feature_id) is not int
        or not 0 <= feature_id < codes.shape[-1]
        or len(grid_shape) != 2
        or any(type(n) is not int or n < 1 for n in grid_shape)
        or math.prod(grid_shape) != codes.shape[1]
    ):
        raise ValueError("Feature ID/grid must match the measured spatial tokens")
    return codes[..., feature_id].reshape(codes.shape[0], *grid_shape)
