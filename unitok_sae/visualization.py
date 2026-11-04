"""Small PIL-based inspection helpers; no notebook or plotting framework needed."""

import torch


def reconstruction_grid(images, baseline, edited):
    """Return a PIL grid: input, native reconstruction, edited reconstruction.

    Inputs are [0,1]; reconstructions are raw model [-1,1]-scale outputs. Rows are
    source images; clipping is for visualization only, never for stored metrics.
    """
    from torchvision.transforms.functional import to_pil_image

    if (
        images.shape != baseline.shape
        or images.shape != edited.shape
        or images.ndim != 4
    ):
        raise ValueError("Expected matching BCHW input/reconstruction tensors")
    rows = []
    for image, base, edit in zip(
        images.detach().cpu(), baseline.detach().cpu(), edited.detach().cpu()
    ):
        rows.append(
            torch.cat([image, (base + 1) / 2, (edit + 1) / 2], dim=-1).clamp(0, 1)
        )
    return to_pil_image(torch.cat(rows, dim=-2))


def feature_overlay(image, spatial_map, *, opacity=0.5, maximum=None):
    """Return a red heatmap over one CHW [0,1] image and explicit HxW feature map.

    ``maximum`` can be frozen across images for comparable intensity. Without it
    each overlay normalizes independently and is unsuitable for magnitude
    comparisons. Map geometry must be validated at the model hook beforehand.
    """
    from torchvision.transforms.functional import to_pil_image

    if (
        image.ndim != 3
        or image.shape[0] != 3
        or spatial_map.ndim != 2
        or not 0 <= opacity <= 1
        or not torch.isfinite(image).all()
        or not torch.isfinite(spatial_map).all()
        or (spatial_map < 0).any()
    ):
        raise ValueError(
            "Expected RGB image and nonnegative finite spatial feature map"
        )
    heat = torch.nn.functional.interpolate(
        spatial_map.detach().float()[None, None].cpu(),
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0]
    maximum = float(heat.max()) if maximum is None else float(maximum)
    if not torch.isfinite(torch.tensor(maximum)) or maximum < 0:
        raise ValueError("maximum must be finite and nonnegative")
    alpha = (heat / max(maximum, 1e-12)).clamp(0, 1) * opacity
    red = torch.zeros_like(image, device="cpu")
    red[0] = 1
    return to_pil_image((image.detach().cpu() * (1 - alpha) + red * alpha).clamp(0, 1))
