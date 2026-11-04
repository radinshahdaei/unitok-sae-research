"""UniTok tensor access and temporary, composable layer interventions.

Public image tensors are float32 RGB BCHW in [0,1]. Upstream receives x*2-1.
Returned image reconstructions remain raw [-1,1]-scale decoder outputs; they
can exceed that range. Clamp only when displaying. Methods preserve the caller's
gradient context. Use ``torch.inference_mode()`` yourself for inspection.
"""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


@dataclass
class LayerState:
    """Measured layer tensor and optional SAE decomposition, all graph-connected.

    activation/codes are the state before this layer's edit; edited_codes and
    edited_activation are the applied state. Later-layer captures already include
    upstream interventions. Captures without an SAE have no code tensors.
    """

    activation: torch.Tensor
    codes: torch.Tensor | None = None
    reconstruction: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    edited_codes: torch.Tensor | None = None
    edited_activation: torch.Tensor | None = None


@dataclass
class SAEAttachment:
    """Temporary SAE measurement/edit at one block.

    ``edit`` receives the original codes and returns new codes of equal shape.
    Default ``mode='residual'`` uses H + decoder(Z_new-Z). ``replacement`` uses
    sae.decode(Z_new), discarding the residual; even without an edit it can damage
    downstream outputs. SAE must be in eval mode. BatchTopK needs calibration.
    Multiple attached layers execute in encoder order; later states reflect
    earlier edits. Their effects are not independent or necessarily additive.
    """

    sae: torch.nn.Module
    edit: Callable[[torch.Tensor], torch.Tensor] | None = None
    mode: str = "residual"


@dataclass
class ForwardResult:
    """UniTok outputs plus measured tensors; release between attack steps.

    ``indices`` are [B,codebooks,T], ``quantized`` follows upstream's STE, and
    ``layers`` uses zero-based block IDs. Reconstructions are None when decoding
    is skipped. ``image_features`` are normalized text-aligned embeddings.
    """

    pre_quantization: torch.Tensor
    quantized: torch.Tensor
    indices: torch.Tensor
    image_features: torch.Tensor
    reconstruction: torch.Tensor | None
    layers: dict[int, LayerState] = field(default_factory=dict)


class UniTokSession:
    """Wrap an already loaded UniTok without changing its weights or mode.

    Do not run concurrent sessions/forwards on the same model: temporary hooks
    are model-global. Original quantizer usage counters update even in eval.
    ``image_size`` is explicit checkpoint metadata, not a resizing operation.
    """

    def __init__(self, model, *, image_size: int):
        if type(image_size) is not int or image_size < 1:
            raise ValueError("Explicit positive checkpoint image_size required")
        self.model, self.image_size = model, image_size
        self._running = False

    def _check_model(self):
        if any(module.training for module in self.model.modules()):
            raise ValueError("UniTok must be in evaluation mode")

    def _images(self, images):
        self._check_model()
        if (
            images.ndim != 4
            or images.shape[0] == 0
            or images.shape[1] != 3
            or images.shape[2:] != (self.image_size, self.image_size)
            or images.dtype != torch.float32
            or not torch.isfinite(images).all()
            or images.min() < 0
            or images.max() > 1
        ):
            raise ValueError(
                "Expected nonempty float32 [B,3,image_size,image_size] in [0,1]"
            )
        return images * 2 - 1

    def forward(self, images, *, layers=(), attachments=None, decode=True):
        """Encode/classify, optionally decode and capture/edit arbitrary blocks.

        ``layers`` captures native activations without requiring an SAE.
        ``attachments={15: SAEAttachment(sae, edit=...)}`` captures codes and
        applies edits before the remaining native encoder and quantizer.
        No hardcoded spatial token count or dictionary width is used.
        """
        pre, captured = self._encode(images, layers=layers, attachments=attachments)
        quantized = self.model.quantizer(pre)[0]
        post = self.model.post_quant_proj(quantized)
        features = F.normalize(
            self.model.projection(self.model.fc_norm(post.mean(1))), dim=-1
        )
        reconstruction = self.model.decoder(post).float() if decode else None
        indices = self.model.quantizer.f_to_idx(pre)
        return ForwardResult(
            pre, quantized, indices, features, reconstruction, captured
        )

    def pre_quantization(self, images, *, attachments=None):
        """Return differentiable [B,T,D] vectors at the native quantizer input.

        Runs the encoder and quant_proj only, with the same optional SAE edits as
        forward(). It does not execute codebooks, the classification projection,
        the image decoder or quantizer usage-counter updates. This is the space
        used by the tokenizer embedding-distance attacks of arXiv:2602.18252v1.
        """
        return self._encode(images, attachments=attachments)[0]

    def _encode(self, images, *, layers=(), attachments=None):
        """Shared encoder path; all temporary SAE hooks are removed on exit."""
        inputs = self._images(images)
        attachments = {} if attachments is None else dict(attachments)
        requested = set(layers) | set(attachments)
        for block in requested:
            if type(block) is not int or not 0 <= block < len(
                self.model.encoder.blocks
            ):
                raise ValueError("Layer must be a zero-based encoder block index")
        if self._running:
            raise RuntimeError("A session cannot run nested/concurrent forwards")
        captured = {}

        def hook_for(block):
            def hook(_module, _args, activation):
                if block in captured:
                    raise RuntimeError(
                        "Block executed twice; gradient checkpointing is unsupported here"
                    )
                if not isinstance(activation, torch.Tensor) or activation.ndim != 3:
                    raise ValueError("Expected block output [B,T,D]")
                state = LayerState(activation=activation, edited_activation=activation)
                captured[block] = state
                if block not in attachments:
                    return None
                spec = attachments[block]
                if spec.mode not in {"residual", "replacement"}:
                    raise ValueError("SAE mode must be residual or replacement")
                if spec.sae.training:
                    raise ValueError("Attached SAE must be in evaluation mode")
                if getattr(spec.sae, "checkpoint_block", block) != block:
                    raise ValueError("SAE checkpoint belongs to another layer")
                if activation.shape[-1] != spec.sae.input_dim:
                    raise ValueError("SAE width does not match layer width")
                state.codes = spec.sae.encode(activation)[0]
                state.reconstruction = spec.sae.decode(state.codes)
                state.residual = activation - state.reconstruction
                # Protect original state against user callbacks that edit in place.
                changed = (
                    state.codes if spec.edit is None else spec.edit(state.codes.clone())
                )
                if (
                    not isinstance(changed, torch.Tensor)
                    or changed.shape != state.codes.shape
                    or changed.dtype != state.codes.dtype
                    or changed.device != state.codes.device
                    or not torch.isfinite(changed).all()
                    or (
                        getattr(spec.sae, "nonnegative_codes", True)
                        and (changed < 0).any()
                    )
                ):
                    raise ValueError(
                        "Edit must preserve shape/dtype/device and the SAE's finite/sign contract"
                    )
                state.edited_activation = (
                    activation + spec.sae.decoder(changed - state.codes)
                    if spec.mode == "residual"
                    else spec.sae.decode(changed)
                )
                state.edited_codes = changed
                return state.edited_activation

            return hook

        self._running = True
        try:
            with ExitStack() as stack:
                for block in sorted(requested):
                    handle = self.model.encoder.blocks[block].register_forward_hook(
                        hook_for(block)
                    )
                    stack.callback(handle.remove)
                tokens = self.model.encoder(inputs).float()
                pre = self.model.quant_proj(tokens)
            if set(captured) != requested:
                raise RuntimeError("Forward did not execute all requested layers")
            return pre, captured
        finally:
            self._running = False

    def decode_tokens(self, indices):
        """Decode integer [B,codebooks,T] indices through the native image decoder.

        Hard indexing has no image gradient. This method validates ID ranges;
        the upstream decoder validates its configured spatial token count.
        """
        self._check_model()
        q = self.model.quantizer
        if (
            indices.ndim != 3
            or indices.numel() == 0
            or indices.dtype != torch.int64
            or indices.shape[1] != len(q.codebooks)
        ):
            raise ValueError("Expected nonempty int64 [B,codebooks,T] indices")
        for i, quantizer in enumerate(q.codebooks):
            if indices[:, i].min() < 0 or indices[:, i].max() >= quantizer.vocab_size:
                raise ValueError("Token ID outside codebook")
        return self.model.decoder(
            self.model.post_quant_proj(q.idx_to_f(indices))
        ).float()

    def logits(self, images, text_features, *, attachments=None, use_logit_scale=True):
        """Differentiable zero-shot scores, omitting the unused image decoder.

        text_features must be normalized [classes,embedding_dim], ordered by the
        caller's recorded class vocabulary. ImageNet attacks need all 1000 rows.
        Gradients follow upstream hard-forward/STE-backward quantization.
        """
        features = self.forward(
            images, attachments=attachments, decode=False
        ).image_features
        if (
            text_features.ndim != 2
            or text_features.shape[1] != features.shape[1]
            or text_features.shape[0] < 2
            or not torch.isfinite(text_features).all()
            or not torch.allclose(
                text_features.norm(dim=-1),
                torch.ones_like(text_features[:, 0]),
                atol=1e-4,
            )
        ):
            raise ValueError(
                "Expected finite unit-normalized [classes,embedding_dim] text prototypes"
            )
        scores = features @ text_features.T
        return scores * self.model.logit_scale.exp() if use_logit_scale else scores

    @torch.no_grad()
    def text_prototypes(self, class_names, tokenize, *, templates=("a photo of a {}",)):
        """Encode explicit class names/templates in supplied order, one class at a time.

        Pass upstream's tokenizer from ``loading.load_unitok``. No vocabulary is
        guessed or downloaded. Template embeddings are individually normalized,
        then averaged and normalized again. Record class order and templates.
        """
        self._check_model()
        if len(class_names) < 2 or not templates:
            raise ValueError("At least two classes and one template required")
        device = next(self.model.parameters()).device
        rows = []
        for name in class_names:
            tokens = tokenize([template.format(name) for template in templates]).to(
                device
            )
            features = self.model.encode_text(tokens, normalize=True)
            rows.append(F.normalize(features.mean(0), dim=0))
        return torch.stack(rows)


def preprocess_image(image, *, image_size):
    """PIL RGB -> float32 CHW [0,1], bilinear square resize, antialias enabled.

    This matches the existing smoke preprocessing. It intentionally does not
    apply ImageNet mean/std, center cropping, or CLIP preprocessing.
    """
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize, to_tensor

    return to_tensor(
        resize(
            image.convert("RGB"),
            [image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
    )
