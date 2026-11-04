"""Explicit audited asset loading for the pinned external UniTok source.

No downloads, fallback checkpoints, implicit DDP stripping or unsafe pickle
loading. Source-verification helpers and compatibility patches are packaged with
the toolkit. Relative paths resolve against the current working directory; JSON
config file location does not change path resolution.
"""

import torch

from ._runtime import import_original, load_state_audited, resolve, select_state, sha256
from .model import UniTokSession


def checked_asset(spec):
    """Resolve {path, sha256, provider}; fail before deserializing mismatched bytes."""
    if not spec.get("provider") or not spec.get("sha256"):
        raise ValueError("Checkpoint requires provider and expected SHA-256")
    path = resolve(spec["path"])
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {path}")
    return path


def load_unitok(config, *, device="cpu"):
    """Return (session, tokenizer, audit) from the existing explicit model config.

    Required keys: implementation_repo, model_args, model_args_provenance,
    unitok_checkpoint ({path,sha256,provider,state_dict_path}). Paths resolve
    relative to this repository root. Model arguments must come from the
    checkpoint's own configuration. Architecture construction never downloads.
    Device selection never silently falls back. Float32; grad checkpointing off.
    """
    if not config.get("model_args_provenance"):
        raise ValueError("Record model_args_provenance")
    spec = config["unitok_checkpoint"]
    if not isinstance(spec.get("state_dict_path"), list):
        raise ValueError(
            "Explicit state_dict_path required, including [] for raw weights"
        )
    path = checked_asset(spec)
    UniTok, _, Args, tokenize = import_original(config["implementation_repo"])
    args = Args(explicit_bool=True).parse_args(known_only=True, args=[])
    required = {
        "model",
        "img_size",
        "num_query",
        "quant_proj",
        "vocab_size",
        "vocab_width",
        "num_codebooks",
        "vq_beta",
        "le",
        "e_temp",
        "embed_dim",
        "text_width",
        "text_heads",
        "text_layers",
        "text_vocab_size",
        "text_context_length",
        "drop_path",
    }
    if set(config["model_args"]) != required:
        raise ValueError(
            "Supply all reviewed model_args explicitly; no implicit architecture defaults"
        )
    for key, value in config["model_args"].items():
        setattr(args, key, value)
    args.grad_ckpt, args.device = False, str(device)
    model = UniTok(args)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    audit = load_state_audited(
        model,
        select_state(
            checkpoint, spec["state_dict_path"], spec.get("strip_module_prefix", False)
        ),
    )
    model.eval().requires_grad_(False).to(device)
    return UniTokSession(model, image_size=args.img_size), tokenize, audit


def load_pretrained_sae(
    spec, *, implementation_repo, block, block_provenance=None, device="cpu"
):
    """Load the original author's TopKSAE class, retaining checkpoint semantics.

    Missing block metadata needs independently recorded block_provenance (the
    author block-15 checkpoint omits it). Returns (sae, audit). The loaded object
    carries checkpoint_block so attaching it to another layer fails explicitly.
    """
    path = checked_asset(spec)
    _, TopKSAE, _, _ = import_original(implementation_repo)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    actual = checkpoint.get("block")
    if actual is None:
        if not block_provenance:
            raise ValueError(
                "Missing checkpoint block: independently verified block_provenance required"
            )
        actual = block
    if type(block) is not int or block < 0 or actual != block:
        raise ValueError("SAE block mismatch")
    sae = TopKSAE(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["k"])
    audit = load_state_audited(sae, checkpoint["sae"])
    sae.checkpoint_block = block
    sae.eval().requires_grad_(False).to(device)
    return sae, audit
