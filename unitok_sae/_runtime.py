"""Self-contained source/asset verification for the pinned executable UniTok.

Relative asset/source paths resolve against the current working directory.
Patches ship as package data, so wheel installs need no private research tree.
No implicit downloads; the external checkout remains separately provisioned.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

REVISION = "3a22e697e8c6586e8c270808458f17b24d8a575f"
PATCHES_DIR = Path(__file__).parent / "patches"
SOURCE_SUFFIXES = {
    ".py",
    ".pyi",
    ".pyw",
    ".pyc",
    ".pyo",
    ".pyx",
    ".pxd",
    ".so",
    ".pyd",
    ".dylib",
    ".dll",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cu",
    ".cuh",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ipynb",
}


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else path.resolve()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_diff(text):
    """Compare diffs ignoring per-line trailing whitespace (blank context lines)."""
    return "\n".join(line.rstrip() for line in text.splitlines())


def _untracked_source(repo):
    # Include ignored files: .gitignore does not prevent a file from being imported.
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--others", "-z"]
    ).split(b"\0")
    source = []
    for raw_name in filter(None, names):
        name = os.fsdecode(raw_name)
        path = repo / name
        # Normal import caches are generated from tracked source. Standalone .pyc
        # files, unlike __pycache__ entries, can supply sourceless Python modules.
        if (
            not path.is_symlink()
            and "__pycache__" in path.relative_to(repo).parts
            and path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if (
            path.is_symlink()
            or path.suffix.lower() in SOURCE_SUFFIXES
            or path.is_dir()
            or (path.is_file() and path.stat().st_mode & 0o111)
        ):
            source.append(name)
    return sorted(source)


def verify_repo(value):
    repo = resolve(value)
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != REVISION:
        raise ValueError(f"Implementation HEAD must be {REVISION}, got {head}")
    diff_args = [
        "git",
        "-C",
        str(repo),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--no-renames",
        "--binary",
        "--abbrev=7",
    ]
    # HEAD includes staged edits as well as changes left in the working tree.
    diff = subprocess.check_output([*diff_args, "HEAD", "--"], text=True)
    recorded = "\n".join(
        p.read_text() for p in sorted(PATCHES_DIR.glob("*.patch")) if p.is_file()
    )
    if _normalize_diff(diff) != _normalize_diff(recorded):
        raise ValueError(
            "Checkout changes do not match the recorded patches/ files; "
            "record/review any patch before using the checkout"
        )
    staged = set(
        subprocess.check_output(
            [*diff_args, "--cached", "--name-only", "-z", "--"]
        ).split(b"\0")
    )
    unstaged = set(
        subprocess.check_output([*diff_args, "--name-only", "-z", "--"]).split(b"\0")
    )
    mixed = sorted(os.fsdecode(name) for name in (staged & unstaged) if name)
    if mixed:
        raise ValueError("Staged and working-tree versions differ: " + ", ".join(mixed))
    untracked = _untracked_source(repo)
    if untracked:
        raise ValueError(
            "Untracked executable/import source is not recorded: "
            + ", ".join(untracked)
        )
    for name in (
        "models/unitok.py",
        "sae/model.py",
        "open_clip/bpe_simple_vocab_16e6.txt.gz",
    ):
        if not (repo / name).is_file():
            raise ValueError(f"Incomplete executable checkout: {name}")
    return repo


def import_original(value):
    repo = verify_repo(value)
    for name in ("models", "utils", "sae", "open_clip"):
        loaded = sys.modules.get(name)
        if loaded and getattr(loaded, "__file__", None):
            if not Path(loaded.__file__).resolve().is_relative_to(repo):
                raise ValueError(f"Conflicting imported package: {name}")
    # Model constructors must not retrieve anything from the network.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    sys.path.insert(0, str(repo))
    from models.unitok import UniTok
    from open_clip.tokenizer import tokenize
    from sae.model import TopKSAE
    from utils.config import Args

    return UniTok, TopKSAE, Args, tokenize


def load_state_audited(module, state_dict):
    """Fail on any unmatched key; caller must resolve an exception explicitly.

    Loading can mutate matched parameters before failing. Discard the model after
    a failure; do not continue with a partly loaded object.
    """
    incompatible = module.load_state_dict(state_dict, strict=False)
    audit = {
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }
    if audit["missing"] or audit["unexpected"]:
        raise ValueError(f"Checkpoint keys require review: {audit}")
    return audit


def select_state(checkpoint, keys, strip_prefix=False):
    import torch

    state = checkpoint
    for key in keys:
        state = state[key]
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()
        )
    ):
        raise ValueError("Selected checkpoint branch is not a tensor state dict")
    if strip_prefix:
        renamed = {key.removeprefix("module."): value for key, value in state.items()}
        if len(renamed) != len(state):
            raise ValueError("DDP prefix stripping collides with existing keys")
        state = renamed
    return state
