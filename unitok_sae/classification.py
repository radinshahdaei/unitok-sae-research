"""Audit an explicit class vocabulary before running classification experiments.

Diagnostics do not modify prompts or prototypes. Save their output together with
the class order, templates, tokenizer identity and checkpoint provenance.
"""

import math

import torch
from torch.nn import functional as F


def _collisions(rows):
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row, []).append(index)
    return [ids for ids in groups.values() if len(ids) > 1]


@torch.no_grad()
def audit_classifier(prototypes, *, prompts=None, tokens=None, near_cosine=0.9999):
    """Return JSON-ready collision and self-query diagnostics, without encoding.

    ``prototypes`` is finite floating [classes, embedding_dim], with nonzero
    rows in the recorded class order. ``prompts`` optionally contains one exact
    string per class; ``tokens`` contains the corresponding integer [classes,L]
    tokenizer output, including padding. For multiple templates audit each
    template separately. Similarity uses float64 normalized CPU copies and costs
    O(classes**2) memory. No model or tokenizer is inferred or downloaded.

    Exact duplicate prototypes make later class IDs unreachable under the usual
    first-index argmax. Near duplicates and self-query failures are diagnostics,
    not estimates of image classification accuracy or an automatic repair rule.
    """
    if (prototypes.ndim != 2 or min(prototypes.shape) < 1
            or prototypes.shape[0] < 2 or not prototypes.is_floating_point()
            or not torch.isfinite(prototypes).all()):
        raise ValueError("Expected finite floating [classes, embedding_dim] prototypes")
    if not math.isfinite(near_cosine) or not -1 <= near_cosine <= 1:
        raise ValueError("near_cosine must be in [-1,1]")
    rows = prototypes.detach().double().cpu()
    norms = rows.norm(dim=1)
    if (norms == 0).any():
        raise ValueError("Prototype rows must be nonzero")
    count = len(rows)
    if prompts is not None and (len(prompts) != count
                               or any(not isinstance(p, str) for p in prompts)):
        raise ValueError("Provide one prompt string per class")
    if tokens is not None and (tokens.ndim != 2 or tokens.shape[0] != count
                              or tokens.shape[1] == 0
                              or tokens.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Expected integer [classes,L] token rows")
    normalized = F.normalize(rows, dim=1)
    scores = normalized @ normalized.T
    self_predicted = scores.argmax(1)
    off_diagonal = scores.clone()
    off_diagonal.fill_diagonal_(-float("inf"))
    pairs = torch.triu(off_diagonal >= near_cosine, diagonal=1).nonzero().tolist()
    return {
        "classes": count,
        "norm_range": [float(norms.min()), float(norms.max())],
        "prompt_collisions": None if prompts is None else _collisions(prompts),
        "token_collisions": (None if tokens is None else
                             _collisions(tuple(r) for r in tokens.cpu().tolist())),
        "exact_duplicate_groups": _collisions(tuple(r) for r in rows.tolist()),
        "near_cosine_threshold": float(near_cosine),
        "near_duplicate_pairs": pairs,
        "max_off_diagonal_cosine": float(off_diagonal.max()),
        "self_predicted": self_predicted.tolist(),
        "self_query_failures": (self_predicted != torch.arange(count)).nonzero()
                               .flatten().tolist(),
    }
