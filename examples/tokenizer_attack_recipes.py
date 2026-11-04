"""Composable paper-based tokenizer methods; nothing runs at import.

These implement objectives from arXiv:2602.18252v1 using installed upstream APGD.
They are not an official author release or a complete reproduction/training run.
Supply verified loaded sessions, raw [0,1] images and explicit pixel budgets.
"""

import torch

from unitok_sae.tokenizer_attacks import (
    EmbeddingAttackConfig,
    UniTokEmbedding,
    attack_embeddings,
    embedding_consistency_loss,
    token_change_counts,
)


def distort_tokenizer(session, images, *, epsilon, steps=100, attachments=None):
    """Unlabeled per-image pixel attack maximizing tokenizer embedding distance."""
    return attack_embeddings(
        UniTokEmbedding(session, attachments=attachments),
        images,
        EmbeddingAttackConfig(epsilon=epsilon, steps=steps),
    )


def match_target_image(session, images, target_images, *, epsilon, steps=100):
    """Match a single target image for all sources, or one target per source.

    A target-image objective is distinct from a target-class/caption objective.
    Lower embedding distance does not guarantee the downstream target is reached.
    """
    extractor = UniTokEmbedding(session)
    with torch.no_grad():
        reference = extractor(target_images).detach()
    return attack_embeddings(
        extractor,
        images,
        EmbeddingAttackConfig(epsilon=epsilon, steps=steps),
        mode="match",
        reference_features=reference,
    )


def frozen_reference_loss(
    student_session, original_teacher_session, clean_images, *, epsilon, steps=10
):
    """Return inner attack and outer differentiable loss; caller owns the update.

    Teacher must be a SEPARATE, frozen original model; never pass the student as
    its own teacher. Both use eval mode (input and parameter gradients still work).
    Caller explicitly chooses/freeze-audits the trainable encoder parameters and
    supplies the optimizer. No codebook/decoder updates happen in this helper.
    This illustrates Eq.(2), not the paper's complete adversarial-training recipe.
    """
    if student_session.model is original_teacher_session.model:
        raise ValueError("Student and frozen original teacher must be separate models")
    if any(p.requires_grad for p in original_teacher_session.model.parameters()):
        raise ValueError("Freeze the original teacher explicitly before calling")
    student = UniTokEmbedding(student_session)
    teacher = UniTokEmbedding(original_teacher_session)
    with torch.no_grad():
        reference = teacher(clean_images).detach()
    result = attack_embeddings(
        student,
        clean_images,
        EmbeddingAttackConfig(epsilon=epsilon, steps=steps),
        reference_features=reference,
        mode="diverge",
    )
    # Attack temporarily freezes/restores parameters. Recompute for the outer loss.
    loss = embedding_consistency_loss(student(result.images), reference)
    return result, loss


@torch.no_grad()
def inspect_hard_token_changes(session, clean_images, attacked_images):
    """Measure actual codebook/position disagreement using the native hard forward.

    Classification and reconstruction effects require their own downstream checks;
    a high disagreement fraction is not evidence of a successful semantic attack.
    """
    clean = session.forward(clean_images)
    attacked = session.forward(attacked_images)
    return token_change_counts(clean.indices, attacked.indices)
