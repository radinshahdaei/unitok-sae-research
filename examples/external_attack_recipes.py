"""External attack calls over a verified, already loaded UniTok session.

These functions neither choose a cohort nor run on import. Install the optional
requirements explicitly, and supply raw [0,1] images plus all 1000 ordered text
prototypes for ImageNet. Save the returned settings and model/prototype identities.
The fixed-target PGD route is distinct from AutoAttack's internal target search.
"""

from unitok_sae.external_attacks import (
    AutoAttackConfig,
    UniTokClassifier,
    run_autoattack,
    run_foolbox,
)


def official_evaluation(session, prototypes, images, true_labels, *, epsilon, norm):
    """Run the unchanged official standard profile; this can be expensive."""
    model = UniTokClassifier(session, prototypes, max_batch_size=1)
    return run_autoattack(
        model,
        images,
        true_labels,
        AutoAttackConfig(epsilon=epsilon, norm=norm, version="standard", batch_size=1),
    )


def explicit_target_attack(
    session, prototypes, images, true_labels, target_labels, *, epsilon, steps
):
    """Target per-image classes using upstream Linf PGD; no shared delta is fitted."""
    model = UniTokClassifier(session, prototypes, max_batch_size=1)
    return run_foolbox(
        model,
        images,
        true_labels,
        targets=target_labels,
        attack="pgd_linf",
        epsilon=epsilon,
        attack_kwargs={"steps": steps, "abs_stepsize": epsilon / 4},
        seed=0,
    )


def explicit_component_check(
    session, prototypes, images, true_labels, *, epsilon, steps, max_evaluations
):
    """Run a custom APGD-DLR check with a hard image-forward cap, not standard AA."""
    model = UniTokClassifier(session, prototypes, max_batch_size=1)
    config = AutoAttackConfig(
        epsilon=epsilon,
        version="custom",
        attacks=("apgd-dlr",),
        iterations=steps,
        restarts=1,
        batch_size=1,
    )
    return run_autoattack(
        model, images, true_labels, config, max_evaluations=max_evaluations
    )
