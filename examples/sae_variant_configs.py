"""Build architecture configurations without choosing data, seeds or a pipeline.

All coefficients/thresholds/prefixes are explicit arguments: published values
cannot be transferred blindly to raw UniTok activations. Tune on calibration,
then compare measured sparsity and fidelity, not equal coefficients. Set the
same caller-controlled seed before each constructor if comparing initializations.
"""

from unitok_sae import SAE_VARIANTS, SAEConfig, SparsitySchedule


def variant_configs(
    *, input_dim, hidden_dim, k, l1, l0, threshold, bandwidth, prefixes
):
    """Return eight configs, ready for SparseAutoencoder(config).to(device).

    The Matryoshka prefixes must end at hidden_dim. Multi-TopK uses the documented
    k/4k, 1/8 weighting defaults. These are independent configurations; they do
    not imply equal realized L0, training compute, or feature identities.
    """
    result = {}
    for variant in SAE_VARIANTS:
        options = {}
        if variant in {"relu", "gated"}:
            options["l1_coefficient"] = l1
        if variant == "jumprelu":
            options.update(
                l0_coefficient=l0, initial_threshold=threshold, bandwidth=bandwidth
            )
        if variant == "matryoshka":
            options["prefixes"] = prefixes
        result[variant] = SAEConfig(input_dim, hidden_dim, variant, k, **options)
    return result


def curriculum_examples(*, warmup_steps, initial_k, final_k, anneal_steps, final_p):
    """Return separate schedule ablations; no schedule is enabled automatically.

    Use warmup on ReLU/Gated/JumpReLU, k curriculum on Top-K families, and Lp
    annealing only on ReLU/Gated. For continuation record both the schedule and
    start_step. This does not reproduce adaptive-coefficient p-annealing.
    """
    return {
        "warmup": SparsitySchedule(warmup_steps=warmup_steps),
        "k_curriculum": SparsitySchedule(
            k_start=initial_k, k_end=final_k, k_steps=anneal_steps
        ),
        "lp_annealing": SparsitySchedule(
            p_start=1.0, p_end=final_p, p_steps=anneal_steps
        ),
    }
