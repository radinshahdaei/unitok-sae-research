"""Per-image steering outcomes and class-balanced bounded-score uncertainty.

No alpha schedule, hypothesis, effect margin or confirmation index is inferred.
Freeze these choices before final evaluation. Repeated tensors, targets or seeds
on one image do not supply independent observations.
"""

import math
from collections import Counter, defaultdict


def _probability(value, name):
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must lie in (0,1)")


def steering_events(clean_pred, edited_pred, mse, *, target, distortion_cap,
                    roundtrip_pred=None):
    """Separate target classification, distortion and optional decoded recognition.

    Sequence inputs contain one entry per image in identical order. Predictions
    and target are nonnegative integer class IDs. MSE uses a recorded reference
    and raw scale; it is not a perceptual or semantic quality measurement.
    Already-target images remain in the denominator, with J_steer=0.
    J_roundtrip is R AND D; J_both is J_steer AND R.
    Unmeasured round-trip recognition is omitted, never recorded as zero.
    """
    n = len(clean_pred)
    if not n or len(edited_pred) != n or len(mse) != n:
        raise ValueError("Expected aligned nonempty per-image sequences")
    if type(target) is not int or target < 0:
        raise ValueError("target must be a nonnegative integer")
    if not math.isfinite(distortion_cap) or distortion_cap < 0:
        raise ValueError("distortion_cap must be finite and nonnegative")
    if roundtrip_pred is not None and len(roundtrip_pred) != n:
        raise ValueError("roundtrip_pred must align with the same images")
    predictions = [*clean_pred, *edited_pred]
    if roundtrip_pred is not None:
        predictions.extend(roundtrip_pred)
    if any(type(p) is not int or p < 0 for p in predictions):
        raise ValueError("Predictions must be nonnegative integer class IDs")
    if any(not math.isfinite(m) or m < 0 for m in mse):
        raise ValueError("MSE must be finite and nonnegative")
    c = [int(p == target) for p in edited_pred]
    d = [int(m <= distortion_cap) for m in mse]
    eligible = [int(p != target) for p in clean_pred]
    result = {"C": c, "D": d, "eligible": eligible,
              "J_steer": [a * b * e for a, b, e in zip(c, d, eligible)]}
    if roundtrip_pred is not None:
        r = [int(p == target) for p in roundtrip_pred]
        result.update(R=r, J_roundtrip=[a * b for a, b in zip(r, d)],
                      J_both=[a * b * z * e for a, b, z, e in zip(c, d, r, eligible)])
    return result


def per_image_scores(edit_scores):
    """Average frozen edit scores [edits][images] into one [0,1] score per image."""
    if not edit_scores or not edit_scores[0]:
        raise ValueError("Expected at least one edit and one image")
    n = len(edit_scores[0])
    if any(len(row) != n for row in edit_scores):
        raise ValueError("Every edit must use the same ordered images")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for row in edit_scores for v in row):
        raise ValueError("Scores must be finite and in [0,1]")
    return [sum(row[i] for row in edit_scores) / len(edit_scores) for i in range(n)]


def summarize_scores(scores, labels, *, sample_ids, alpha, difference=False):
    """Class-balanced estimate and one-sided Hoeffding bounds at explicit alpha.

    Provide one bounded score per *unique* image and its class label, or one
    paired difference per image when difference=True. Averages over fixed edits
    are allowed. Scores are in [0,1], paired differences in [-1,1]. Equal class
    weights give s=sum_c (1/K)^2/n_c and effective_n=1/s. Radius is
    sqrt(log(1/alpha)*s/2), doubled for a paired difference.

    The bound assumes independent image observations conditional on the fixed
    class frame and frozen edits. It does not capture fitting randomness or
    uncertainty from sampling new classes. alpha is per tail after the caller's
    multiplicity/sequential correction, not automatically a family error rate.
    """
    _probability(alpha, "alpha")
    n = len(scores)
    if not n or len(labels) != n or len(sample_ids) != n:
        raise ValueError("Scores, labels and IDs must align and be nonempty")
    if (any(not isinstance(ident, str) or not ident for ident in sample_ids)
            or len(set(sample_ids)) != n):
        raise ValueError("Provide one unique nonempty string ID per image")
    if any(type(label) is not int or label < 0 for label in labels):
        raise ValueError("Labels must be nonnegative integer class IDs")
    lower_range = -1 if difference else 0
    if any(not math.isfinite(v) or not lower_range <= v <= 1 for v in scores):
        raise ValueError("Scores lie outside the declared bounded range")
    counts, totals = Counter(labels), defaultdict(float)
    for label, value in zip(labels, scores):
        totals[label] += value
    class_means = {label: totals[label] / count for label, count in counts.items()}
    estimate = sum(class_means.values()) / len(counts)
    s = sum((1 / len(counts)) ** 2 / count for count in counts.values())
    radius = math.sqrt((2 if difference else .5) * math.log(1 / alpha) * s)
    return {"estimate": estimate, "radius": radius, "lower": estimate - radius,
            "upper": estimate + radius, "effective_n": 1 / s,
            "n_images": n, "n_classes": len(counts), "class_means": class_means,
            "alpha_per_tail": alpha, "difference": difference}


def required_effective_n(alternative, margin, *, alpha, beta, difference=True):
    """Sufficient effective image count for a prespecified bounded-score test.

    Uses the Hoeffding upper bound on failure to clear margin when the true
    mean is alternative: width**2/2 * (sqrt(log(1/alpha)) +
    sqrt(log(1/beta)))**2 / (alternative-margin)**2. Supply development-derived
    alternatives, never retrospective observed power. Return None when the
    alternative does not exceed the margin. This does not model class sampling
    or dependence between multiple fits; freeze that limitation in the design.
    """
    _probability(alpha, "alpha")
    _probability(beta, "beta")
    low = -1 if difference else 0
    if any(not math.isfinite(v) or not low <= v <= 1 for v in (alternative, margin)):
        raise ValueError("Alternative and margin must be within the score range")
    if alternative <= margin:
        return None
    factor = 2 if difference else .5
    return factor * (math.sqrt(math.log(1 / alpha)) + math.sqrt(math.log(1 / beta))) ** 2 / (alternative - margin) ** 2
