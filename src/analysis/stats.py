import math

import scipy.stats as scipy_stats


def wilson_ci(n_successes: int, n_total: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    :param n_successes: Number of successes (e.g. flagged claims).
    :param n_total: Total number of trials.
    :param confidence: Confidence level (default 0.95).
    :return: (lower, upper) bounds as proportions; (0.0, 0.0) if n_total == 0.
    """
    if n_total == 0:
        return (0.0, 0.0)
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = n_successes / n_total
    denominator = 1 + z ** 2 / n_total
    center = (p_hat + z ** 2 / (2 * n_total)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n_total + z ** 2 / (4 * n_total ** 2))) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def two_proportion_z_test(n1: int, total1: int, n2: int, total2: int) -> float:
    """
    Two-proportion z-test via chi-square contingency table.

    :param n1: Successes in group 1.
    :param total1: Total in group 1.
    :param n2: Successes in group 2.
    :param total2: Total in group 2.
    :return: p-value; 1.0 if either total is 0, or if both groups have identical
        all-zero or all-one proportions (which would produce a zero column sum).
    """
    if total1 == 0 or total2 == 0:
        return 1.0
    if n1 + n2 == 0 or (total1 - n1) + (total2 - n2) == 0:
        return 1.0
    table = [
        [n1, total1 - n1],
        [n2, total2 - n2],
    ]
    _, p_value, _, _ = scipy_stats.chi2_contingency(table, correction=False)
    return float(p_value)
