import math
from collections.abc import Sequence

import numpy as np
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


def cluster_bootstrap_ci(
    strata: Sequence[Sequence[tuple[int, int]]],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
    chunk_size: int = 1000,
) -> tuple[float, float]:
    """
    Percentile stratified cluster-bootstrap confidence interval for a pooled
    proportion.

    Within each stratum (e.g. a subtype run with a fixed number of
    conversations), resamples whole clusters (e.g. conversations) with
    replacement, keeping each stratum's cluster count fixed as in the
    design. Each resample's statistic is sum(successes) / sum(totals) over
    all strata, so within-cluster correlation between observations is
    preserved.

    :param strata: One list per stratum of (n_successes, n_total) pairs,
        one pair per cluster.
    :param confidence: Confidence level (default 0.95).
    :param n_resamples: Number of bootstrap resamples.
    :param seed: Seed for the random generator, for reproducible intervals.
    :param chunk_size: Resamples drawn per batch, bounding memory use.
    :return: (lower, upper) bounds as proportions; (0.0, 0.0) if there are
        no clusters or no observations.
    """
    stratum_counts = [
        np.array([(s, t) for s, t in stratum if t > 0], dtype=float).reshape(-1, 2)
        for stratum in strata
    ]
    stratum_counts = [counts for counts in stratum_counts if len(counts) > 0]
    if not stratum_counts:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    rates = np.empty(n_resamples)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        successes = np.zeros(stop - start)
        totals = np.zeros(stop - start)
        for counts in stratum_counts:
            n_clusters = len(counts)
            idx = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
            successes += counts[idx, 0].sum(axis=1)
            totals += counts[idx, 1].sum(axis=1)
        rates[start:stop] = successes / totals
    alpha = 1 - confidence
    lower, upper = np.quantile(rates, [alpha / 2, 1 - alpha / 2])
    return (float(lower), float(upper))


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
