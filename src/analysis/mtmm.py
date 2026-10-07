import json
import os
from collections.abc import Callable
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
from ordinalcorr import tetrachoric

from . import claim_grouping
from .plots import format_scenario_name, save_mtmm_matrix_grid, save_mtmm_summary_bar

SUBTYPES = ["Falsehood", "Omission", "Equivocation", "Paltering"]
CONV_THRESHOLD = 0.3
ZERO_CELL_CORRECTION = 0.5


def _fmt_pct(v: float) -> str:
    return f"{v:.0%}" if not np.isnan(v) else "n/a"


def _load_flagging_results(path: str) -> dict:
    """
    Load a flagging_results.json file from disk.

    :param path: Absolute or relative path to flagging_results.json.
    :return: Parsed JSON dict.
    """
    if not os.path.isfile(path):
        raise ValueError(f"flagging_results.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def _find_flagging_files(results_dir: str) -> list[str]:
    """
    Recursively find all flagging_results.json files under a directory tree.

    :param results_dir: Root directory to search.
    :return: Sorted list of absolute or relative file paths.
    :raises ValueError: If results_dir does not exist.
    """
    if not os.path.isdir(results_dir):
        raise ValueError(f"Directory not found: {results_dir}")
    found = []
    for root, _dirs, files in os.walk(results_dir):
        for fname in files:
            if fname == "flagging_results.json":
                found.append(os.path.join(root, fname))
    return sorted(found)


def _merge_flagging_results(paths: list[str]) -> dict:
    """
    Load and merge multiple flagging_results.json files into a single dict.

    Claim IDs are prefixed with the file index to avoid collisions across
    different agent models or scenario subtypes.

    :param paths: Ordered list of flagging_results.json file paths.
    :return: Merged dict with combined ``claim_evaluations`` and union of
             ``flagging_models``.
    :raises ValueError: If paths is empty.
    """
    if not paths:
        raise ValueError("No flagging_results.json paths provided to merge")
    merged_evals = []
    all_judges: set[str] = set()
    for idx, path in enumerate(paths):
        data = _load_flagging_results(path)
        all_judges.update(data.get("flagging_models", []))
        for entry in data.get("claim_evaluations", []):
            new_entry = dict(entry)
            new_entry["claim_id"] = f"{idx}_{entry['claim_id']}"
            merged_evals.append(new_entry)
    return {
        "flagging_models": sorted(all_judges),
        "claim_evaluations": merged_evals,
    }


def _load_or_merge_flagging_results(flagging_results_path: str, verbose: bool = False) -> dict:
    """
    Load a flagging_results.json file, or merge all such files under a directory tree.

    When ``flagging_results_path`` is a directory, it is searched recursively for
    ``flagging_results.json`` files, which are merged into a single dict. Otherwise
    it is treated as a path to a single such file.

    :param flagging_results_path: Path to a flagging_results.json file, or to a
        directory that will be searched recursively for such files.
    :param verbose: Print progress messages when True.
    :return: Parsed or merged flagging_results dict.
    :raises ValueError: If flagging_results_path is a directory containing no
        flagging_results.json files.
    """
    if os.path.isdir(flagging_results_path):
        paths = _find_flagging_files(flagging_results_path)
        if not paths:
            raise ValueError(
                f"No flagging_results.json files found under: {flagging_results_path}"
            )
        if verbose:
            print(f"Found {len(paths)} flagging_results.json files under {flagging_results_path}")
        flagging_results = _merge_flagging_results(paths)
    else:
        flagging_results = _load_flagging_results(flagging_results_path)

    if verbose:
        print(f"Loaded {len(flagging_results.get('claim_evaluations', []))} claim evaluations")

    return flagging_results


def build_claim_matrix(flagging_results: dict, mode: str = "all") -> pd.DataFrame:
    """
    Build a wide binary DataFrame from a flagging_results dict.

    Rows are unique claim_ids; columns are ``"{judge}::{subtype}"`` pairs with
    binary 0/1 integer values. Claims evaluated by only a subset of judges have
    NaN for the missing judge columns.

    :param flagging_results: Parsed flagging_results.json content.
    :param mode: Claim grouping mode from :mod:`~src.analysis.claim_grouping`.
        ``"all"`` uses the 4 raw deception types; ``"falsehood_omission"``
        restricts to Falsehood/Omission only; ``"no_paltering"`` drops Paltering and keeps the other 3 types;
        ``"active_vs_passive"`` merges Falsehood/Paltering into "Active" and maps
        Omission to "Passive", dropping Equivocation; ``"no_equivocation"`` drops
        Equivocation and keeps the other 3 types.
    :return: DataFrame indexed by claim_id with one column per (judge, subtype).
    """
    subtypes = claim_grouping.mode_subtypes(mode)
    records: dict[str, dict[str, int]] = {}
    for entry in flagging_results.get("claim_evaluations", []):
        claim_id = entry["claim_id"]
        judge = entry["flagging_model"]
        indicators = claim_grouping.indicators_for_mode(entry.get("deception_indicators", {}), mode)
        if claim_id not in records:
            records[claim_id] = {}
        for subtype in subtypes:
            col = f"{judge}::{subtype}"
            records[claim_id][col] = int(bool(indicators.get(subtype, False)))
    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "claim_id"
    return df.sort_index()


def _tetrachoric_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate the tetrachoric correlation between two binary variables.

    Assumes each binary indicator is a thresholded latent bivariate-normal
    variable and delegates to :func:`ordinalcorr.tetrachoric`, which fixes the
    thresholds at the observed marginals and maximises the 2x2 likelihood over
    ``rho`` (two-step estimator). Pearson/phi correlation attenuates
    badly on sparse binary indicators because it also depends on the
    marginal split; tetrachoric correlation removes that dependence.

    If any cell of the 2x2 table is empty, ``ZERO_CELL_CORRECTION`` is added
    to each empty cell (Savalei, 2011). ordinalcorr accepts only raw binary
    columns, so the corrected table is doubled to whole-number counts and
    rebuilt as columns. Scaling every cell by the same factor leaves the cell
    proportions, and therefore the thresholds and the maximum-likelihood
    ``rho``, unchanged.

    :param x: First binary column (may contain NaN for missing observations).
    :param y: Second binary column (may contain NaN for missing observations).
    :return: Estimated tetrachoric correlation in [-1, 1], or NaN if either
        variable is constant or there are fewer than 2 paired observations.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = x.shape[0]
    if n < 2:
        return float("nan")

    n11 = float(np.sum((x == 1) & (y == 1)))
    n10 = float(np.sum((x == 1) & (y == 0)))
    n01 = float(np.sum((x == 0) & (y == 1)))
    n00 = float(np.sum((x == 0) & (y == 0)))

    p_x = (n11 + n10) / n
    p_y = (n11 + n01) / n
    if p_x <= 0.0 or p_x >= 1.0 or p_y <= 0.0 or p_y >= 1.0:
        return float("nan")

    cells = [n00, n01, n10, n11]
    if 0.0 in cells:
        doubled = [int(2 * (c + ZERO_CELL_CORRECTION if c == 0.0 else c)) for c in cells]
        x = np.repeat([0, 0, 1, 1], doubled)
        y = np.repeat([0, 1, 0, 1], doubled)

    return float(tetrachoric(x, y))


def compute_mtmm_correlation_matrix(claim_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the tetrachoric correlation matrix over all (judge, subtype) column pairs.

    Binary MTMM indicators are sparse (most claims are not flagged for most
    subtypes), and Pearson/phi correlation attenuates toward zero as the
    marginal split departs from 50/50, which biases the convergent- and
    discriminant-validity comparisons at the heart of MTMM. Tetrachoric
    correlation instead estimates the correlation of the latent continuous
    traits assumed to underlie the binary flags, so it is not attenuated by
    unequal marginals. Pairwise complete observations are used so NaN-filled
    cells do not drop entire columns.

    :param claim_matrix: Output of :func:`build_claim_matrix`.
    :return: Symmetric DataFrame of shape (n_cols, n_cols) with diagonal 1.0.
    :raises ValueError: If claim_matrix is empty.
    """
    if claim_matrix.empty:
        raise ValueError("claim_matrix is empty; cannot compute correlations")

    cols = list(claim_matrix.columns)
    n_cols = len(cols)
    values = {col: claim_matrix[col].to_numpy(dtype=float) for col in cols}
    corr = np.full((n_cols, n_cols), np.nan)
    np.fill_diagonal(corr, 1.0)
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            r = _tetrachoric_correlation(values[cols[i]], values[cols[j]])
            corr[i, j] = r
            corr[j, i] = r
    return pd.DataFrame(corr, index=cols, columns=cols)


def _parse_column(col: str) -> tuple[str, str]:
    """
    Split a ``"{judge}::{subtype}"`` column name into its components.

    :param col: Column name with ``::`` separator.
    :return: Tuple of (judge, subtype).
    """
    judge, subtype = col.split("::", 1)
    return judge, subtype


def classify_mtmm_correlations(
    corr_matrix: pd.DataFrame,
) -> dict[str, list[float]]:
    """
    Categorise each off-diagonal upper-triangle correlation into one of the
    three canonical MTMM block types.

    - monotrait_heteromethod: same subtype, different judge
    - heterotrait_monomethod: different subtype, same judge
    - heterotrait_heteromethod: different subtype, different judge

    NaN correlations (from constant columns) are silently skipped.

    :param corr_matrix: Output of :func:`compute_mtmm_correlation_matrix`.
    :return: Dict mapping block-type name to list of correlation values.
    """
    result: dict[str, list[float]] = {
        "monotrait_heteromethod": [],
        "heterotrait_monomethod": [],
        "heterotrait_heteromethod": [],
    }
    cols = list(corr_matrix.columns)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix.iloc[i, j]
            if pd.isna(r):
                continue
            judge_i, subtype_i = _parse_column(cols[i])
            judge_j, subtype_j = _parse_column(cols[j])
            same_judge = judge_i == judge_j
            same_subtype = subtype_i == subtype_j
            if same_judge:
                result["heterotrait_monomethod"].append(float(r))
            elif same_subtype:
                result["monotrait_heteromethod"].append(float(r))
            else:
                result["heterotrait_heteromethod"].append(float(r))
    return result


def _is_hthm_neighbour(judge_k: str, subtype_k: str, judge_anchor: str, subtype_anchor: str) -> bool:
    """
    Test whether column ``k`` is a heterotrait-heteromethod neighbour of an anchor.

    :param judge_k: Judge of the candidate neighbour column.
    :param subtype_k: Subtype of the candidate neighbour column.
    :param judge_anchor: Judge of the anchor column.
    :param subtype_anchor: Subtype of the anchor column.
    :return: True if the judge and subtype both differ from the anchor's.
    """
    return judge_k != judge_anchor and subtype_k != subtype_anchor


def _is_htmonm_neighbour(judge_k: str, subtype_k: str, judge_anchor: str, subtype_anchor: str) -> bool:
    """
    Test whether column ``k`` is a heterotrait-monomethod neighbour of an anchor.

    :param judge_k: Judge of the candidate neighbour column.
    :param subtype_k: Subtype of the candidate neighbour column.
    :param judge_anchor: Judge of the anchor column.
    :param subtype_anchor: Subtype of the anchor column.
    :return: True if the judge matches the anchor's and the subtype differs.
    """
    return judge_k == judge_anchor and subtype_k != subtype_anchor


def _neighbour_correlations(
    corr_matrix: pd.DataFrame,
    col_i: str,
    col_j: str,
    is_neighbour: Callable[[str, str, str, str], bool],
    dedupe: bool,
) -> list[float]:
    """
    Collect correlations from both endpoints of an MTHM pair to matching neighbour columns.

    For each endpoint (``col_i`` and ``col_j``), collect r(endpoint, col_k) for every
    other column ``col_k`` where ``is_neighbour(judge_k, subtype_k, judge_endpoint,
    subtype_endpoint)`` is True.

    :param corr_matrix: Full MTMM correlation matrix.
    :param col_i: First column of the MTHM pair (format ``"{judge}::{subtype}"``).
    :param col_j: Second column of the MTHM pair.
    :param is_neighbour: Predicate deciding whether ``col_k`` neighbours an endpoint.
    :param dedupe: If True, skip (endpoint, col_k) pairs already counted from the
        other endpoint.
    :return: List of correlation values matching the predicate across both endpoints.
    """
    judge_i, subtype_i = _parse_column(col_i)
    judge_j, subtype_j = _parse_column(col_j)
    cols = list(corr_matrix.columns)
    pairs_seen: set[tuple[str, str]] = set()
    result = []
    for anchor, judge_a, subtype_a in [
        (col_i, judge_i, subtype_i),
        (col_j, judge_j, subtype_j),
    ]:
        for col_k in cols:
            if col_k == anchor:
                continue
            judge_k, subtype_k = _parse_column(col_k)
            if not is_neighbour(judge_k, subtype_k, judge_a, subtype_a):
                continue
            if dedupe:
                pair = (min(anchor, col_k), max(anchor, col_k))
                if pair in pairs_seen:
                    continue
                pairs_seen.add(pair)
            r = corr_matrix.loc[anchor, col_k]
            if not pd.isna(r):
                result.append(float(r))
    return result


def _hthm_row_col_neighbours(corr_matrix: pd.DataFrame, col_i: str, col_j: str) -> list[float]:
    """
    Return HTHM values sharing a row or column with the MTHM coefficient at (col_i, col_j).

    For each endpoint, collect r(endpoint, col_k) where judge_k differs from the
    endpoint's judge and subtype_k differs from the endpoint's subtype. Deduplicates
    pairs across both endpoints.

    :param corr_matrix: Full MTMM correlation matrix.
    :param col_i: First column of the MTHM pair (format ``"{judge}::{subtype}"``).
    :param col_j: Second column of the MTHM pair.
    :return: List of HTHM correlation values neighbouring this MTHM entry.
    """
    return _neighbour_correlations(corr_matrix, col_i, col_j, _is_hthm_neighbour, dedupe=True)


def _htmonm_local_values(corr_matrix: pd.DataFrame, col_i: str, col_j: str) -> list[float]:
    """
    Return HTMonoM values from the method blocks of both endpoints of an MTHM coefficient.

    For each endpoint, collect r(endpoint, col_k) where judge_k equals the endpoint's
    judge and subtype_k differs (same-judge, different-subtype pairs).

    :param corr_matrix: Full MTMM correlation matrix.
    :param col_i: First column of the MTHM pair.
    :param col_j: Second column of the MTHM pair.
    :return: List of HTMonoM correlation values from both method blocks.
    """
    return _neighbour_correlations(corr_matrix, col_i, col_j, _is_htmonm_neighbour, dedupe=False)


def _safe_mean(values: list[float]) -> float:
    """
    Compute the mean of a list of values, or NaN if the list is empty.

    :param values: List of numeric values.
    :return: Mean of values, or NaN if values is empty.
    """
    return float(np.mean(values)) if values else float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    """
    Compute a ratio, or NaN if the denominator is zero.

    :param numerator: Ratio numerator.
    :param denominator: Ratio denominator.
    :return: numerator / denominator, or NaN if denominator is 0.
    """
    return float(numerator / denominator) if denominator else float("nan")


def assess_construct_validity(classified: dict[str, list[float]], corr_matrix: pd.DataFrame) -> dict:
    """
    Assess construct validity following Campbell and Fiske (1959).

    :param classified: Output of :func:`classify_mtmm_correlations`.
    :param corr_matrix: Full MTMM correlation matrix from
        :func:`compute_mtmm_correlation_matrix`, used for per-coefficient
        structural comparisons.
    :return: Dict with mean values per block, boolean validity flags, the
             fraction of individual MTHM correlations above the HTHM mean,
             a threshold gate (>= CONV_THRESHOLD), a one-sample t-test
             significance flag, and exact per-coefficient structural checks.
    """
    mthm = classified.get("monotrait_heteromethod", [])
    htmonm = classified.get("heterotrait_monomethod", [])
    hthm = classified.get("heterotrait_heteromethod", [])

    mean_mthm = _safe_mean(mthm)
    mean_htmonm = _safe_mean(htmonm)
    mean_hthm = _safe_mean(hthm)

    mthm_is_positive = bool(mean_mthm > 0) if mthm else False
    mthm_all_positive = bool(all(r > 0 for r in mthm)) if mthm else False
    mthm_pct_positive = _safe_ratio(sum(1 for r in mthm if r > 0), len(mthm))
    mean_mthm_exceeds_mean_hthm = bool(mean_mthm > mean_hthm) if (mthm and hthm) else False
    mean_mthm_exceeds_mean_htmonm = bool(mean_mthm > mean_htmonm) if (mthm and htmonm) else False

    pct = _safe_ratio(sum(1 for r in mthm if r > mean_hthm), len(mthm)) if hthm else float("nan")

    mthm_meets_threshold = bool(mean_mthm >= CONV_THRESHOLD) if mthm else False

    if len(mthm) >= 2:
        t_result = scipy.stats.ttest_1samp(mthm, 0, alternative="greater")
        mthm_significant_gt_zero = bool(t_result.pvalue < 0.05)
    else:
        mthm_significant_gt_zero = False

    block_ordering_holds = bool(1.0 > mean_mthm > mean_htmonm > mean_hthm) if (mthm and htmonm and hthm) else False

    mthm_exceed_row_col_hthm_flags = []
    mthm_exceed_local_htmonm_flags = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            judge_ci, subtype_ci = _parse_column(cols[i])
            judge_cj, subtype_cj = _parse_column(cols[j])
            if judge_ci == judge_cj or subtype_ci != subtype_cj:
                continue
            r_val = corr_matrix.iloc[i, j]
            if pd.isna(r_val):
                continue
            r_val = float(r_val)
            hthm_nb = _hthm_row_col_neighbours(corr_matrix, cols[i], cols[j])
            htmonm_nb = _htmonm_local_values(corr_matrix, cols[i], cols[j])
            mthm_exceed_row_col_hthm_flags.append(bool(hthm_nb and all(r_val > v for v in hthm_nb)))
            mthm_exceed_local_htmonm_flags.append(bool(htmonm_nb and all(r_val > v for v in htmonm_nb)))

    mthm_all_exceed_row_col_hthm = bool(mthm_exceed_row_col_hthm_flags and all(mthm_exceed_row_col_hthm_flags))
    mthm_all_exceed_local_htmonm = bool(mthm_exceed_local_htmonm_flags and all(mthm_exceed_local_htmonm_flags))
    mthm_pct_exceed_row_col_hthm = _safe_ratio(
        sum(mthm_exceed_row_col_hthm_flags), len(mthm_exceed_row_col_hthm_flags)
    )
    mthm_pct_exceed_local_htmonm = _safe_ratio(
        sum(mthm_exceed_local_htmonm_flags), len(mthm_exceed_local_htmonm_flags)
    )

    return {
        "mean_mthm": mean_mthm,
        "mean_htmonm": mean_htmonm,
        "mean_hthm": mean_hthm,
        "mthm_is_positive": mthm_is_positive,
        "mthm_all_positive": mthm_all_positive,
        "mthm_pct_positive": mthm_pct_positive,
        "mthm_exceeds_hthm": mean_mthm_exceeds_mean_hthm,
        "mthm_exceeds_htmonm": mean_mthm_exceeds_mean_htmonm,
        "mthm_pct_above_hthm_mean": pct,
        "mthm_meets_threshold": mthm_meets_threshold,
        "mthm_significant_gt_zero": mthm_significant_gt_zero,
        "mthm_all_exceed_row_col_hthm": mthm_all_exceed_row_col_hthm,
        "mthm_pct_exceed_row_col_hthm": mthm_pct_exceed_row_col_hthm,
        "mthm_all_exceed_local_htmonm": mthm_all_exceed_local_htmonm,
        "mthm_pct_exceed_local_htmonm": mthm_pct_exceed_local_htmonm,
        "block_ordering_holds": block_ordering_holds,
    }


def _get_heterotrait_monomethod_block(
    corr_matrix: pd.DataFrame, judge: str, subtypes: list[str]
) -> np.ndarray:
    """
    Extract the square subtype-intercorrelation block for a single judge.

    :param corr_matrix: Full MTMM correlation matrix.
    :param judge: Judge identifier.
    :param subtypes: Ordered list of subtype names.
    :return: NumPy array of shape (n_subtypes, n_subtypes).
    """
    cols = [f"{judge}::{s}" for s in subtypes]
    return corr_matrix.loc[cols, cols].values


def _assess_two_trait_consistency(corr_matrix: pd.DataFrame, judges: list[str], subtypes: list[str]) -> dict:
    """
    Assess judge-consistency of a 2-trait relationship.

    With exactly two traits, each judge's heterotrait-monomethod block
    collapses to a single scalar correlation between the two traits, so the
    cross-judge Spearman comparison used for 3+ traits (which needs at least
    two paired dimensions per judge) is not defined. Instead this reports the
    per-judge scalar directly and summarises its spread across judges.

    :param corr_matrix: Full MTMM correlation matrix.
    :param judges: List of judge identifiers present in corr_matrix.
    :param subtypes: Ordered list of exactly 2 subtype/trait names.
    :return: Dict with ``per_judge_correlation``, dispersion statistics, an
             ``all_same_sign`` agreement flag, and ``mean_consistency`` set to
             the mean per-judge correlation (for interface parity with the
             3+ trait case).
    """
    col_a = f"::{subtypes[0]}"
    col_b = f"::{subtypes[1]}"
    per_judge: dict[str, float] = {}
    for judge in judges:
        a, b = f"{judge}{col_a}", f"{judge}{col_b}"
        if a not in corr_matrix.columns or b not in corr_matrix.columns:
            continue
        r_val = corr_matrix.loc[a, b]
        per_judge[judge] = float(r_val) if not pd.isna(r_val) else float("nan")

    values = [v for v in per_judge.values() if not np.isnan(v)]
    mean_consistency = float(np.mean(values)) if values else float("nan")
    sd_correlation = float(np.std(values)) if values else float("nan")
    min_correlation = float(np.min(values)) if values else float("nan")
    max_correlation = float(np.max(values)) if values else float("nan")
    all_same_sign = bool(all(v > 0 for v in values) or all(v < 0 for v in values)) if values else False
    pattern_meets_threshold = bool(mean_consistency >= CONV_THRESHOLD) if values else False

    return {
        "pairwise_spearman": {},
        "per_judge_correlation": per_judge,
        "mean_consistency": mean_consistency,
        "sd_correlation": sd_correlation,
        "min_correlation": min_correlation,
        "max_correlation": max_correlation,
        "all_same_sign": all_same_sign,
        "pattern_meets_threshold": pattern_meets_threshold,
    }


def check_trait_pattern_consistency(
    corr_matrix: pd.DataFrame, judges: list[str], subtypes: list[str]
) -> dict:
    """
    Assess whether trait intercorrelation patterns are consistent across judges.

    For each judge, the upper-triangle of the heterotrait-monomethod block is
    extracted as a vector of C(n_subtypes, 2) values. Pairwise Spearman
    correlations between these vectors measure method-block consistency. With
    exactly 2 subtypes that vector has length 1, so Spearman is undefined;
    :func:`_assess_two_trait_consistency` is used instead in that case.

    :param corr_matrix: Full MTMM correlation matrix.
    :param judges: List of judge identifiers present in corr_matrix.
    :param subtypes: Ordered list of subtype names.
    :return: Dict with ``pairwise_spearman`` (judge-pair keys) and
             ``mean_consistency`` (mean Spearman r across all pairs).
    """
    if len(subtypes) == 2:
        return _assess_two_trait_consistency(corr_matrix, judges, subtypes)

    triu_idx = np.triu_indices(len(subtypes), k=1)
    judge_vectors: dict[str, np.ndarray] = {}
    for judge in judges:
        cols = [f"{judge}::{s}" for s in subtypes]
        if not all(c in corr_matrix.columns for c in cols):
            continue
        block = _get_heterotrait_monomethod_block(corr_matrix, judge, subtypes)
        judge_vectors[judge] = block[triu_idx]

    pairwise_spearman: dict[str, float] = {}
    for j1, j2 in combinations(list(judge_vectors.keys()), 2):
        v1 = judge_vectors[j1]
        v2 = judge_vectors[j2]
        mask = ~(np.isnan(v1) | np.isnan(v2))
        if mask.sum() < 2:
            r = float("nan")
        else:
            try:
                result = scipy.stats.spearmanr(v1[mask], v2[mask])
                r = float(result.statistic)
            except Exception:
                r = float("nan")
        pairwise_spearman[f"{j1} vs {j2}"] = r

    values = [v for v in pairwise_spearman.values() if not np.isnan(v)]
    mean_consistency = float(np.mean(values)) if values else float("nan")

    pattern_meets_threshold = bool(mean_consistency >= CONV_THRESHOLD) if values else False

    return {
        "pairwise_spearman": pairwise_spearman,
        "mean_consistency": mean_consistency,
        "pattern_meets_threshold": pattern_meets_threshold,
    }


def save_mtmm_results(results: dict, output_dir: str) -> None:
    """
    Write MTMM analysis outputs to disk.

    Produces ``mtmm_results.json`` with validity and consistency statistics
    and ``mtmm_corr_matrix.csv`` with the full correlation matrix.

    :param results: Output dict from :func:`run_mtmm`.
    :param output_dir: Directory to write outputs into.
    """
    os.makedirs(output_dir, exist_ok=True)

    json_data = {
        "judges": results["judges"],
        "subtypes": results["subtypes"],
        "validity": results["validity"],
        "consistency": {k: v for k, v in results["consistency"].items()},
        "classified_means": {
            k: float(np.mean(v)) if v else None
            for k, v in results["classified"].items()
        },
        "classified_sds": {
            k: float(np.std(v)) if v else None
            for k, v in results["classified"].items()
        },
    }

    json_path = os.path.join(output_dir, "mtmm_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    csv_path = os.path.join(output_dir, "mtmm_corr_matrix.csv")
    results["corr_matrix"].to_csv(csv_path)


def run_mtmm(
    flagging_results_path: str,
    output_dir: str,
    mode: str = "all",
    scenario: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run the full MTMM analysis pipeline from a flagging_results.json file or
    a directory tree containing multiple such files.

    When a directory is supplied, all ``flagging_results.json`` files found
    recursively are merged into a single claim matrix, enabling analysis across
    multiple agent models and scenario subtypes simultaneously.

    Steps: load/merge → build claim matrix → compute correlation matrix →
    classify correlations → assess convergent validity →
    check trait pattern consistency → save results and plots.

    :param flagging_results_path: Path to a flagging_results.json file, or to
        a directory that will be searched recursively for such files.
    :param output_dir: Directory to write JSON, CSV, and plot outputs.
    :param mode: Claim grouping mode from :mod:`~src.analysis.claim_grouping`
        (``"all"``, ``"falsehood_omission"``, ``"no_paltering"``,
        ``"active_vs_passive"``, or ``"no_equivocation"``).
    :param scenario: Scenario type identifier (e.g. "product_promotion");
        prefixed to plot titles when provided.
    :param verbose: Print progress messages when True.
    :return: Dict with keys: claim_matrix, corr_matrix, classified, validity,
             consistency, judges, subtypes.
    """
    flagging_results = _load_or_merge_flagging_results(flagging_results_path, verbose)

    claim_matrix = build_claim_matrix(flagging_results, mode=mode)

    if claim_matrix.empty:
        raise ValueError("No claim evaluations found in flagging_results.json")

    if verbose:
        print(f"Claim matrix: {claim_matrix.shape[0]} claims × {claim_matrix.shape[1]} columns")

    corr_matrix = compute_mtmm_correlation_matrix(claim_matrix)

    if verbose:
        print(f"Correlation matrix: {corr_matrix.shape[0]} × {corr_matrix.shape[1]}")

    cols = list(claim_matrix.columns)
    judges = sorted(set(_parse_column(c)[0] for c in cols))
    mode_subtypes = claim_grouping.mode_subtypes(mode)
    subtypes_present = [s for s in mode_subtypes if any(_parse_column(c)[1] == s for c in cols)]

    if len(judges) < 2:
        raise ValueError(
            f"MTMM requires at least 2 judges; found {len(judges)}: {judges}"
        )

    classified = classify_mtmm_correlations(corr_matrix)
    validity = assess_construct_validity(classified, corr_matrix)
    consistency = check_trait_pattern_consistency(corr_matrix, judges, subtypes_present)

    if verbose:
        print(f"Judges ({len(judges)}): {judges}")
        print(f"Subtypes: {subtypes_present}")
        print(
            f"Mean MTHM={validity['mean_mthm']:.3f}  "
            f"Mean HTMonoM={validity['mean_htmonm']:.3f}  "
            f"Mean HTHM={validity['mean_hthm']:.3f}"
        )
        threshold_label = f"Mean MTHM meets threshold (>={CONV_THRESHOLD}):"
        print(f"{threshold_label:<43}{validity['mthm_meets_threshold']}")
        print(f"{'Mean MTHM significantly > 0:':<43}{validity['mthm_significant_gt_zero']}")
        print(f"{'All MTHM coefficients > 0:':<43}{validity['mthm_all_positive']}  ({_fmt_pct(validity['mthm_pct_positive'])})")
        print(f"{'All MTHM > row/col HTHM neighbours:':<43}{validity['mthm_all_exceed_row_col_hthm']}  ({_fmt_pct(validity['mthm_pct_exceed_row_col_hthm'])})")
        print(f"{'All MTHM > local HTMonoM:':<43}{validity['mthm_all_exceed_local_htmonm']}  ({_fmt_pct(validity['mthm_pct_exceed_local_htmonm'])})")
        print(f"{'Block ordering (diag>MTHM>HTMonoM>HTHM):':<43}{validity['block_ordering_holds']}")
        print(f"Mean trait-pattern consistency: {consistency['mean_consistency']:.3f}  "
              f"meets threshold: {consistency['pattern_meets_threshold']}")

    results = {
        "claim_matrix": claim_matrix,
        "corr_matrix": corr_matrix,
        "classified": classified,
        "validity": validity,
        "consistency": consistency,
        "judges": judges,
        "subtypes": subtypes_present,
    }

    os.makedirs(output_dir, exist_ok=True)
    save_mtmm_results(results, output_dir)
    save_mtmm_matrix_grid(
        corr_matrices=[corr_matrix],
        col_labels=[f"{format_scenario_name(scenario)} {claim_grouping.taxonomy_label(mode)}".strip()],
        output_path=os.path.join(output_dir, "mtmm_heatmap.png"),
    )
    save_mtmm_summary_bar(
        classified,
        os.path.join(output_dir, "mtmm_summary.png"),
        scenario=scenario,
    )

    if verbose:
        print(f"Results written to {output_dir}")

    return results


def mtmm_table_row(trial_dir: str, mode: str) -> dict:
    """
    Collect the Table 4 MTMM statistics of one taxonomy from a finished trial.

    C4 is left empty for two-trait taxonomies, where each judge's HTMonoM block
    holds a single correlation and the pairwise Spearman correlation is undefined.

    :param trial_dir: Trial results directory.
    :param mode: Claim grouping mode.
    :return: Dict with mean_mthm, mean_htmonm, mean_hthm, c1_met, c2_pct, c3_pct, c4_rho, and order.
    """
    path = os.path.join(trial_dir, f"mtmm{claim_grouping.output_suffix(mode)}", "mtmm_results.json")
    with open(path) as f:
        results = json.load(f)
    validity = results["validity"]
    two_trait = len(results["subtypes"]) < 3
    return {
        "mean_mthm": validity["mean_mthm"],
        "mean_htmonm": validity["mean_htmonm"],
        "mean_hthm": validity["mean_hthm"],
        "c1_met": validity["mthm_meets_threshold"] and validity["mthm_significant_gt_zero"],
        "c2_pct": 100 * validity["mthm_pct_exceed_row_col_hthm"],
        "c3_pct": 100 * validity["mthm_pct_exceed_local_htmonm"],
        "c4_rho": None if two_trait else results["consistency"]["mean_consistency"],
        "order": validity["block_ordering_holds"],
    }


def save_mtmm_matrix_figures(trials: list[tuple[str, str]], output_dir: str) -> list[str]:
    """
    Save paper-style MTMM correlation matrices for the 4-type and Binary taxonomies,
    one panel per trial side by side.

    :param trials: (trial results directory, scenario) pairs, in display order.
    :param output_dir: Directory to write the PNGs to.
    :return: Paths of the saved figures.
    """
    paths = []
    for name, mode in [("4type", "all"), ("binary", "falsehood_omission")]:
        output_path = os.path.join(output_dir, f"mtmm_matrices_{name}.png")
        save_mtmm_matrix_grid(
            corr_matrices=[
                pd.read_csv(
                    os.path.join(trial_dir, f"mtmm{claim_grouping.output_suffix(mode)}", "mtmm_corr_matrix.csv"),
                    index_col=0,
                )
                for trial_dir, _ in trials
            ],
            col_labels=[format_scenario_name(scenario) for _, scenario in trials],
            output_path=output_path,
        )
        paths.append(output_path)
    return paths
