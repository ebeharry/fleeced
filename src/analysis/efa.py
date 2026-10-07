import collections
import inspect
import json
import os
from itertools import permutations

import numpy as np
import pandas as pd
import factor_analyzer.factor_analyzer as _factor_analyzer_module
from factor_analyzer import FactorAnalyzer
from sklearn.utils import check_array

from . import claim_grouping
from .mtmm import (
    _load_or_merge_flagging_results,
    build_claim_matrix,
    compute_mtmm_correlation_matrix,
)
from .plots import (
    format_scenario_name,
    save_efa_loading_bar_chart,
    save_efa_loading_grid,
    save_efa_membership_grid,
    save_scree_grid,
    save_type_purity_dot_plot,
)

LOADING_GRID_PANEL_WIDTH = 5.8
LOADING_GRID_DPI = 600


def _check_array_compat(X: np.ndarray, *args, force_all_finite: bool | str | None = None, **kwargs) -> np.ndarray:
    """
    Call sklearn's check_array with the ``force_all_finite`` keyword renamed.

    :param X: Array to validate.
    :param force_all_finite: Legacy keyword forwarded as ``ensure_all_finite``.
    :return: Validated array.
    """
    if force_all_finite is not None:
        kwargs["ensure_all_finite"] = force_all_finite
    return check_array(X, *args, **kwargs)


if "force_all_finite" not in inspect.signature(check_array).parameters:
    # factor_analyzer 0.5.1 still passes the keyword scikit-learn 1.8 removed.
    _factor_analyzer_module.check_array = _check_array_compat


def _drop_constant_cols(claim_matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop columns whose correlation with every other column is NaN (constant columns).

    Constant columns contribute no variance and cannot be factored.

    :param claim_matrix: Binary claim matrix from :func:`build_claim_matrix`.
    :return: Tuple of (filtered claim_matrix, retained column names).
    """
    corr = claim_matrix.corr(method="pearson")
    valid_cols = [c for c in corr.columns if not corr[c].isna().all()]
    return claim_matrix[valid_cols], valid_cols


def _ensure_positive_definite(corr: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """
    Repair a non-Gramian correlation matrix with Knol-Berger eigenvalue smoothing.

    Pairwise tetrachoric estimates are not guaranteed to form a positive
    semidefinite matrix. If any eigenvalue is negative, the matrix is
    decomposed as ``R = K D K^T``, eigenvalues below ``epsilon`` are replaced
    with ``epsilon``, and the reconstructed matrix is rescaled to a unit
    diagonal: ``Diag(K D+ K^T)^(-1/2) [K D+ K^T] Diag(K D+ K^T)^(-1/2)``
    (Knol & Berger, 1991). Positive semidefinite matrices are returned unchanged.

    :param corr: Symmetric correlation matrix of shape (n_cols, n_cols).
    :param epsilon: Constant that replaces eigenvalues below it.
    :return: Gramian correlation matrix with unit diagonal.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    if eigenvalues.min() >= 0.0:
        return corr
    smoothed_eigenvalues = np.maximum(eigenvalues, epsilon)
    covariance = eigenvectors @ np.diag(smoothed_eigenvalues) @ eigenvectors.T
    inv_sqrt_diag = np.diag(np.diag(covariance) ** -0.5)
    return inv_sqrt_diag @ covariance @ inv_sqrt_diag


def _build_efa_correlation_matrix(claim_matrix: pd.DataFrame) -> np.ndarray:
    """
    Compute the tetrachoric correlation matrix used for factor extraction.

    Reuses the pairwise-complete tetrachoric estimator from MTMM so both analyses
    operate on the same correlations. Pairs whose correlation is undefined (a
    variable is constant over the paired observations) are set to 0, and the
    result is repaired with Knol-Berger smoothing if it is not positive
    semidefinite.

    :param claim_matrix: Binary claim matrix with no constant columns.
    :return: Gramian correlation matrix of shape (n_cols, n_cols).
    """
    corr = compute_mtmm_correlation_matrix(claim_matrix).to_numpy(dtype=float)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    return _ensure_positive_definite(corr)


def _sorted_eigenvalues(corr: np.ndarray) -> np.ndarray:
    """
    Return the eigenvalues of a symmetric matrix in descending order.

    :param corr: Symmetric correlation matrix.
    :return: Eigenvalues sorted from largest to smallest.
    """
    return np.sort(np.linalg.eigvalsh(corr))[::-1]


def _permute_columns(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Independently shuffle the rows of each column, preserving each column's marginals.

    :param values: Array of shape (n_claims, n_cols).
    :param rng: Random number generator.
    :return: Column-permuted copy of ``values``.
    """
    return np.column_stack([rng.permutation(values[:, j]) for j in range(values.shape[1])])


def _run_parallel_analysis(
    claim_matrix: pd.DataFrame,
    n_random: int,
    seed: int,
) -> dict:
    """
    Compare observed eigenvalues against the mean eigenvalues from column permutations.

    Generates ``n_random`` random datasets by independently permuting each
    column of ``claim_matrix``, which preserves every variable's flag rate and
    binary coding while removing associations between variables. Each dataset's
    tetrachoric correlation matrix is built with the same estimator and
    Knol-Berger repair as the observed matrix, and its PCA eigenvalues are
    averaged across replicates. Factors are retained in order while the
    observed eigenvalue exceeds the mean random eigenvalue, stopping at the
    first factor that does not (Horn, 1965; Garrido et al., 2013).

    :param claim_matrix: Binary claim matrix with no constant columns.
    :param n_random: Number of column-permuted datasets to generate.
    :param seed: Seed for the random number generator.
    :return: Dict with observed_eigenvalues, simulated_mean_eigenvalues,
             and n_factors_retained.
    """
    n_cols = claim_matrix.shape[1]
    rng = np.random.default_rng(seed)
    values = claim_matrix.to_numpy(dtype=float)

    observed_evals = _sorted_eigenvalues(_build_efa_correlation_matrix(claim_matrix))

    sim_evals = np.zeros((n_random, n_cols))
    for i in range(n_random):
        permuted = pd.DataFrame(_permute_columns(values, rng), columns=claim_matrix.columns)
        sim_evals[i] = _sorted_eigenvalues(_build_efa_correlation_matrix(permuted))

    mean_evals = sim_evals.mean(axis=0)
    n_factors = 0
    while n_factors < n_cols and observed_evals[n_factors] > mean_evals[n_factors]:
        n_factors += 1

    return {
        "observed_eigenvalues": observed_evals.tolist(),
        "simulated_mean_eigenvalues": mean_evals.tolist(),
        "n_factors_retained": n_factors,
    }


def _align_factor_correlations(loadings: np.ndarray, phi: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """
    Reorder factor_analyzer's factor correlation matrix to match its loading columns.

    After rotation, factor_analyzer sorts the loading and structure columns by
    variance but leaves ``phi_`` in the pre-sort order. Its ``structure_`` is
    ``loadings @ phi`` computed before the sort and sorted with the loadings,
    so the correct order is the permutation of ``phi_`` that reproduces it.

    :param loadings: factor_analyzer ``loadings_`` of shape (n_vars, n_factors).
    :param phi: factor_analyzer ``phi_`` of shape (n_factors, n_factors).
    :param structure: factor_analyzer ``structure_`` of shape (n_vars, n_factors).
    :return: Factor correlation matrix in the same factor order as ``loadings``.
    :raises ValueError: If no permutation of ``phi`` reproduces ``structure``.
    """
    for order in permutations(range(phi.shape[0])):
        candidate = phi[np.ix_(order, order)]
        if np.allclose(loadings @ candidate, structure, atol=1e-8):
            return candidate
    raise ValueError("No factor order of phi_ reproduces factor_analyzer's structure_ matrix.")


def _fit_efa_loadings(corr: np.ndarray, n_factors: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a common factor model to a correlation matrix with oblique Promax rotation.

    Uses factor_analyzer's MINRES extraction, which analyses only shared variance
    and is more tolerant than maximum likelihood of correlation matrices that
    needed smoothing. Promax rotation lets the factors correlate. With a single
    factor no rotation is applied and the factor correlation matrix is 1.

    :param corr: Positive definite correlation matrix of shape (n_cols, n_cols).
    :param n_factors: Number of factors to extract.
    :return: Tuple of (pattern_loadings, factor_correlations), where
        pattern_loadings has shape (n_cols, n_factors) and factor_correlations
        has shape (n_factors, n_factors).
    """
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation="promax" if n_factors > 1 else None,
        method="minres",
        is_corr_matrix=True,
    )
    fa.fit(corr)
    loadings = np.asarray(fa.loadings_)
    if fa.phi_ is None:
        return loadings, np.eye(n_factors)
    return loadings, _align_factor_correlations(loadings, np.asarray(fa.phi_), np.asarray(fa.structure_))


def _check_simple_loading_structure(
    loadings: np.ndarray,
    col_names: list[str],
    cross_loading_threshold: float,
) -> dict:
    """
    Assess whether each variable shows a simple loading pattern.

    A variable is simple if its maximum absolute loading reaches
    ``cross_loading_threshold`` (so it is interpretable on that factor) and all
    remaining absolute loadings fall below the same threshold.

    :param loadings: Rotated loadings of shape (n_vars, n_factors).
    :param col_names: Variable names corresponding to rows of loadings.
    :param cross_loading_threshold: Loading magnitude at or above which a loading
        is treated as salient; the primary loading must reach it and every
        cross-loading must stay below it.
    :return: Dict with per-variable results and overall simple_structure_fraction.
    """
    n_vars, n_factors = loadings.shape
    per_variable = []
    n_simple = 0
    for i, name in enumerate(col_names):
        abs_row = np.abs(loadings[i])
        primary_factor = int(np.argmax(abs_row))
        primary_loading = float(abs_row[primary_factor])
        cross_loadings = [float(abs_row[j]) for j in range(n_factors) if j != primary_factor]
        is_simple = primary_loading >= cross_loading_threshold and all(
            c < cross_loading_threshold for c in cross_loadings
        )
        if is_simple:
            n_simple += 1
        per_variable.append(
            {
                "variable": name,
                "primary_factor": primary_factor + 1,
                "primary_loading": round(primary_loading, 4),
                "max_cross_loading": round(max(cross_loadings), 4) if cross_loadings else 0.0,
                "is_simple": is_simple,
            }
        )
    return {
        "per_variable": per_variable,
        "n_simple": n_simple,
        "n_vars": n_vars,
        "simple_structure_fraction": round(n_simple / n_vars, 4) if n_vars > 0 else 0.0,
    }


def _compute_factor_importance(
    loadings: np.ndarray, phi: np.ndarray, col_names: list[str]
) -> dict:
    """
    Compute per-factor importance and per-variable communalities.

    Under oblique rotation the factors overlap, so a factor's importance is the
    sum of its squared structure coefficients (structure = pattern @ phi). These
    sums are not variances explained by the rotated factors and do not add up to
    the total common variance. The total common variance explained is instead
    the sum of communalities, diag(pattern @ phi @ pattern.T). Total variance
    equals the trace of the correlation matrix, which is n_vars because the
    tetrachoric matrix has a unit diagonal.

    :param loadings: Pattern loadings of shape (n_vars, n_factors).
    :param phi: Factor correlation matrix of shape (n_factors, n_factors).
    :param col_names: Variable names corresponding to rows of loadings.
    :return: Dict with total_variance, factor_importance, factor_importance_pct,
             total_common_pct, and communalities (keyed by variable name).
    """
    total_variance = float(len(col_names))
    structure = loadings @ phi
    factor_importance = np.sum(structure**2, axis=0)
    communalities = np.sum(loadings * structure, axis=1)
    return {
        "total_variance": round(total_variance, 4),
        "factor_importance": [round(float(v), 4) for v in factor_importance],
        "factor_importance_pct": [round(float(v) / total_variance * 100, 2) for v in factor_importance],
        "total_common_pct": round(float(communalities.sum()) / total_variance * 100, 2),
        "communalities": {name: round(float(h2), 4) for name, h2 in zip(col_names, communalities)},
    }


def type_purity(loadings: pd.DataFrame, threshold: float) -> float | None:
    """
    Compute the type purity of an EFA solution.

    For each factor, the salient loadings (``|lambda| >= threshold``) are grouped by
    deception type. Purity is the size of the largest group, summed over factors,
    divided by the number of salient loadings. A cross-loaded variable counts once on
    each factor where it is salient.

    :param loadings: Pattern loadings indexed by ``"{judge}::{type}"``, one column per factor.
    :param threshold: Minimum absolute pattern coefficient for a loading to be salient.
    :return: Type purity as a percentage, or None if no loading is salient.
    """
    largest = 0
    total = 0
    for factor in loadings.columns:
        types = [var.split("::", 1)[1] for var in loadings.index[loadings[factor].abs() >= threshold]]
        if types:
            largest += collections.Counter(types).most_common(1)[0][1]
            total += len(types)
    return 100 * largest / total if total else None


def max_abs_factor_correlation(factor_correlations: pd.DataFrame) -> float | None:
    """
    Return the largest absolute off-diagonal factor correlation.

    :param factor_correlations: Square factor correlation matrix.
    :return: Largest ``|phi|`` between two distinct factors, or None for a one-factor solution.
    """
    values = factor_correlations.to_numpy(dtype=float)
    off_diagonal = np.abs(values[~np.eye(len(values), dtype=bool)])
    return float(off_diagonal.max()) if off_diagonal.size else None


def save_efa_results(results: dict, output_dir: str) -> None:
    """
    Write EFA outputs to disk.

    Produces ``efa_results.json`` with parallel analysis and loading structure
    statistics, ``efa_loadings.csv`` with the rotated (pattern) factor loadings
    matrix, ``efa_structure.csv`` with the structure coefficients, and
    ``efa_factor_correlations.csv`` with the oblique factor correlation matrix.

    :param results: Output dict from :func:`run_efa`.
    :param output_dir: Directory to write outputs into.
    """
    os.makedirs(output_dir, exist_ok=True)

    factor_correlations = results.get("factor_correlations")
    json_data = {
        "n_claims": results["n_claims"],
        "n_vars": results["n_vars"],
        "n_factors_retained": results["parallel"]["n_factors_retained"],
        "observed_eigenvalues": results["parallel"]["observed_eigenvalues"],
        "simulated_mean_eigenvalues": results["parallel"]["simulated_mean_eigenvalues"],
        "simple_structure": results.get("simple_structure"),
        "factor_importance": results.get("factor_importance"),
        "factor_correlations": factor_correlations.round(4).to_dict() if factor_correlations is not None else None,
        "max_abs_factor_correlation": results.get("max_abs_factor_correlation"),
        "type_purity": results.get("type_purity"),
    }
    json_path = os.path.join(output_dir, "efa_results.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    if results.get("loadings_df") is not None:
        csv_path = os.path.join(output_dir, "efa_loadings.csv")
        results["loadings_df"].to_csv(csv_path)

    if results.get("structure_df") is not None:
        results["structure_df"].to_csv(os.path.join(output_dir, "efa_structure.csv"))

    if factor_correlations is not None:
        corr_csv_path = os.path.join(output_dir, "efa_factor_correlations.csv")
        factor_correlations.to_csv(corr_csv_path)


def run_efa(
    flagging_results_path: str,
    output_dir: str,
    n_random: int = 500,
    seed: int = 42,
    cross_loading_threshold: float = 0.40,
    mode: str = "all",
    scenario: str | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run EFA and parallel analysis on a flagging_results.json file or directory tree.

    Applies the same data pipeline as MTMM: load/merge flagging results, build the
    binary claim matrix, and compute the same pairwise tetrachoric correlations.
    Parallel analysis (Horn, 1965) against column permutations determines the number
    of retained factors. Loadings are extracted from the tetrachoric matrix with
    factor_analyzer's MINRES common factor model and rotated to an oblique Promax
    solution, which analyses only shared variance and allows the retained factors
    to correlate. Each variable is assessed for simple loading structure.

    :param flagging_results_path: Path to a flagging_results.json file, or to a
        directory that will be searched recursively for such files.
    :param output_dir: Directory to write JSON, CSV, and plot outputs.
    :param n_random: Number of random matrices for the parallel analysis simulation.
    :param seed: Seed for reproducible column permutations in the parallel analysis.
    :param cross_loading_threshold: Salient loading cutoff for simple structure
        classification; primary loadings must reach it and cross-loadings must
        stay below it.
    :param mode: Claim grouping mode from :mod:`~src.analysis.claim_grouping`
        (``"all"``, ``"falsehood_omission"``, ``"no_paltering"``,
        ``"active_vs_passive"``, or ``"no_equivocation"``).
    :param scenario: Scenario type identifier (e.g. "product_promotion");
        prefixed to plot titles when provided.
    :param verbose: Print progress messages when True.
    :return: Dict with keys: claim_matrix, parallel, loadings_df,
             factor_correlations, simple_structure, factor_importance,
             structure_df, max_abs_factor_correlation, type_purity,
             n_claims, n_vars, col_names.
    """
    flagging_results = _load_or_merge_flagging_results(flagging_results_path, verbose)

    claim_matrix = build_claim_matrix(flagging_results, mode=mode)
    if claim_matrix.empty:
        raise ValueError("No claim evaluations found in flagging_results.json")

    claim_matrix, col_names = _drop_constant_cols(claim_matrix)
    n_claims, n_vars = claim_matrix.shape

    if n_vars == 0:
        raise ValueError(
            "All columns are constant after removing invariant columns; cannot perform EFA."
        )

    if verbose:
        print(f"Claim matrix: {n_claims} claims × {n_vars} variables (after dropping constant columns)")

    parallel = _run_parallel_analysis(claim_matrix, n_random, seed)
    n_factors = parallel["n_factors_retained"]

    if verbose:
        print(f"Parallel analysis suggests retaining {n_factors} factor(s).")

    loadings_df = None
    simple_structure = None
    factor_importance = None
    structure_df = None
    factor_correlations = None
    max_abs_phi = None
    purity = None

    if n_factors > 0:
        corr = _build_efa_correlation_matrix(claim_matrix)
        loadings, phi = _fit_efa_loadings(corr, n_factors)
        factor_cols = [f"Factor {i + 1}" for i in range(n_factors)]
        loadings_df = pd.DataFrame(loadings, index=col_names, columns=factor_cols)
        factor_correlations = pd.DataFrame(phi, index=factor_cols, columns=factor_cols)
        simple_structure = _check_simple_loading_structure(loadings, col_names, cross_loading_threshold)
        structure_df = pd.DataFrame(loadings @ phi, index=col_names, columns=factor_cols)
        factor_importance = _compute_factor_importance(loadings, phi, col_names)
        max_abs_phi = max_abs_factor_correlation(factor_correlations)
        purity = type_purity(loadings_df, cross_loading_threshold)
        if verbose:
            frac = simple_structure["simple_structure_fraction"]
            print(f"Simple structure fraction: {frac:.3f} ({simple_structure['n_simple']}/{n_vars} variables)")
            total = factor_importance["total_variance"]
            for k, (v, pct) in enumerate(zip(factor_importance["factor_importance"], factor_importance["factor_importance_pct"])):
                print(f"  Factor {k + 1} importance (sum of squared structure coefficients): {v:.2f} / {total:.2f} = {pct:.1f}%")
            print(f"  Total common variance (sum of communalities): {factor_importance['total_common_pct']:.1f}%")
            for name, h2 in factor_importance["communalities"].items():
                print(f"  {name}: {h2:.3f}")
    else:
        if verbose:
            print("No factors retained; skipping loading extraction.")

    results = {
        "claim_matrix": claim_matrix,
        "parallel": parallel,
        "loadings_df": loadings_df,
        "factor_correlations": factor_correlations,
        "simple_structure": simple_structure,
        "factor_importance": factor_importance,
        "structure_df": structure_df,
        "max_abs_factor_correlation": max_abs_phi,
        "type_purity": purity,
        "n_claims": n_claims,
        "n_vars": n_vars,
        "col_names": col_names,
    }

    save_efa_results(results, output_dir)
    panel_title = format_scenario_name(scenario)
    panel_row = claim_grouping.taxonomy_label(mode)
    save_scree_grid(
        eigen_grid=[[(
            parallel["observed_eigenvalues"],
            parallel["simulated_mean_eigenvalues"],
            n_factors,
        )]],
        row_labels=[panel_row],
        col_labels=[panel_title],
        output_path=os.path.join(output_dir, "efa_scree_plot.png"),
        n_show=len(parallel["observed_eigenvalues"]),
    )
    if loadings_df is not None:
        save_efa_loading_bar_chart(
            loadings_df=loadings_df,
            cross_loading_threshold=cross_loading_threshold,
            output_path=os.path.join(output_dir, "efa_loadings_bar.png"),
            scenario=scenario,
        )
        save_efa_loading_grid(
            loadings_grid=[[loadings_df]],
            row_labels=[panel_row],
            col_labels=[panel_title],
            output_path=os.path.join(output_dir, "efa_loadings_heatmap.png"),
            cross_loading_threshold=cross_loading_threshold,
        )

    if verbose:
        print(f"Results written to {output_dir}")

    return results


def _efa_dir(trial_dir: str, mode: str) -> str:
    """
    Return the EFA output directory of one taxonomy.

    :param trial_dir: Trial results directory.
    :param mode: Claim grouping mode.
    :return: Path such as ``{trial_dir}/efa_falsehood_omission``.
    """
    return os.path.join(trial_dir, f"efa{claim_grouping.output_suffix(mode)}")


def _load_efa_results(trial_dir: str, mode: str) -> dict:
    """
    Load the saved efa_results.json of one taxonomy.

    :param trial_dir: Trial results directory.
    :param mode: Claim grouping mode.
    :return: Parsed efa_results.json.
    """
    with open(os.path.join(_efa_dir(trial_dir, mode), "efa_results.json")) as f:
        return json.load(f)


def _load_loadings(trial_dir: str, mode: str) -> pd.DataFrame | None:
    """
    Load the saved rotated loadings of one taxonomy.

    :param trial_dir: Trial results directory.
    :param mode: Claim grouping mode.
    :return: Loadings indexed by ``"{judge}::{type}"``, or None if no factor was retained.
    """
    if _load_efa_results(trial_dir, mode)["n_factors_retained"] == 0:
        return None
    return pd.read_csv(os.path.join(_efa_dir(trial_dir, mode), "efa_loadings.csv"), index_col=0)


def efa_table_row(trial_dir: str, mode: str, salient_threshold: float) -> dict:
    """
    Collect the Table 4 EFA statistics and type purity of one taxonomy from a finished trial.

    Results written before run_efa stored type purity and the largest factor
    correlation are recomputed from the saved loadings and factor correlations.

    :param trial_dir: Trial results directory.
    :param mode: Claim grouping mode.
    :param salient_threshold: Salient loading cutoff used when recomputing type purity.
    :return: Dict with k, ss_pct, max_abs_phi, and type_purity.
    """
    results = _load_efa_results(trial_dir, mode)
    k = results["n_factors_retained"]
    if k == 0:
        return {"k": 0, "ss_pct": None, "max_abs_phi": None, "type_purity": None}
    if "type_purity" in results:
        max_abs_phi = results["max_abs_factor_correlation"]
        purity = results["type_purity"]
    else:
        max_abs_phi = max_abs_factor_correlation(pd.DataFrame(results["factor_correlations"]))
        purity = type_purity(_load_loadings(trial_dir, mode), salient_threshold)
    return {
        "k": k,
        "ss_pct": 100 * results["simple_structure"]["simple_structure_fraction"],
        "max_abs_phi": max_abs_phi,
        "type_purity": purity,
    }


def _save_taxonomy_scree_grid(trials: list[tuple[str, str]], output_dir: str) -> str:
    """
    Save the paper-style scree grid: one row per trial, one column per taxonomy.

    :param trials: (trial results directory, scenario) pairs, in display order.
    :param output_dir: Directory to write the PNG to.
    :return: Path of the saved figure.
    """
    eigen_grid = []
    for trial_dir, _ in trials:
        row = []
        for _, mode in claim_grouping.TAXONOMIES:
            results = _load_efa_results(trial_dir, mode)
            row.append((
                results["observed_eigenvalues"],
                results["simulated_mean_eigenvalues"],
                results["n_factors_retained"],
            ))
        eigen_grid.append(row)
    output_path = os.path.join(output_dir, "efa_scree_grid.png")
    save_scree_grid(
        eigen_grid=eigen_grid,
        row_labels=[format_scenario_name(scenario) for _, scenario in trials],
        col_labels=[label for label, _ in claim_grouping.TAXONOMIES],
        output_path=output_path,
    )
    return output_path


def _save_taxonomy_loading_grids(
    trials: list[tuple[str, str]],
    loadings: dict[str, list[pd.DataFrame | None]],
    output_dir: str,
    salient_threshold: float,
) -> list[str]:
    """
    Save one paper-style loading grid per trial, one row per taxonomy.

    Within a taxonomy, each trial's loadings are padded to the union of
    variables across trials, so a variable dropped as constant in one scenario
    is drawn as a grey column. Taxonomies with no retained factor are omitted.

    :param trials: (trial results directory, scenario) pairs, in display order.
    :param loadings: Claim grouping mode -> loadings per trial (None if no factor was retained).
    :param output_dir: Directory to write the PNGs to.
    :param salient_threshold: Salience cutoff for outlining loadings.
    :return: Paths of the saved figures.
    """
    paths = []
    for i, (_, scenario) in enumerate(trials):
        grid = []
        row_labels = []
        for label, mode in claim_grouping.TAXONOMIES:
            if loadings[mode][i] is None:
                continue
            present = [df for df in loadings[mode] if df is not None]
            variables = sorted(set().union(*(df.index for df in present)))
            grid.append([loadings[mode][i].reindex(variables)])
            row_labels.append(label)
        if not grid:
            continue
        output_path = os.path.join(output_dir, f"efa_loading_grid_{scenario}.png")
        save_efa_loading_grid(
            loadings_grid=grid,
            row_labels=row_labels,
            col_labels=[format_scenario_name(scenario)],
            output_path=output_path,
            cross_loading_threshold=salient_threshold,
            panel_width=LOADING_GRID_PANEL_WIDTH,
            dpi=LOADING_GRID_DPI,
        )
        paths.append(output_path)
    return paths


def _save_taxonomy_membership_grid(
    trials: list[tuple[str, str]],
    loadings: dict[str, list[pd.DataFrame | None]],
    output_dir: str,
    salient_threshold: float,
) -> str | None:
    """
    Save the paper-style factor-membership grid: one row per taxonomy, one column per trial.

    Taxonomies where any trial retained no factor are omitted.

    :param trials: (trial results directory, scenario) pairs, in display order.
    :param loadings: Claim grouping mode -> loadings per trial (None if no factor was retained).
    :param output_dir: Directory to write the PNG to.
    :param salient_threshold: Salience cutoff for counting loadings.
    :return: Path of the saved figure, or None if no taxonomy has factors in every trial.
    """
    grid = []
    row_labels = []
    for label, mode in claim_grouping.TAXONOMIES:
        if any(df is None for df in loadings[mode]):
            continue
        grid.append(loadings[mode])
        row_labels.append(label)
    if not grid:
        return None
    output_path = os.path.join(output_dir, "efa_membership_grid.png")
    save_efa_membership_grid(
        loadings_grid=grid,
        row_labels=row_labels,
        col_labels=[format_scenario_name(scenario) for _, scenario in trials],
        output_path=output_path,
        cross_loading_threshold=salient_threshold,
    )
    return output_path


def save_efa_taxonomy_figures(
    trials: list[tuple[str, str]], output_dir: str, salient_threshold: float = 0.40
) -> list[str]:
    """
    Save the paper-style EFA figures that combine all taxonomies, read from each
    trial's saved per-taxonomy EFA outputs.

    Writes efa_scree_grid.png, efa_loading_grid_{scenario}.png per trial,
    efa_membership_grid.png, and efa_type_purity.png (Figure 2).

    :param trials: (trial results directory, scenario) pairs, in display order.
    :param output_dir: Directory to write the PNGs to.
    :param salient_threshold: Salient loading cutoff for outlines, membership counts, and type purity.
    :return: Paths of the saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    loadings = {
        mode: [_load_loadings(trial_dir, mode) for trial_dir, _ in trials]
        for _, mode in claim_grouping.TAXONOMIES
    }
    paths = [_save_taxonomy_scree_grid(trials, output_dir)]
    paths.extend(_save_taxonomy_loading_grids(trials, loadings, output_dir, salient_threshold))
    membership_path = _save_taxonomy_membership_grid(trials, loadings, output_dir, salient_threshold)
    if membership_path is not None:
        paths.append(membership_path)
    purity_path = os.path.join(output_dir, "efa_type_purity.png")
    save_type_purity_dot_plot(
        purities=[
            [efa_table_row(trial_dir, mode, salient_threshold)["type_purity"] for _, mode in claim_grouping.TAXONOMIES]
            for trial_dir, _ in trials
        ],
        scenarios=[scenario for _, scenario in trials],
        taxonomy_labels=[label for label, _ in claim_grouping.TAXONOMIES],
        output_path=purity_path,
    )
    paths.append(purity_path)
    return paths
