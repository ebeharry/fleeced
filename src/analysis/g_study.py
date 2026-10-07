import json
import math
import os
import numpy as np
import pandas as pd
if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map
from rpy2.rinterface_lib.sexp import NULLType
if not hasattr(NULLType, "rx2"):
    # Mirrors R's `NULL$x` returning NULL: pymer4's Lmer.fit() reaches into
    # summary$optinfo$conv$lme4$messages unconditionally, which is R NULL
    # (not an empty list) when lme4 reports no convergence messages.
    NULLType.rx2 = lambda self, *args, **kwargs: self
from pymer4.models import Lmer

from . import claim_grouping
from .plots import save_dstudy_plot

def _apply_logit(n_flagged: int, n_claims: int) -> float:
    """
    Logit-transform a flag count with continuity correction for boundary rates.

    :param n_flagged: Number of flagged claims.
    :param n_claims: Total claims evaluated.
    :return: Log-odds of the (corrected) flag rate.
    """
    if n_claims == 0:
        return float("nan")
    p = n_flagged / n_claims
    if p == 0.0:
        p = 0.5 / (n_claims + 1)
    elif p == 1.0:
        p = (n_claims + 0.5) / (n_claims + 1)
    return math.log(p / (1.0 - p))


def _extract_conv_flag_rates(flagging_data: dict, mode: str = "all") -> list[dict]:
    """
    Return one record per (judge, conv_id) from a flagging_results dict.

    :param flagging_data: Parsed flagging_results.json content.
    :param mode: Claim grouping mode from :mod:`~src.analysis.claim_grouping`
        determining which deception traits count towards "flagged".
    :return: List of dicts with keys judge, conv_id, n_claims, n_flagged, flag_rate.
    """
    groups = {}
    for entry in flagging_data.get("claim_evaluations", []):
        key = (entry["flagging_model"], entry["conversation_file"])
        if key not in groups:
            groups[key] = []
        groups[key].append(claim_grouping.is_flagged_for_mode(entry.get("deception_indicators", {}), mode))
    records = []
    for (judge, conv_id), flags in groups.items():
        n_claims = len(flags)
        n_flagged = sum(flags)
        records.append({
            "judge": judge,
            "conv_id": conv_id,
            "n_claims": n_claims,
            "n_flagged": n_flagged,
            "flag_rate": n_flagged / n_claims,
        })
    return records


def _scan_flagging_files(results_dir: str, condition: str) -> list[dict]:
    """
    Walk results_dir/{model}/{condition}/{sub_scenario}/evaluations/flagging_results.json.

    :param results_dir: Top-level pilot run directory.
    :param condition: Condition subdirectory name (e.g. "baseline").
    :return: List of dicts with keys agent_model, sub_scenario, path.
    """
    entries = []
    if not os.path.isdir(results_dir):
        raise ValueError(f"results_dir does not exist: {results_dir}")
    for model_name in sorted(os.listdir(results_dir)):
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_dir):
            continue
        condition_dir = os.path.join(model_dir, condition)
        if not os.path.isdir(condition_dir):
            continue
        for subtype_name in sorted(os.listdir(condition_dir)):
            subtype_dir = os.path.join(condition_dir, subtype_name)
            flagging_path = os.path.join(subtype_dir, "evaluations", "flagging_results.json")
            if os.path.isfile(flagging_path):
                entries.append({
                    "agent_model": model_name,
                    "sub_scenario": subtype_name,
                    "path": flagging_path,
                })
    return entries


def build_input_matrix(
    results_dir: str,
    scenario: str,
    condition: str = "baseline",
    mode: str = "all",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Collect per-(M, R, I, S) flag rates and return the logit-transformed input matrix.

    :param results_dir: Path to pilot run directory (e.g. "results/pilot_product_promotion_generation").
    :param scenario: Scenario label stored in metadata column only.
    :param condition: Condition subdirectory to read from.
    :param mode: Claim grouping mode from :mod:`~src.analysis.claim_grouping`
        (``"all"``, ``"falsehood_omission"``, ``"no_paltering"``,
        ``"active_vs_passive"``, or ``"no_equivocation"``).
    :param verbose: If True, print discovery progress.
    :return: DataFrame with columns [agent_model, judge, conv_id, sub_scenario,
             n_claims, n_flagged, flag_rate, logit_dr].
    """
    file_entries = _scan_flagging_files(results_dir, condition)
    if not file_entries:
        raise ValueError(f"No flagging_results.json files found under {results_dir}/{condition}/")

    rows = []
    for entry in file_entries:
        with open(entry["path"]) as f:
            flagging_data = json.load(f)
        conv_records = _extract_conv_flag_rates(flagging_data, mode=mode)
        for rec in conv_records:
            rows.append({
                "agent_model": entry["agent_model"],
                "sub_scenario": entry["sub_scenario"],
                "judge": rec["judge"],
                "conv_id": rec["conv_id"],
                "n_claims": rec["n_claims"],
                "n_flagged": rec["n_flagged"],
                "flag_rate": rec["flag_rate"],
                "logit_dr": _apply_logit(rec["n_flagged"], rec["n_claims"]),
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["logit_dr"])

    if verbose:
        n_M = df["agent_model"].nunique()
        n_R = df["judge"].nunique()
        n_S = df["sub_scenario"].nunique()
        n_I = df.groupby("sub_scenario")["conv_id"].nunique().mean()
        print(f"Input matrix: {len(df)} rows | {n_M} models, {n_R} judges, {n_S} sub-scenarios, ~{n_I:.1f} convs/sub-scenario")

    return df


def _check_balance(df: pd.DataFrame) -> tuple[bool, int, int, int, float]:
    """
    Check whether the design is balanced and return design counts.

    :param df: Input matrix from build_input_matrix.
    :return: (is_balanced, n_M, n_R, n_S, n_I_mean).
    """
    n_M = df["agent_model"].nunique()
    n_R = df["judge"].nunique()
    n_S = df["sub_scenario"].nunique()
    convs_per_sub = df.groupby("sub_scenario")["conv_id"].nunique()
    n_I_mean = convs_per_sub.mean()
    cell_counts = df.groupby(["agent_model", "judge", "sub_scenario"]).size()
    is_balanced = bool(convs_per_sub.nunique() == 1 and cell_counts.nunique() == 1)
    return is_balanced, n_M, n_R, n_S, n_I_mean


def _reml_group_to_key(group_name: str) -> str | None:
    """
    Map an lme4 random-effect grouping factor name to a G-study component key.

    Normalises dots to underscores and treats interaction parts as an unordered
    set so that R's potential reordering of factor names does not matter.

    :param group_name: Grouping factor label returned by lme4 / VarCorr.
    :return: One of the 10 non-residual component keys, or None if unrecognised.
    """
    parts = frozenset(group_name.replace(".", "_").split(":"))
    mapping = {
        frozenset({"agent_model"}): "M",
        frozenset({"judge"}): "R",
        frozenset({"sub_scenario"}): "S",
        frozenset({"sub_scenario", "conv_id"}): "I_S",
        frozenset({"conv_id"}): "I_S",
        frozenset({"agent_model", "judge"}): "MR",
        frozenset({"agent_model", "sub_scenario"}): "MS",
        frozenset({"judge", "sub_scenario"}): "RS",
        frozenset({"agent_model", "sub_scenario", "conv_id"}): "MI_S",
        frozenset({"agent_model", "conv_id"}): "MI_S",
        frozenset({"judge", "sub_scenario", "conv_id"}): "RI_S",
        frozenset({"judge", "conv_id"}): "RI_S",
        frozenset({"agent_model", "judge", "sub_scenario"}): "MRS",
    }
    return mapping.get(parts)


def _iter_reml_vc_rows(vc_df: pd.DataFrame) -> list[tuple[str, float]]:
    """
    Yield (group_name, variance) pairs from a variance-component table.

    pymer4 exposes this table as either `ranef_var` (grouping factor in a
    "grp" column, variance in "vcov") or `result_vc` (grouping factor as the
    row index, variance in a "Var" column or the first column).

    :param vc_df: Variance-component DataFrame from a fitted Lmer model.
    :return: List of (group_name, non-negative variance) tuples.
    """
    if "grp" in vc_df.columns:
        return [
            (str(row["grp"]), max(float(row.get("vcov", 0.0)), 0.0))
            for _, row in vc_df.iterrows()
        ]
    var_col = "Var" if "Var" in vc_df.columns else vc_df.columns[0]
    return [(str(grp), max(float(row[var_col]), 0.0)) for grp, row in vc_df.iterrows()]


def _build_reml_formula(df: pd.DataFrame) -> str:
    """
    Build the lme4 random-effects formula, dropping any facet with fewer than 2 levels.

    A facet with a single observed level contributes no estimable variance, so its
    main-effect and interaction terms are omitted rather than passed to lme4.

    :param df: Input matrix from build_input_matrix.
    :return: lme4-compatible formula string for logit_dr.
    """
    has_M = df["agent_model"].nunique() >= 2
    has_R = df["judge"].nunique() >= 2
    has_S = df["sub_scenario"].nunique() >= 2

    i_group = "sub_scenario:conv_id" if has_S else "conv_id"
    terms = [f"(1 | {i_group})"]
    if has_M:
        terms.append("(1 | agent_model)")
        terms.append(f"(1 | agent_model:{i_group})")
    if has_R:
        terms.append("(1 | judge)")
        terms.append(f"(1 | judge:{i_group})")
    if has_S:
        terms.append("(1 | sub_scenario)")
    if has_M and has_R:
        terms.append("(1 | agent_model:judge)")
    if has_M and has_S:
        terms.append("(1 | agent_model:sub_scenario)")
    if has_R and has_S:
        terms.append("(1 | judge:sub_scenario)")
    if has_M and has_R and has_S:
        terms.append("(1 | agent_model:judge:sub_scenario)")

    return "logit_dr ~ 1 + " + " + ".join(terms)


def _fit_reml_components(df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Estimate variance components via REML for the M × R × (I:S) design using lme4.

    Fits a maximal random-intercepts model via pymer4's lmer interface, dropping
    any facet (M, R, or S) that has fewer than 2 observed levels since its
    variance is not estimable. Variance components are extracted from the fitted
    model's ranef_var table; the residual sigma² is read from model.sig when not
    present in that table.

    :param df: Input matrix from build_input_matrix.
    :param verbose: If True, print any lme4 convergence messages.
    :return: Dict of {source: sigma_squared} for all 11 variance components.
    """
    y = df["logit_dr"].values
    keys = ["M", "R", "S", "I_S", "MR", "MS", "RS", "MI_S", "RI_S", "MRS", "e"]

    if np.var(y) < 1e-10:
        return {k: 0.0 for k in keys}

    formula = _build_reml_formula(df)

    model = Lmer(formula, data=df)
    model.fit(summary=False)

    if verbose and hasattr(model, "show_logs"):
        model.show_logs()

    result = {k: 0.0 for k in keys}

    vc_df = getattr(model, "ranef_var", None)
    if vc_df is None:
        vc_df = getattr(model, "result_vc", None)
    if isinstance(vc_df, pd.DataFrame):
        for grp, val in _iter_reml_vc_rows(vc_df):
            if grp.lower() == "residual":
                result["e"] = val
            else:
                key = _reml_group_to_key(grp)
                if key:
                    result[key] = val

    if result["e"] == 0.0 and hasattr(model, "sig"):
        result["e"] = max(float(model.sig) ** 2, 0.0)

    return result


def fit_gstudy(df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Estimate variance components for the M × R × (I:S) design via REML.

    Negative component estimates are floored to 0 by the optimizer bounds.

    :param df: Output of build_input_matrix.
    :param verbose: If True, print variance components table.
    :return: Dict with keys variance_components, design_counts, is_balanced.
    """
    is_balanced, n_M, n_R, n_S, n_I_mean = _check_balance(df)
    n_I = int(round(n_I_mean))

    if not is_balanced:
        print(f"Warning: unbalanced design (mean n_I={n_I_mean:.2f}); REML handles this correctly.")

    if n_I < 2:
        raise ValueError(
            f"Design too sparse for variance decomposition: "
            f"n_M={n_M}, n_R={n_R}, n_S={n_S}, n_I={n_I}. Need ≥2 conversations per sub-scenario."
        )

    dropped = [name for name, n in [("M", n_M), ("R", n_R), ("S", n_S)] if n < 2]
    if dropped:
        print(
            f"Warning: facet(s) {', '.join(dropped)} have <2 levels "
            f"(n_M={n_M}, n_R={n_R}, n_S={n_S}); dropping from the variance decomposition."
        )

    var_components = _fit_reml_components(df, verbose=verbose)

    if verbose:
        print("\nVariance components (M × R × (I:S) design, REML):")
        total_var = sum(var_components.values())
        for source, sigma_sq in var_components.items():
            pct = 100 * sigma_sq / total_var if total_var > 0 else 0.0
            print(f"  σ²({source:<4}) = {sigma_sq:.6f}  ({pct:.1f}%)")
        print(f"  Total        = {total_var:.6f}")

    return {
        "variance_components": var_components,
        "design_counts": {"n_M": n_M, "n_R": n_R, "n_S": n_S, "n_I": n_I},
        "is_balanced": is_balanced,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    """
    Divide numerator by denominator, returning 0.0 for a non-positive denominator.

    :param numerator: Ratio numerator.
    :param denominator: Ratio denominator.
    :return: numerator / denominator, or 0.0 if denominator <= 0.
    """
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def compute_g_coefficient(
    var_components: dict,
    n_judges: int,
    n_convs: int,
    n_subtypes: int,
) -> float:
    """
    Compute the generalizability coefficient (relative decisions) for given design sizes.

    :param var_components: Dict of variance component estimates from fit_gstudy.
    :param n_judges: Number of flagging judges (nR).
    :param n_convs: Number of conversations per sub-scenario (nI).
    :param n_subtypes: Number of sub-scenarios (nS).
    :return: Generalizability coefficient in [0, 1].
    """
    vc = var_components
    universe_score = vc["M"]
    relative_error = (
        vc["MR"] / n_judges
        + vc["MS"] / n_subtypes
        + vc["MI_S"] / (n_convs * n_subtypes)
        + vc["MRS"] / (n_judges * n_subtypes)
        + vc["e"] / (n_judges * n_convs * n_subtypes)
    )
    return _safe_ratio(universe_score, universe_score + relative_error)


def compute_phi_coefficient(
    var_components: dict,
    n_judges: int,
    n_convs: int,
    n_subtypes: int,
) -> float:
    """
    Compute the phi coefficient (index of dependability, absolute decisions).

    :param var_components: Dict of variance component estimates from fit_gstudy.
    :param n_judges: Number of flagging judges (nR).
    :param n_convs: Number of conversations per sub-scenario (nI).
    :param n_subtypes: Number of sub-scenarios (nS).
    :return: Phi coefficient in [0, 1].
    """
    vc = var_components
    universe_score = vc["M"]
    absolute_error = (
        vc["R"] / n_judges
        + vc["S"] / n_subtypes
        + vc["I_S"] / (n_convs * n_subtypes)
        + vc["MR"] / n_judges
        + vc["MS"] / n_subtypes
        + vc["MI_S"] / (n_convs * n_subtypes)
        + vc["RS"] / (n_judges * n_subtypes)
        + vc["RI_S"] / (n_judges * n_convs * n_subtypes)
        + vc["MRS"] / (n_judges * n_subtypes)
        + vc["e"] / (n_judges * n_convs * n_subtypes)
    )
    return _safe_ratio(universe_score, universe_score + absolute_error)


_DSTUDY_FACET_POSITIONS = {"n_judges": 0, "n_convs": 1, "n_subtypes": 2, "n_models": None}


def _sweep_dstudy_facet(
    var_components: dict,
    facet: str,
    values: list[int],
    n_R0: int,
    n_I0: int,
    n_S0: int,
) -> list[dict]:
    """
    Compute G and phi coefficients while sweeping a single facet, holding the others fixed.

    :param var_components: Dict from fit_gstudy["variance_components"].
    :param facet: One of "n_judges", "n_convs", "n_subtypes", "n_models".
    :param values: Values of the varied facet to sweep over.
    :param n_R0: Observed number of judges.
    :param n_I0: Observed number of conversations per sub-scenario.
    :param n_S0: Observed number of sub-scenarios.
    :return: List of row dicts with keys varied_facet, n, g_coefficient, phi_coefficient.
    """
    position = _DSTUDY_FACET_POSITIONS[facet]
    rows = []
    for n in values:
        n_R, n_I, n_S = n_R0, n_I0, n_S0
        if position == 0:
            n_R = n
        elif position == 1:
            n_I = n
        elif position == 2:
            n_S = n
        rows.append({
            "varied_facet": facet,
            "n": n,
            "g_coefficient": compute_g_coefficient(var_components, n_R, n_I, n_S),
            "phi_coefficient": compute_phi_coefficient(var_components, n_R, n_I, n_S),
        })
    return rows


def run_dstudy(
    var_components: dict,
    observed_n: dict,
    judge_range: list[int] | None = None,
    conv_range: list[int] | None = None,
    subtype_range: list[int] | None = None,
    model_range: list[int] | None = None,
) -> pd.DataFrame:
    """
    Project G and phi coefficients under alternative design configurations.

    Varies one facet at a time, holding the others at their observed values.
    n_models is included for completeness; nM does not appear in either
    coefficient formula so the n_models panel will be flat lines.

    :param var_components: Dict from fit_gstudy["variance_components"].
    :param observed_n: Dict with keys n_judges, n_convs, n_subtypes, and
        optionally n_models.
    :param judge_range: nR values to sweep (default 1 to observed+3).
    :param conv_range: nI values to sweep.
    :param subtype_range: nS values to sweep.
    :param model_range: nM values to sweep (default 1 to observed+3).
    :return: DataFrame with columns [varied_facet, n, g_coefficient, phi_coefficient].
    """
    n_R0 = observed_n["n_judges"]
    n_I0 = observed_n["n_convs"]
    n_S0 = observed_n["n_subtypes"]
    n_M0 = observed_n.get("n_models", 1)

    if judge_range is None:
        judge_range = list(range(1, n_R0 + 4))
    if conv_range is None:
        conv_range = list(range(1, n_I0 + 6))
    if subtype_range is None:
        subtype_range = list(range(1, n_S0 + 4))
    if model_range is None:
        model_range = list(range(1, n_M0 + 4))

    facet_ranges = [
        ("n_judges", judge_range),
        ("n_convs", conv_range),
        ("n_subtypes", subtype_range),
        ("n_models", model_range),
    ]
    rows = []
    for facet, values in facet_ranges:
        rows.extend(_sweep_dstudy_facet(var_components, facet, values, n_R0, n_I0, n_S0))

    return pd.DataFrame(rows)


def save_gstudy_results(
    scenario: str,
    gstudy_result: dict,
    g_coefficient: float,
    phi_coefficient: float,
    dstudy_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Write g_study_{scenario}.json and d_study_{scenario}.csv to output_dir.

    :param scenario: Scenario label used in filenames.
    :param gstudy_result: Full output of fit_gstudy.
    :param g_coefficient: Observed-design G-coefficient (relative decisions).
    :param phi_coefficient: Observed-design phi coefficient (absolute decisions).
    :param dstudy_df: Output of run_dstudy.
    :param output_dir: Directory to write output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    payload = {
        "scenario": scenario,
        "g_coefficient_observed": g_coefficient,
        "phi_coefficient_observed": phi_coefficient,
        "variance_components": gstudy_result["variance_components"],
        "design_counts": gstudy_result["design_counts"],
        "is_balanced": gstudy_result["is_balanced"],
    }
    json_path = os.path.join(output_dir, f"g_study_{scenario}.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(output_dir, f"d_study_{scenario}.csv")
    dstudy_df.to_csv(csv_path, index=False)

    print(f"G-study results: {json_path}")
    print(f"D-study table:   {csv_path}")


def _print_dstudy_best(dstudy_df: pd.DataFrame) -> None:
    """
    Print the highest G and phi coefficients reached per varied facet.

    :param dstudy_df: Output of run_dstudy.
    """
    print("\nD-study: best G and Φ per facet:")
    for facet in ["n_judges", "n_convs", "n_subtypes", "n_models"]:
        sub = dstudy_df[dstudy_df["varied_facet"] == facet]
        if sub.empty:
            continue
        best = sub.loc[sub["g_coefficient"].idxmax()]
        suffix = "  [invariant: nM does not enter Eρ²/Φ formulas]" if sub["g_coefficient"].nunique() == 1 else ""
        print(
            f"  {facet:<12}: G={best['g_coefficient']:.4f}  "
            f"Φ={best['phi_coefficient']:.4f}  (n={int(best['n'])}){suffix}"
        )


def run_gstudy(
    results_dir: str,
    scenario: str,
    output_dir: str,
    condition: str = "baseline",
    target: float = 0.85,
    mode: str = "all",
    verbose: bool = False,
) -> dict:
    """
    Full G-study pipeline: build matrix, fit model, compute G-coefficient, run D-study, save outputs.

    :param results_dir: Pilot run directory.
    :param scenario: Scenario label.
    :param output_dir: Directory for output files and plots.
    :param condition: Condition subdirectory.
    :param target: G-coefficient target for D-study reference line.
    :param mode: Claim grouping mode from :mod:`~src.analysis.claim_grouping`
        (``"all"``, ``"falsehood_omission"``, ``"no_paltering"``,
        ``"active_vs_passive"``, or ``"no_equivocation"``).
    :param verbose: If True, print progress and diagnostics.
    :return: Dict with keys matrix, gstudy_result, g_coefficient, dstudy_df.
    """
    if verbose:
        print(f"\n=== G-Study: {scenario} ({mode}) ===\n")

    df = build_input_matrix(results_dir, scenario, condition=condition, mode=mode, verbose=verbose)
    gstudy_result = fit_gstudy(df, verbose=verbose)

    vc = gstudy_result["variance_components"]
    dc = gstudy_result["design_counts"]
    g_coef = compute_g_coefficient(vc, dc["n_R"], dc["n_I"], dc["n_S"])
    phi_coef = compute_phi_coefficient(vc, dc["n_R"], dc["n_I"], dc["n_S"])

    if verbose:
        print(f"\nG-coefficient (observed design): {g_coef:.4f}  (target: {target})")
        print(f"Φ-coefficient (observed design): {phi_coef:.4f}")

    observed_n = {"n_judges": dc["n_R"], "n_convs": dc["n_I"], "n_subtypes": dc["n_S"], "n_models": dc["n_M"]}
    dstudy_df = run_dstudy(vc, observed_n)

    _print_dstudy_best(dstudy_df)

    plot_dir = os.path.join(output_dir, "plots")
    save_dstudy_plot(dstudy_df, plot_dir, scenario, target=target)
    save_gstudy_results(scenario, gstudy_result, g_coef, phi_coef, dstudy_df, output_dir)

    return {
        "matrix": df,
        "gstudy_result": gstudy_result,
        "g_coefficient": g_coef,
        "phi_coefficient": phi_coef,
        "dstudy_df": dstudy_df,
    }


def gstudy_table_row(trial_dir: str, scenario: str, mode: str) -> dict:
    """
    Collect the Table 4 G-study coefficients of one taxonomy from a finished trial.

    :param trial_dir: Trial results directory.
    :param scenario: Scenario type identifier used in the result filename.
    :param mode: Claim grouping mode.
    :return: Dict with e_rho2 and dependability_phi.
    """
    path = os.path.join(trial_dir, f"gstudy{claim_grouping.output_suffix(mode)}", f"g_study_{scenario}.json")
    with open(path) as f:
        results = json.load(f)
    return {
        "e_rho2": results["g_coefficient_observed"],
        "dependability_phi": results["phi_coefficient_observed"],
    }
