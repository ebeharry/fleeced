import math
import os

import pandas as pd

from . import claim_grouping
from .efa import efa_table_row
from .g_study import gstudy_table_row
from .mtmm import mtmm_table_row
from .plots import format_scenario_name

SECTIONS = ("gstudy", "mtmm", "efa")
TABLE_COLUMNS = [
    "scenario", "taxonomy",
    "e_rho2", "dependability_phi",
    "mean_mthm", "mean_htmonm", "mean_hthm", "c1_met", "c2_pct", "c3_pct", "c4_rho", "order",
    "k", "ss_pct", "max_abs_phi", "type_purity",
]
_LATEX_CELLS = [
    ("e_rho2", 3), ("dependability_phi", 3),
    ("mean_mthm", 2), ("mean_htmonm", 2), ("mean_hthm", 2),
    ("c2_pct", 0), ("c3_pct", 0), ("c4_rho", 2), ("order", None),
    ("k", None), ("ss_pct", 0), ("max_abs_phi", 2),
]
_LATEX_HEADER = [
    r"\begin{tabular}{lcccccccccccc}",
    r"\hline",
    r" & \multicolumn{2}{c}{\textbf{G-study}} & \multicolumn{7}{c}{\textbf{MTMM}}"
    r" & \multicolumn{3}{c}{\textbf{EFA}} \\",
    r"\cline{2-3} \cline{4-10} \cline{11-13}",
    r"\textbf{Taxonomy} & $E\rho^2$ & $\Phi$ & $\mu_{\mathrm{MTHM}}$ & $\mu_{\mathrm{HTMonoM}}$"
    r" & $\mu_{\mathrm{HTHM}}$ & C2 \% & C3 \% & C4 $\rho$ & Order & $k$ & SS \% & $|\phi|_{\max}$ \\",
]


def build_validity_table(
    trials: list[tuple[str, str]], sections: set[str], salient_threshold: float
) -> list[dict]:
    """
    Build the Table 4 rows (G-study, MTMM, EFA) for every trial and taxonomy.

    Columns of analyses not in ``sections`` are left as None so that outputs of
    analyses that were not run are never read.

    :param trials: (trial results directory, scenario) pairs, in display order.
    :param sections: Analyses to include, a subset of :data:`SECTIONS`.
    :param salient_threshold: Salient loading cutoff for type purity.
    :return: One dict per (trial, taxonomy) with keys :data:`TABLE_COLUMNS`.
    :raises ValueError: If ``sections`` contains an unknown analysis.
    """
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Unknown sections {sorted(unknown)}; expected a subset of {list(SECTIONS)}")
    rows = []
    for trial_dir, scenario in trials:
        for label, mode in claim_grouping.TAXONOMIES:
            row = dict.fromkeys(TABLE_COLUMNS)
            row["scenario"] = scenario
            row["taxonomy"] = label
            if "gstudy" in sections:
                row.update(gstudy_table_row(trial_dir, scenario, mode))
            if "mtmm" in sections:
                row.update(mtmm_table_row(trial_dir, mode))
            if "efa" in sections:
                row.update(efa_table_row(trial_dir, mode, salient_threshold))
            rows.append(row)
    return rows


def _latex_cell(row: dict, column: str, digits: int | None) -> str:
    """
    Format one Table 4 cell for LaTeX.

    :param row: Table row from :func:`build_validity_table`.
    :param column: Column key.
    :param digits: Decimal places, or None for integers and Y/N flags.
    :return: Cell text; "N/A" for an undefined C4, "--" for anything not computed.
    """
    value = row[column]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A" if column == "c4_rho" and row["mean_mthm"] is not None else "--"
    if isinstance(value, bool):
        return "Y" if value else "N"
    if digits is None:
        return str(value)
    text = f"{value:.{digits}f}"
    return f"$-${text[1:]}" if text.startswith("-") else text


def _latex_table(rows: list[dict]) -> str:
    """
    Render Table 4 as a LaTeX tabular, one block per scenario.

    :param rows: Table rows from :func:`build_validity_table`.
    :return: LaTeX source of the tabular environment.
    """
    lines = list(_LATEX_HEADER)
    for scenario in dict.fromkeys(row["scenario"] for row in rows):
        label = format_scenario_name(scenario).replace("&", r"\&")
        lines.append(r"\hline")
        lines.append(rf"\multicolumn{{13}}{{l}}{{\textit{{{label}}}}} \\")
        for row in rows:
            if row["scenario"] == scenario:
                cells = [_latex_cell(row, column, digits) for column, digits in _LATEX_CELLS]
                lines.append(" & ".join([row["taxonomy"], *cells]) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def save_validity_table(
    trials: list[tuple[str, str]], output_dir: str, sections: set[str], salient_threshold: float = 0.40
) -> list[str]:
    """
    Write Table 4 as validity_table.csv (unrounded values) and validity_table.tex.

    :param trials: (trial results directory, scenario) pairs, in display order.
    :param output_dir: Directory to write into.
    :param sections: Analyses to include, a subset of :data:`SECTIONS`.
    :param salient_threshold: Salient loading cutoff for type purity.
    :return: Paths of the written files.
    """
    rows = build_validity_table(trials, sections, salient_threshold)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "validity_table.csv")
    pd.DataFrame(rows, columns=TABLE_COLUMNS).to_csv(csv_path, index=False)
    tex_path = os.path.join(output_dir, "validity_table.tex")
    with open(tex_path, "w") as f:
        f.write(_latex_table(rows))
    return [csv_path, tex_path]
