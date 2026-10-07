import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis import claim_grouping
from src.analysis.validity_table import (
    TABLE_COLUMNS,
    _latex_cell,
    _latex_table,
    build_validity_table,
    save_validity_table,
)

SCENARIO = "loan_qa"


def _write_json(path: str, data: dict) -> None:
    """
    Write a dict as JSON, creating parent directories.

    :param path: Destination file path.
    :param data: Content to write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _write_trial(trial_dir: str, store_efa_metrics: bool = True, n_factors: int = 2) -> None:
    """
    Write minimal G-study, MTMM, and EFA outputs for every taxonomy of a finished trial.

    :param trial_dir: Directory to create the trial in.
    :param store_efa_metrics: If False, omit type_purity and max_abs_factor_correlation from
        efa_results.json and write efa_loadings.csv, as in results from before they were stored.
    :param n_factors: Factors retained in every EFA run.
    """
    for _, mode in claim_grouping.TAXONOMIES:
        suffix = claim_grouping.output_suffix(mode)
        _write_json(
            os.path.join(trial_dir, f"gstudy{suffix}", f"g_study_{SCENARIO}.json"),
            {"g_coefficient_observed": 0.91, "phi_coefficient_observed": 0.72},
        )
        _write_json(
            os.path.join(trial_dir, f"mtmm{suffix}", "mtmm_results.json"),
            {
                "subtypes": claim_grouping.mode_subtypes(mode),
                "validity": {
                    "mean_mthm": 0.6,
                    "mean_htmonm": 0.5,
                    "mean_hthm": 0.3,
                    "mthm_meets_threshold": True,
                    "mthm_significant_gt_zero": True,
                    "mthm_pct_exceed_row_col_hthm": 0.4333,
                    "mthm_pct_exceed_local_htmonm": 0.2167,
                    "block_ordering_holds": True,
                },
                "consistency": {"mean_consistency": -0.05},
            },
        )
        efa_results = {
            "n_factors_retained": n_factors,
            "simple_structure": {"simple_structure_fraction": 0.8333},
            "factor_correlations": {
                "Factor 1": {"Factor 1": 1.0, "Factor 2": -0.47},
                "Factor 2": {"Factor 1": -0.47, "Factor 2": 1.0},
            },
        }
        if store_efa_metrics:
            efa_results["max_abs_factor_correlation"] = 0.47
            efa_results["type_purity"] = 64.0
        _write_json(os.path.join(trial_dir, f"efa{suffix}", "efa_results.json"), efa_results)
        if not store_efa_metrics:
            pd.DataFrame(
                {"Factor 1": [0.8, 0.5, 0.45], "Factor 2": [0.0, 0.1, 0.9]},
                index=["j1::Falsehood", "j2::Falsehood", "j1::Omission"],
            ).to_csv(os.path.join(trial_dir, f"efa{suffix}", "efa_loadings.csv"))


@pytest.fixture
def trials(tmp_path) -> list[tuple[str, str]]:
    path = str(tmp_path / "trial")
    _write_trial(path)
    return [(path, SCENARIO)]


class TestBuildValidityTable:
    def test_one_row_per_taxonomy(self, trials):
        rows = build_validity_table(trials, {"gstudy", "mtmm", "efa"}, 0.40)
        assert [row["taxonomy"] for row in rows] == [label for label, _ in claim_grouping.TAXONOMIES]
        assert all(set(row) == set(TABLE_COLUMNS) for row in rows)

    def test_reads_all_sections(self, trials):
        row = build_validity_table(trials, {"gstudy", "mtmm", "efa"}, 0.40)[0]
        assert row["e_rho2"] == pytest.approx(0.91)
        assert row["dependability_phi"] == pytest.approx(0.72)
        assert row["mean_mthm"] == pytest.approx(0.6)
        assert row["c1_met"] is True
        assert row["c2_pct"] == pytest.approx(43.33)
        assert row["c3_pct"] == pytest.approx(21.67)
        assert row["order"] is True
        assert row["k"] == 2
        assert row["ss_pct"] == pytest.approx(83.33)
        assert row["max_abs_phi"] == pytest.approx(0.47)
        assert row["type_purity"] == pytest.approx(64.0)

    @pytest.mark.parametrize("label, defined", [
        ("4-type", True),
        ("Binary", False),
        ("IDT", True),
        ("Rogers", True),
        ("Active/Passive", False),
    ])
    def test_c4_undefined_for_two_trait_taxonomies(self, trials, label, defined):
        rows = build_validity_table(trials, {"mtmm"}, 0.40)
        row = next(r for r in rows if r["taxonomy"] == label)
        assert (row["c4_rho"] is not None) == defined

    def test_sections_not_listed_are_empty(self, trials):
        row = build_validity_table(trials, {"efa"}, 0.40)[0]
        assert row["e_rho2"] is None
        assert row["mean_mthm"] is None
        assert row["k"] == 2

    def test_no_factors_leaves_efa_statistics_empty(self, tmp_path):
        path = str(tmp_path / "trial")
        _write_trial(path, n_factors=0)
        row = build_validity_table([(path, SCENARIO)], {"efa"}, 0.40)[0]
        assert row["k"] == 0
        assert row["ss_pct"] is None
        assert row["max_abs_phi"] is None
        assert row["type_purity"] is None

    def test_recomputes_metrics_missing_from_older_results(self, tmp_path):
        path = str(tmp_path / "trial")
        _write_trial(path, store_efa_metrics=False)
        row = build_validity_table([(path, SCENARIO)], {"efa"}, 0.40)[0]
        assert row["max_abs_phi"] == pytest.approx(0.47)
        assert row["type_purity"] == pytest.approx(100 * 3 / 4)

    def test_raises_on_unknown_section(self, trials):
        with pytest.raises(ValueError, match="Unknown sections"):
            build_validity_table(trials, {"tables"}, 0.40)


class TestLatexTable:
    @pytest.mark.parametrize("column, value, digits, expected", [
        ("e_rho2", 0.9094, 3, "0.909"),
        ("c2_pct", 43.33, 0, "43"),
        ("c4_rho", -0.0512, 2, "$-$0.05"),
        ("order", True, None, "Y"),
        ("order", False, None, "N"),
        ("k", 3, None, "3"),
        ("k", None, None, "--"),
    ])
    def test_cell_formatting(self, column, value, digits, expected):
        row = dict.fromkeys(TABLE_COLUMNS)
        row[column] = value
        assert _latex_cell(row, column, digits) == expected

    @pytest.mark.parametrize("mean_mthm, expected", [(0.6, "N/A"), (None, "--")])
    def test_missing_c4_marks_na_only_when_mtmm_ran(self, mean_mthm, expected):
        row = dict.fromkeys(TABLE_COLUMNS)
        row["mean_mthm"] = mean_mthm
        assert _latex_cell(row, "c4_rho", 2) == expected

    def test_scenario_block_escapes_ampersand(self, trials):
        rows = build_validity_table(trials, {"gstudy", "mtmm", "efa"}, 0.40)
        latex = _latex_table(rows)
        assert r"\textit{Loan Q\&A}" in latex
        assert latex.count(r" \\") == len(claim_grouping.TAXONOMIES) + 3


class TestSaveValidityTable:
    def test_writes_csv_and_tex(self, trials, tmp_path):
        out_dir = str(tmp_path / "out")
        paths = save_validity_table(trials, out_dir, {"gstudy"})
        assert sorted(os.path.basename(p) for p in paths) == ["validity_table.csv", "validity_table.tex"]
        table = pd.read_csv(os.path.join(out_dir, "validity_table.csv"))
        assert list(table.columns) == TABLE_COLUMNS
        assert table["mean_mthm"].isna().all()
