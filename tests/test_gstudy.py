import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.g_study import (
    _apply_logit,
    _check_balance,
    _extract_conv_flag_rates,
    _reml_group_to_key,
    _scan_flagging_files,
    build_input_matrix,
    compute_g_coefficient,
    compute_phi_coefficient,
    fit_gstudy,
    run_dstudy,
    run_gstudy,
    save_gstudy_results,
)



def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_flagging_results(judges: list[str], conv_ids: list[str], flag: bool = False) -> dict:
    """
    Minimal flagging_results.json with one claim per (judge, conv_id).
    """
    evals = []
    for judge in judges:
        for conv_id in conv_ids:
            evals.append({
                "flagging_model": judge,
                "conversation_file": conv_id,
                "claim_id": f"{conv_id}_turn1",
                "claim_type": "product",
                "turn": 1,
                "claim": "Test claim",
                "deception_indicators": {"Falsehood": flag, "Omission": False,
                                         "Equivocation": False, "Paltering": False},
                "reasons": {},
                "is_flagged": flag,
                "raw_response": "",
                "timestamp": "2026-01-01 00:00:00",
            })
    return {
        "created_timestamp": "2026-01-01 00:00:00",
        "flagging_models": judges,
        "summary": None,
        "claim_evaluations": evals,
        "errors": [],
    }


def _make_pilot_dir(tmp_path, models, judges, subtypes, conv_ids, flag_pattern=None) -> str:
    """
    Build a results_dir/{model}/baseline/{subtype}/evaluations/flagging_results.json tree.

    flag_pattern: callable(model, subtype, judge, conv_id) -> bool, defaults to False.
    """
    for model in models:
        for subtype in subtypes:
            evals = []
            for judge in judges:
                for conv_id in conv_ids:
                    flag = flag_pattern(model, subtype, judge, conv_id) if flag_pattern else False
                    n_claims = 4
                    n_flagged = n_claims if flag else 0
                    for k in range(n_claims):
                        evals.append({
                            "flagging_model": judge,
                            "conversation_file": conv_id,
                            "claim_id": f"{conv_id}_turn{k}",
                            "claim_type": "product",
                            "turn": k,
                            "claim": "Test claim",
                            "deception_indicators": {
                                "Falsehood": flag, "Omission": False,
                                "Equivocation": False, "Paltering": False,
                            },
                            "reasons": {},
                            "is_flagged": flag,
                            "raw_response": "",
                            "timestamp": "2026-01-01 00:00:00",
                        })
            data = {
                "created_timestamp": "2026-01-01",
                "flagging_models": judges,
                "summary": None,
                "claim_evaluations": evals,
                "errors": [],
            }
            path = str(tmp_path / model / "baseline" / subtype / "evaluations" / "flagging_results.json")
            _write_json(path, data)
    return str(tmp_path)


def _make_balanced_df(
    n_M: int = 2,
    n_R: int = 2,
    n_S: int = 2,
    n_I: int = 2,
    value_fn=None,
) -> pd.DataFrame:
    """
    Build a fully crossed (M, R, I, S) DataFrame with logit_dr set by value_fn(m, r, i, s).
    Defaults to all zeros.
    """
    rows = []
    for m in range(n_M):
        for r in range(n_R):
            for s in range(n_S):
                for i in range(n_I):
                    val = value_fn(m, r, i, s) if value_fn else 0.0
                    rows.append({
                        "agent_model": f"model_{m}",
                        "judge": f"judge_{r}",
                        "sub_scenario": f"sub_{s}",
                        "conv_id": f"conv_{s}_{i}",
                        "n_claims": 4,
                        "n_flagged": 0,
                        "flag_rate": 0.0,
                        "logit_dr": val,
                    })
    return pd.DataFrame(rows)



class TestApplyLogit:
    def test_normal_rate(self):
        val = _apply_logit(2, 4)
        assert val == pytest.approx(math.log(0.5 / 0.5), abs=1e-9)

    def test_zero_rate_applies_correction(self):
        val = _apply_logit(0, 4)
        p = 0.5 / 5
        expected = math.log(p / (1 - p))
        assert val == pytest.approx(expected, rel=1e-6)

    def test_full_rate_applies_correction(self):
        val = _apply_logit(4, 4)
        p = 4.5 / 5
        expected = math.log(p / (1 - p))
        assert val == pytest.approx(expected, rel=1e-6)

    def test_zero_claims_returns_nan(self):
        assert math.isnan(_apply_logit(0, 0))

    @pytest.mark.parametrize("n_flagged,n_claims", [(1, 10), (5, 10), (9, 10)])
    def test_no_correction_for_interior_rates(self, n_flagged, n_claims):
        p = n_flagged / n_claims
        expected = math.log(p / (1 - p))
        assert _apply_logit(n_flagged, n_claims) == pytest.approx(expected, rel=1e-6)

    def test_zero_rate_is_finite(self):
        assert math.isfinite(_apply_logit(0, 100))

    def test_full_rate_is_finite(self):
        assert math.isfinite(_apply_logit(100, 100))



class TestExtractConvFlagRates:
    def test_single_judge_two_convs(self):
        data = _make_flagging_results(["judge_a"], ["conv_001.json", "conv_002.json"])
        records = _extract_conv_flag_rates(data)
        assert len(records) == 2
        conv_ids = {r["conv_id"] for r in records}
        assert conv_ids == {"conv_001.json", "conv_002.json"}

    def test_two_judges_one_conv(self):
        data = _make_flagging_results(["judge_a", "judge_b"], ["conv_001.json"])
        records = _extract_conv_flag_rates(data)
        assert len(records) == 2
        judges = {r["judge"] for r in records}
        assert judges == {"judge_a", "judge_b"}

    def test_counts_correct(self):
        flagged = {"Falsehood": True, "Omission": False, "Equivocation": False, "Paltering": False}
        unflagged = {"Falsehood": False, "Omission": False, "Equivocation": False, "Paltering": False}
        data = {
            "flagging_models": ["judge_a"],
            "claim_evaluations": [
                {"flagging_model": "judge_a", "conversation_file": "conv_001.json", "deception_indicators": flagged},
                {"flagging_model": "judge_a", "conversation_file": "conv_001.json", "deception_indicators": unflagged},
                {"flagging_model": "judge_a", "conversation_file": "conv_001.json", "deception_indicators": flagged},
            ],
        }
        records = _extract_conv_flag_rates(data)
        assert len(records) == 1
        r = records[0]
        assert r["n_claims"] == 3
        assert r["n_flagged"] == 2
        assert r["flag_rate"] == pytest.approx(2 / 3)

    def test_empty_evaluations(self):
        data = {"flagging_models": [], "claim_evaluations": []}
        assert _extract_conv_flag_rates(data) == []

    def test_all_unflagged(self):
        data = _make_flagging_results(["j"], ["c1.json", "c2.json"], flag=False)
        records = _extract_conv_flag_rates(data)
        for r in records:
            assert r["n_flagged"] == 0
            assert r["flag_rate"] == 0.0



class TestScanFlaggingFiles:
    def test_finds_all_files(self, tmp_path):
        for model in ["m1", "m2"]:
            for sub in ["Sub_A", "Sub_B"]:
                path = str(tmp_path / model / "baseline" / sub / "evaluations" / "flagging_results.json")
                _write_json(path, {})
        results = _scan_flagging_files(str(tmp_path), "baseline")
        assert len(results) == 4

    def test_extracts_model_and_subtype(self, tmp_path):
        path = str(tmp_path / "gpt-4" / "baseline" / "Shoes (50%)" / "evaluations" / "flagging_results.json")
        _write_json(path, {})
        results = _scan_flagging_files(str(tmp_path), "baseline")
        assert len(results) == 1
        assert results[0]["agent_model"] == "gpt-4"
        assert results[0]["sub_scenario"] == "Shoes (50%)"

    def test_missing_condition_returns_empty(self, tmp_path):
        path = str(tmp_path / "m1" / "treatment" / "Sub_A" / "evaluations" / "flagging_results.json")
        _write_json(path, {})
        results = _scan_flagging_files(str(tmp_path), "baseline")
        assert results == []

    def test_nonexistent_results_dir_raises(self):
        with pytest.raises(ValueError, match="does not exist"):
            _scan_flagging_files("/nonexistent/path/xyz", "baseline")

    def test_no_flagging_files_returns_empty(self, tmp_path):
        os.makedirs(str(tmp_path / "m1" / "baseline" / "Sub_A" / "evaluations"))
        results = _scan_flagging_files(str(tmp_path), "baseline")
        assert results == []



class TestBuildInputMatrix:
    @pytest.fixture
    def minimal_pilot(self, tmp_path):
        return _make_pilot_dir(
            tmp_path,
            models=["model_a", "model_b"],
            judges=["judge_x", "judge_y"],
            subtypes=["Sub_1", "Sub_2"],
            conv_ids=["conv_001.json", "conv_002.json"],
        )

    def test_returns_dataframe(self, minimal_pilot):
        df = build_input_matrix(minimal_pilot, "test_scenario")
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, minimal_pilot):
        df = build_input_matrix(minimal_pilot, "test_scenario")
        for col in ["agent_model", "judge", "conv_id", "sub_scenario",
                    "n_claims", "n_flagged", "flag_rate", "logit_dr"]:
            assert col in df.columns

    def test_expected_row_count(self, minimal_pilot):
        df = build_input_matrix(minimal_pilot, "test_scenario")
        assert len(df) == 16

    def test_no_nan_in_logit_dr(self, minimal_pilot):
        df = build_input_matrix(minimal_pilot, "test_scenario")
        assert df["logit_dr"].notna().all()

    def test_facet_counts(self, minimal_pilot):
        df = build_input_matrix(minimal_pilot, "test_scenario")
        assert df["agent_model"].nunique() == 2
        assert df["judge"].nunique() == 2
        assert df["sub_scenario"].nunique() == 2

    def test_raises_on_no_files(self, tmp_path):
        with pytest.raises(ValueError, match="No flagging_results.json"):
            build_input_matrix(str(tmp_path), "test_scenario")

    def test_logit_dr_is_finite(self, minimal_pilot):
        df = build_input_matrix(minimal_pilot, "test_scenario")
        assert df["logit_dr"].apply(math.isfinite).all()

    def test_all_zero_flag_rate_applies_correction(self, tmp_path):
        _make_pilot_dir(
            tmp_path,
            models=["m1", "m2"],
            judges=["j1", "j2"],
            subtypes=["s1", "s2"],
            conv_ids=["c1.json", "c2.json"],
            flag_pattern=lambda m, s, j, c: False,
        )
        df = build_input_matrix(str(tmp_path), "scenario")
        assert (df["logit_dr"] < 0).all()

    def test_all_flagged_rate_applies_correction(self, tmp_path):
        _make_pilot_dir(
            tmp_path,
            models=["m1", "m2"],
            judges=["j1", "j2"],
            subtypes=["s1", "s2"],
            conv_ids=["c1.json", "c2.json"],
            flag_pattern=lambda m, s, j, c: True,
        )
        df = build_input_matrix(str(tmp_path), "scenario")
        assert (df["logit_dr"] > 0).all()



class TestCheckBalance:
    def test_balanced_design(self):
        df = _make_balanced_df(n_M=2, n_R=2, n_S=2, n_I=3)
        is_balanced, n_M, n_R, n_S, n_I_mean = _check_balance(df)
        assert is_balanced
        assert n_M == 2
        assert n_R == 2
        assert n_S == 2
        assert n_I_mean == pytest.approx(3.0)

    def test_unbalanced_flagged(self, tmp_path):
        rows = []
        for r in range(2):
            for s in range(2):
                n_i = 3 if s == 0 else 2
                for i in range(n_i):
                    for m in range(2):
                        rows.append({
                            "agent_model": f"m{m}", "judge": f"j{r}",
                            "sub_scenario": f"sub_{s}", "conv_id": f"sub_{s}_conv_{i}",
                            "n_claims": 1, "n_flagged": 0, "flag_rate": 0.0, "logit_dr": 0.0,
                        })
        df = pd.DataFrame(rows)
        is_balanced, *_ = _check_balance(df)
        assert not is_balanced



class TestRemlGroupToKey:
    @pytest.mark.parametrize("group,expected", [
        ("agent_model", "M"),
        ("judge", "R"),
        ("sub_scenario", "S"),
        ("sub_scenario:conv_id", "I_S"),
        ("conv_id:sub_scenario", "I_S"),
        ("agent_model:judge", "MR"),
        ("judge:agent_model", "MR"),
        ("agent_model:sub_scenario", "MS"),
        ("judge:sub_scenario", "RS"),
        ("agent_model:sub_scenario:conv_id", "MI_S"),
        ("conv_id:sub_scenario:agent_model", "MI_S"),
        ("judge:sub_scenario:conv_id", "RI_S"),
        ("agent_model:judge:sub_scenario", "MRS"),
        ("sub_scenario:judge:agent_model", "MRS"),
    ])
    def test_known_groups(self, group, expected):
        assert _reml_group_to_key(group) == expected

    def test_dot_notation_normalised(self):
        assert _reml_group_to_key("agent.model:judge") == "MR"

    def test_unknown_group_returns_none(self):
        assert _reml_group_to_key("something_else") is None

    def test_residual_not_handled(self):
        assert _reml_group_to_key("Residual") is None



class TestFitGstudy:
    def test_returns_expected_keys(self):
        df = _make_balanced_df()
        result = fit_gstudy(df)
        assert "variance_components" in result
        assert "design_counts" in result
        assert "is_balanced" in result

    def test_eleven_variance_components(self):
        df = _make_balanced_df()
        vc = fit_gstudy(df)["variance_components"]
        assert set(vc.keys()) == {"M", "R", "S", "I_S", "MR", "MS", "RS", "MI_S", "RI_S", "MRS", "e"}

    def test_all_components_nonnegative(self):
        df = _make_balanced_df(n_M=3, n_R=3, n_S=3, n_I=3,
                               value_fn=lambda m, r, i, s: float(np.random.default_rng(42 + m + r + i + s).normal()))
        vc = fit_gstudy(df)["variance_components"]
        for name, val in vc.items():
            assert val >= 0.0, f"σ²({name}) = {val} is negative"

    def test_constant_data_gives_zero_model_variance(self):
        df = _make_balanced_df(value_fn=lambda m, r, i, s: 0.5)
        vc = fit_gstudy(df)["variance_components"]
        assert vc["M"] == pytest.approx(0.0, abs=1e-10)

    def test_pure_model_variation_gives_high_g_coefficient(self):
        df = _make_balanced_df(
            n_M=2, n_R=2, n_S=2, n_I=2,
            value_fn=lambda m, r, i, s: -1.0 if m == 0 else 1.0,
        )
        result = fit_gstudy(df)
        vc = result["variance_components"]
        dc = result["design_counts"]
        g = compute_g_coefficient(vc, dc["n_R"], dc["n_I"], dc["n_S"])
        assert g == pytest.approx(1.0, abs=0.01)

    def test_pure_judge_variation_gives_zero_g_coefficient(self):
        df = _make_balanced_df(
            n_M=2, n_R=2, n_S=2, n_I=2,
            value_fn=lambda m, r, i, s: -1.0 if r == 0 else 1.0,
        )
        vc = fit_gstudy(df)["variance_components"]
        assert vc["M"] == pytest.approx(0.0, abs=1e-6)

    def test_drops_facet_on_too_few_models(self):
        df = _make_balanced_df(n_M=1, n_R=2, n_S=2, n_I=2)
        result = fit_gstudy(df)
        assert result["variance_components"]["M"] == 0.0
        assert result["variance_components"]["MR"] == 0.0

    def test_drops_facet_on_too_few_judges(self):
        df = _make_balanced_df(n_M=2, n_R=1, n_S=2, n_I=2)
        result = fit_gstudy(df)
        assert result["variance_components"]["R"] == 0.0
        assert result["variance_components"]["MR"] == 0.0

    def test_raises_on_too_few_convs(self):
        df = _make_balanced_df(n_M=2, n_R=2, n_S=2, n_I=1)
        with pytest.raises(ValueError, match="too sparse"):
            fit_gstudy(df)

    def test_design_counts_returned(self):
        df = _make_balanced_df(n_M=3, n_R=4, n_S=2, n_I=5)
        dc = fit_gstudy(df)["design_counts"]
        assert dc["n_M"] == 3
        assert dc["n_R"] == 4
        assert dc["n_S"] == 2
        assert dc["n_I"] == 5

    def test_is_balanced_true_for_uniform_design(self):
        df = _make_balanced_df()
        assert fit_gstudy(df)["is_balanced"]



class TestComputeGCoefficient:
    def _zero_vc(self) -> dict:
        return {k: 0.0 for k in ["M", "R", "S", "I_S", "MR", "MS", "RS", "MI_S", "RI_S", "MRS", "e"]}

    def test_zero_universe_variance_returns_zero(self):
        vc = self._zero_vc()
        assert compute_g_coefficient(vc, 2, 2, 2) == 0.0

    def test_pure_model_variance_returns_one(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        assert compute_g_coefficient(vc, 2, 2, 2) == pytest.approx(1.0)

    def test_result_in_zero_one(self):
        vc = self._zero_vc()
        vc["M"] = 0.5
        vc["MR"] = 0.2
        vc["e"] = 0.3
        g = compute_g_coefficient(vc, 4, 5, 3)
        assert 0.0 <= g <= 1.0

    def test_more_judges_increases_coefficient(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["MR"] = 0.5
        g_low = compute_g_coefficient(vc, 1, 4, 4)
        g_high = compute_g_coefficient(vc, 8, 4, 4)
        assert g_high > g_low

    def test_more_convs_increases_coefficient(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["MI_S"] = 0.5
        g_low = compute_g_coefficient(vc, 4, 1, 4)
        g_high = compute_g_coefficient(vc, 4, 20, 4)
        assert g_high > g_low

    def test_more_subtypes_increases_coefficient(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["MS"] = 0.5
        g_low = compute_g_coefficient(vc, 4, 4, 1)
        g_high = compute_g_coefficient(vc, 4, 4, 10)
        assert g_high > g_low

    def test_known_value(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["MR"] = 1.0
        g = compute_g_coefficient(vc, n_judges=2, n_convs=1, n_subtypes=1)
        assert g == pytest.approx(2 / 3, rel=1e-6)



class TestComputePhiCoefficient:
    def _zero_vc(self) -> dict:
        return {k: 0.0 for k in ["M", "R", "S", "I_S", "MR", "MS", "RS", "MI_S", "RI_S", "MRS", "e"]}

    def test_zero_universe_variance_returns_zero(self):
        vc = self._zero_vc()
        assert compute_phi_coefficient(vc, 2, 2, 2) == 0.0

    def test_pure_model_variance_returns_one(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        assert compute_phi_coefficient(vc, 2, 2, 2) == pytest.approx(1.0)

    def test_result_in_zero_one(self):
        vc = self._zero_vc()
        vc["M"] = 0.5
        vc["R"] = 0.2
        vc["MR"] = 0.1
        vc["e"] = 0.3
        g = compute_phi_coefficient(vc, 4, 5, 3)
        assert 0.0 <= g <= 1.0

    def test_phi_leq_g_coefficient(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["R"] = 0.4
        vc["MR"] = 0.3
        vc["e"] = 0.2
        g = compute_g_coefficient(vc, 4, 4, 4)
        phi = compute_phi_coefficient(vc, 4, 4, 4)
        assert phi <= g + 1e-9

    def test_phi_equals_g_when_no_facet_main_effects(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["MR"] = 0.5
        vc["e"] = 0.3
        g = compute_g_coefficient(vc, 3, 4, 2)
        phi = compute_phi_coefficient(vc, 3, 4, 2)
        assert phi == pytest.approx(g, rel=1e-9)

    def test_known_value(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["R"] = 1.0
        phi = compute_phi_coefficient(vc, n_judges=2, n_convs=1, n_subtypes=1)
        assert phi == pytest.approx(2 / 3, rel=1e-6)

    def test_more_judges_increases_phi(self):
        vc = self._zero_vc()
        vc["M"] = 1.0
        vc["R"] = 0.5
        phi_low = compute_phi_coefficient(vc, 1, 4, 4)
        phi_high = compute_phi_coefficient(vc, 8, 4, 4)
        assert phi_high > phi_low



class TestRunDstudy:
    @pytest.fixture
    def simple_vc(self):
        vc = {k: 0.0 for k in ["M", "R", "S", "I_S", "MR", "MS", "RS", "MI_S", "RI_S", "MRS", "e"]}
        vc["M"] = 1.0
        vc["MR"] = 0.5
        vc["MI_S"] = 0.3
        vc["MS"] = 0.2
        vc["e"] = 0.4
        return vc

    @pytest.fixture
    def observed_n(self):
        return {"n_judges": 4, "n_convs": 5, "n_subtypes": 4, "n_models": 3}

    def test_returns_dataframe(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n)
        assert set(df.columns) == {"varied_facet", "n", "g_coefficient", "phi_coefficient"}

    def test_four_facets_present(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n)
        assert set(df["varied_facet"].unique()) == {"n_judges", "n_convs", "n_subtypes", "n_models"}

    def test_n_models_g_coefficient_is_invariant(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n, model_range=[1, 2, 3, 4, 5])
        sub = df[df["varied_facet"] == "n_models"]
        assert len(sub) == 5
        assert sub["g_coefficient"].nunique() == 1

    def test_g_coefficient_monotonically_increases_with_judges(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n, judge_range=list(range(1, 10)))
        sub = df[df["varied_facet"] == "n_judges"].sort_values("n")
        diffs = sub["g_coefficient"].diff().dropna()
        assert (diffs >= 0).all()

    def test_g_coefficient_in_range(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n)
        assert (df["g_coefficient"] >= 0).all()
        assert (df["g_coefficient"] <= 1).all()

    def test_phi_coefficient_in_range(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n)
        assert (df["phi_coefficient"] >= 0).all()
        assert (df["phi_coefficient"] <= 1).all()

    def test_phi_leq_g_everywhere(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n)
        assert (df["phi_coefficient"] <= df["g_coefficient"] + 1e-9).all()

    def test_custom_judge_range(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n, judge_range=[1, 2, 3])
        judge_sub = df[df["varied_facet"] == "n_judges"]
        assert list(judge_sub["n"]) == [1, 2, 3]

    def test_default_ranges_include_observed(self, simple_vc, observed_n):
        df = run_dstudy(simple_vc, observed_n)
        judge_ns = set(df[df["varied_facet"] == "n_judges"]["n"])
        assert observed_n["n_judges"] in judge_ns



class TestSaveGstudyResults:
    @pytest.fixture
    def sample_gstudy_result(self):
        vc = {k: 0.1 for k in ["M", "R", "S", "I_S", "MR", "MS", "RS", "MI_S", "RI_S", "MRS", "e"]}
        return {
            "variance_components": vc,
            "design_counts": {"n_M": 2, "n_R": 4, "n_S": 4, "n_I": 5},
            "is_balanced": True,
        }

    @pytest.fixture
    def sample_dstudy_df(self):
        return pd.DataFrame([
            {"varied_facet": "n_judges", "n": 1, "g_coefficient": 0.4, "phi_coefficient": 0.3},
            {"varied_facet": "n_judges", "n": 2, "g_coefficient": 0.6, "phi_coefficient": 0.5},
        ])

    def test_creates_json(self, tmp_path, sample_gstudy_result, sample_dstudy_df):
        save_gstudy_results("pp", sample_gstudy_result, 0.52, 0.45, sample_dstudy_df, str(tmp_path))
        assert os.path.exists(str(tmp_path / "g_study_pp.json"))

    def test_creates_csv(self, tmp_path, sample_gstudy_result, sample_dstudy_df):
        save_gstudy_results("pp", sample_gstudy_result, 0.52, 0.45, sample_dstudy_df, str(tmp_path))
        assert os.path.exists(str(tmp_path / "d_study_pp.csv"))

    def test_json_has_expected_keys(self, tmp_path, sample_gstudy_result, sample_dstudy_df):
        save_gstudy_results("lq", sample_gstudy_result, 0.77, 0.70, sample_dstudy_df, str(tmp_path))
        with open(str(tmp_path / "g_study_lq.json")) as f:
            data = json.load(f)
        assert data["scenario"] == "lq"
        assert data["g_coefficient_observed"] == pytest.approx(0.77)
        assert data["phi_coefficient_observed"] == pytest.approx(0.70)
        assert "variance_components" in data
        assert "design_counts" in data

    def test_csv_has_expected_columns(self, tmp_path, sample_gstudy_result, sample_dstudy_df):
        save_gstudy_results("pp", sample_gstudy_result, 0.52, 0.45, sample_dstudy_df, str(tmp_path))
        df = pd.read_csv(str(tmp_path / "d_study_pp.csv"))
        assert set(df.columns) == {"varied_facet", "n", "g_coefficient", "phi_coefficient"}

    def test_creates_output_dir_if_missing(self, tmp_path, sample_gstudy_result, sample_dstudy_df):
        output_dir = str(tmp_path / "deep" / "nested" / "dir")
        save_gstudy_results("pp", sample_gstudy_result, 0.5, 0.4, sample_dstudy_df, output_dir)
        assert os.path.isdir(output_dir)

    def test_json_variance_components_match(self, tmp_path, sample_gstudy_result, sample_dstudy_df):
        save_gstudy_results("pp", sample_gstudy_result, 0.5, 0.4, sample_dstudy_df, str(tmp_path))
        with open(str(tmp_path / "g_study_pp.json")) as f:
            data = json.load(f)
        assert data["variance_components"]["M"] == pytest.approx(0.1)



class TestVerboseBranches:
    @pytest.fixture
    def minimal_pilot(self, tmp_path):
        return _make_pilot_dir(
            tmp_path,
            models=["model_a", "model_b"],
            judges=["judge_x", "judge_y"],
            subtypes=["Sub_1", "Sub_2"],
            conv_ids=["conv_001.json", "conv_002.json"],
        )

    def test_build_input_matrix_verbose_prints(self, minimal_pilot, capsys):
        build_input_matrix(minimal_pilot, "scenario", verbose=True)
        out = capsys.readouterr().out
        assert "Input matrix" in out
        assert "models" in out

    def test_fit_gstudy_verbose_prints(self, capsys):
        df = _make_balanced_df(n_M=2, n_R=2, n_S=2, n_I=2,
                               value_fn=lambda m, r, i, s: float(m) * 0.5)
        fit_gstudy(df, verbose=True)
        out = capsys.readouterr().out
        assert "Variance components" in out
        assert "Total" in out

    def test_fit_gstudy_unbalanced_warns(self, capsys):
        rows = []
        for m in range(2):
            for r in range(2):
                for s in range(2):
                    n_i = 3 if s == 0 else 2
                    for i in range(n_i):
                        rows.append({
                            "agent_model": f"m{m}", "judge": f"j{r}",
                            "sub_scenario": f"sub_{s}",
                            "conv_id": f"sub_{s}_conv_{i}",
                            "n_claims": 1, "n_flagged": 0, "flag_rate": 0.0,
                            "logit_dr": float(m),
                        })
        df = pd.DataFrame(rows)
        fit_gstudy(df)
        out = capsys.readouterr().out
        assert "unbalanced" in out

    def test_scan_skips_non_directory_entries(self, tmp_path):
        path = str(tmp_path / "gpt-4" / "baseline" / "Sub_A" / "evaluations" / "flagging_results.json")
        _write_json(path, {})
        stray_file = str(tmp_path / "README.txt")
        with open(stray_file, "w") as f:
            f.write("not a dir")
        results = _scan_flagging_files(str(tmp_path), "baseline")
        agent_models = [r["agent_model"] for r in results]
        assert "README.txt" not in agent_models
        assert "gpt-4" in agent_models



class TestRunGstudy:
    @pytest.fixture
    def minimal_pilot(self, tmp_path):
        return _make_pilot_dir(
            tmp_path,
            models=["model_a", "model_b"],
            judges=["judge_x", "judge_y"],
            subtypes=["Sub_1", "Sub_2"],
            conv_ids=["conv_001.json", "conv_002.json"],
        )

    def test_returns_dict_with_expected_keys(self, minimal_pilot, tmp_path):
        out_dir = str(tmp_path / "gstudy_out")
        result = run_gstudy(minimal_pilot, "pp", out_dir)
        assert set(result.keys()) == {"matrix", "gstudy_result", "g_coefficient", "phi_coefficient", "dstudy_df"}

    def test_g_coefficient_in_range(self, minimal_pilot, tmp_path):
        out_dir = str(tmp_path / "gstudy_out")
        result = run_gstudy(minimal_pilot, "pp", out_dir)
        assert 0.0 <= result["g_coefficient"] <= 1.0

    def test_saves_json_and_csv(self, minimal_pilot, tmp_path):
        out_dir = str(tmp_path / "gstudy_out")
        run_gstudy(minimal_pilot, "pp", out_dir)
        assert os.path.exists(os.path.join(out_dir, "g_study_pp.json"))
        assert os.path.exists(os.path.join(out_dir, "d_study_pp.csv"))

    def test_saves_plot(self, minimal_pilot, tmp_path):
        out_dir = str(tmp_path / "gstudy_out")
        run_gstudy(minimal_pilot, "pp", out_dir)
        assert os.path.exists(os.path.join(out_dir, "plots", "dstudy_pp.png"))

    def test_verbose_prints_header(self, minimal_pilot, tmp_path, capsys):
        out_dir = str(tmp_path / "gstudy_out")
        run_gstudy(minimal_pilot, "pp", out_dir, verbose=True)
        out = capsys.readouterr().out
        assert "G-Study" in out
        assert "G-coefficient" in out

    def test_prints_dstudy_best_per_facet(self, minimal_pilot, tmp_path, capsys):
        out_dir = str(tmp_path / "gstudy_out")
        run_gstudy(minimal_pilot, "pp", out_dir)
        out = capsys.readouterr().out
        assert "D-study: best G and Φ per facet" in out
        assert "n_judges" in out
        assert "n_convs" in out
        assert "n_subtypes" in out
        assert "n_models" in out
