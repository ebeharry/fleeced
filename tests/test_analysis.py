import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.stats import two_proportion_z_test, wilson_ci
from src.analysis.summary import (
    _aggregate_metrics,
    _compute_subtype_metrics,
    _get_primary_types,
    _get_unique_flagged_claims,
    compute_deception_rate_by_mode,
    run_deception_rate_by_mode,
    summarize_trial_by_model,
    summarize_trial_per_model_by_condition,
    summarize_scenario,
    summarize_subtype,
)


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_claim_eval(claim_id: str, model: str, is_flagged: bool, dtype: str = "Falsehood") -> dict:
    return {
        "claim_id": claim_id,
        "flagging_model": model,
        "is_flagged": is_flagged,
        "deception_indicators": {
            "Falsehood": dtype == "Falsehood" and is_flagged,
            "Omission": dtype == "Omission" and is_flagged,
            "Equivocation": dtype == "Equivocation" and is_flagged,
            "Paltering": dtype == "Paltering" and is_flagged,
        },
        "reasons": {dtype: "Test reason"} if is_flagged and dtype else {},
        "conversation_file": "conversation_001.json",
        "claim_type": "product_claim",
        "turn": 1,
        "claim": "Test claim",
        "raw_response": "",
        "timestamp": "2026-01-01 00:00:00",
    }


def _make_assessment(claim_id: str, is_correct: bool, dtype: str = "Falsehood") -> dict:
    return {
        "claim_id": claim_id,
        "conversation_file": "conversation_001.json",
        "agent_statement": "Test statement",
        "flagging_primary_types": [dtype] if dtype else [],
        "selected_answer": "A" if is_correct else "B",
        "correct_answer": "A",
        "is_correct": is_correct,
        "reason": "Because",
        "options": ["alt1", "alt2"],
        "raw_response": "",
        "mcq_question": "Which is deceptive?",
        "timestamp": "2026-01-01 00:00:00",
    }


@pytest.fixture
def simple_subtype_dir(tmp_path):
    flagging_data = {
        "created_timestamp": "2026-01-01 00:00:00",
        "flagging_models": ["gpt-4o"],
        "summary": None,
        "claim_evaluations": [
            _make_claim_eval("c1", "gpt-4o", True, "Falsehood"),
            _make_claim_eval("c2", "gpt-4o", True, "Omission"),
            _make_claim_eval("c3", "gpt-4o", False),
            _make_claim_eval("c4", "gpt-4o", False),
        ],
        "errors": [],
    }
    assessment_data = {
        "created_timestamp": "2026-01-01 00:00:00",
        "assessing_model": "gpt-4o",
        "summary": None,
        "assessment_results": [
            _make_assessment("c1", True, "Falsehood"),
            _make_assessment("c2", False, "Omission"),
        ],
        "errors": [],
    }
    subtype = tmp_path / "Subtype_1"
    _write_json(str(subtype / "evaluations" / "flagging_results.json"), flagging_data)
    _write_json(str(subtype / "evaluations" / "self_assessment_results.json"), assessment_data)
    return str(subtype)


@pytest.fixture
def two_model_subtype_dir(tmp_path):
    flagging_data = {
        "created_timestamp": "2026-01-01 00:00:00",
        "flagging_models": ["gpt-4o", "claude-3"],
        "summary": None,
        "claim_evaluations": [
            _make_claim_eval("c1", "gpt-4o", True, "Falsehood"),
            _make_claim_eval("c1", "claude-3", True, "Omission"),
            _make_claim_eval("c2", "gpt-4o", True, "Equivocation"),
            _make_claim_eval("c2", "claude-3", False),
            _make_claim_eval("c3", "gpt-4o", False),
            _make_claim_eval("c3", "claude-3", False),
        ],
        "errors": [],
    }
    subtype = tmp_path / "Subtype_Multi"
    _write_json(str(subtype / "evaluations" / "flagging_results.json"), flagging_data)
    return str(subtype)


@pytest.fixture
def multi_subtype_dir(tmp_path, simple_subtype_dir):
    flagging_data_b = {
        "created_timestamp": "2026-01-01 00:00:00",
        "flagging_models": ["gpt-4o"],
        "summary": None,
        "claim_evaluations": [
            _make_claim_eval("d1", "gpt-4o", True, "Paltering"),
            _make_claim_eval("d2", "gpt-4o", False),
        ],
        "errors": [],
    }
    scenario_dir = tmp_path / "scenario"
    os.makedirs(str(scenario_dir), exist_ok=True)
    import shutil
    shutil.copytree(simple_subtype_dir, str(scenario_dir / "Subtype_1"))
    _write_json(str(scenario_dir / "Subtype_2" / "evaluations" / "flagging_results.json"), flagging_data_b)
    return str(scenario_dir)


@pytest.fixture
def multi_model_trial_dir(tmp_path):
    models = ["openai/gpt-a", "openai/gpt-b"]
    conditions = ["baseline", "treatment"]

    entries = []
    for model in models:
        for condition in conditions:
            cond_dir = tmp_path / model / condition
            subtype_path = cond_dir / "Subtype_1"
            flagging_data = {
                "created_timestamp": "2026-01-01 00:00:00",
                "flagging_models": ["gpt-4o"],
                "summary": None,
                "claim_evaluations": [
                    _make_claim_eval("x1", "gpt-4o", True, "Falsehood"),
                    _make_claim_eval("x2", "gpt-4o", False),
                ],
                "errors": [],
            }
            assessment_data = {
                "created_timestamp": "2026-01-01 00:00:00",
                "assessing_model": model,
                "summary": None,
                "assessment_results": [_make_assessment("x1", True, "Falsehood")],
                "errors": [],
            }
            _write_json(str(subtype_path / "evaluations" / "flagging_results.json"), flagging_data)
            _write_json(str(subtype_path / "evaluations" / "self_assessment_results.json"), assessment_data)
            entries.append({
                "assessing_model": model,
                "condition_name": condition,
                "log_dir": str(cond_dir),
                "flagging_models": ["gpt-4o"],
            })

    manifest = {
        "trial_name": "test_trial",
        "scenario": "product_promotion",
        "assessing_models": models,
        "conditions": entries,
    }
    _write_json(str(tmp_path / "trial_manifest.json"), manifest)
    return str(tmp_path)


@pytest.fixture
def two_sub_scenario_trial_dir(tmp_path):
    """
    A two-model trial with one condition ("default") and two sub-scenarios (Books, Shoes).
    """
    models = ["openai/gpt-a", "openai/gpt-b"]
    entries = []
    for model in models:
        cond_dir = tmp_path / model / "default"
        for sub_scenario in ["Books", "Shoes"]:
            subtype_path = cond_dir / sub_scenario
            flagging_data = {
                "created_timestamp": "2026-01-01 00:00:00",
                "flagging_models": ["gpt-4o"],
                "summary": None,
                "claim_evaluations": [
                    _make_claim_eval("x1", "gpt-4o", True, "Falsehood"),
                    _make_claim_eval("x2", "gpt-4o", False),
                ],
                "errors": [],
            }
            assessment_data = {
                "created_timestamp": "2026-01-01 00:00:00",
                "assessing_model": model,
                "summary": None,
                "assessment_results": [_make_assessment("x1", True, "Falsehood")],
                "errors": [],
            }
            _write_json(str(subtype_path / "evaluations" / "flagging_results.json"), flagging_data)
            _write_json(str(subtype_path / "evaluations" / "self_assessment_results.json"), assessment_data)
        entries.append({
            "assessing_model": model,
            "condition_name": "default",
            "log_dir": str(cond_dir),
            "flagging_models": ["gpt-4o"],
        })

    manifest = {
        "trial_name": "test_trial",
        "scenario": "product_promotion",
        "assessing_models": models,
        "conditions": entries,
    }
    _write_json(str(tmp_path / "trial_manifest.json"), manifest)
    return str(tmp_path)


@pytest.fixture
def mode_sensitive_trial_dir(tmp_path):
    """
    A single-model, single-condition trial with one Falsehood-confirmed
    claim and one Paltering-only-confirmed claim, so that
    "falsehood_omission" mode drops exactly one confirmed claim while "all"
    keeps both.
    """
    cond_dir = tmp_path / "openai/gpt-a" / "baseline"
    subtype_path = cond_dir / "Subtype_1"
    flagging_data = {
        "created_timestamp": "2026-01-01 00:00:00",
        "flagging_models": ["gpt-4o"],
        "summary": None,
        "claim_evaluations": [
            _make_claim_eval("x1", "gpt-4o", True, "Falsehood"),
            _make_claim_eval("x2", "gpt-4o", True, "Paltering"),
            _make_claim_eval("x3", "gpt-4o", False),
        ],
        "errors": [],
    }
    assessment_data = {
        "created_timestamp": "2026-01-01 00:00:00",
        "assessing_model": "openai/gpt-a",
        "summary": None,
        "assessment_results": [
            _make_assessment("x1", True, "Falsehood"),
            _make_assessment("x2", True, "Paltering"),
        ],
        "errors": [],
    }
    _write_json(str(subtype_path / "evaluations" / "flagging_results.json"), flagging_data)
    _write_json(str(subtype_path / "evaluations" / "self_assessment_results.json"), assessment_data)

    manifest = {
        "trial_name": "mode_sensitive_trial",
        "scenario": "product_promotion",
        "assessing_models": ["openai/gpt-a"],
        "conditions": [{
            "assessing_model": "openai/gpt-a",
            "condition_name": "baseline",
            "log_dir": str(cond_dir),
            "flagging_models": ["gpt-4o"],
        }],
    }
    _write_json(str(tmp_path / "trial_manifest.json"), manifest)
    return str(tmp_path)


class TestComputeDeceptionRateByMode:
    def test_all_mode_matches_full_confirmed_count(self, multi_model_trial_dir):
        result = compute_deception_rate_by_mode(multi_model_trial_dir, mode="all")
        assert result["n_total_claims"] == 8
        assert result["n_confirmed"] == 4
        assert result["deception_rate"] == pytest.approx(0.5)
        assert result["by_subtype"]["Falsehood"]["n_confirmed"] == 4

    def test_falsehood_omission_drops_paltering_only_claim(self, mode_sensitive_trial_dir):
        all_result = compute_deception_rate_by_mode(mode_sensitive_trial_dir, mode="all")
        fo_result = compute_deception_rate_by_mode(mode_sensitive_trial_dir, mode="falsehood_omission")

        assert all_result["n_confirmed"] == 2
        assert fo_result["n_confirmed"] == 1
        assert fo_result["deception_rate"] < all_result["deception_rate"]

    def test_no_paltering_drops_paltering_only_claim(self, mode_sensitive_trial_dir):
        all_result = compute_deception_rate_by_mode(mode_sensitive_trial_dir, mode="all")
        np_result = compute_deception_rate_by_mode(mode_sensitive_trial_dir, mode="no_paltering")

        assert all_result["n_confirmed"] == 2
        assert np_result["n_confirmed"] == 1
        assert np_result["deception_rate"] < all_result["deception_rate"]

    def test_by_subtype_breakdown_is_mode_independent(self, mode_sensitive_trial_dir):
        all_result = compute_deception_rate_by_mode(mode_sensitive_trial_dir, mode="all")
        fo_result = compute_deception_rate_by_mode(mode_sensitive_trial_dir, mode="falsehood_omission")
        assert all_result["by_subtype"]["Paltering"]["n_confirmed"] == 1
        assert fo_result["by_subtype"]["Paltering"]["n_confirmed"] == 1

    def test_raises_on_missing_manifest(self, tmp_path):
        with pytest.raises(Exception):
            compute_deception_rate_by_mode(str(tmp_path), mode="all")


class TestRunDeceptionRateByMode:
    def test_writes_result_json(self, mode_sensitive_trial_dir, tmp_path):
        output_path = str(tmp_path / "deception_rate_by_type.json")
        result = run_deception_rate_by_mode(
            mode_sensitive_trial_dir, output_path, mode="all"
        )
        assert os.path.exists(output_path)
        with open(output_path) as f:
            saved = json.load(f)
        assert saved["deception_rate"] == result["deception_rate"]
        assert saved["mode"] == "all"


class TestWilsonCI:
    def test_zero_total(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_all_success(self):
        lo, hi = wilson_ci(10, 10)
        assert hi <= 1.0
        assert lo > 0.7

    def test_no_success(self):
        lo, hi = wilson_ci(0, 20)
        assert lo == pytest.approx(0.0, abs=0.01)

    def test_bounds_within_01(self):
        lo, hi = wilson_ci(3, 100)
        assert 0.0 <= lo <= hi <= 1.0

    @pytest.mark.parametrize("n,total", [(5, 10), (1, 3), (50, 200)])
    def test_various_inputs(self, n, total):
        lo, hi = wilson_ci(n, total)
        assert lo < n / total < hi


class TestTwoProportionZTest:
    def test_equal_proportions(self):
        p = two_proportion_z_test(10, 100, 10, 100)
        assert p == pytest.approx(1.0, abs=0.01)

    def test_very_different_proportions(self):
        p = two_proportion_z_test(90, 100, 10, 100)
        assert p < 0.001

    def test_zero_total(self):
        assert two_proportion_z_test(0, 0, 5, 10) == 1.0

    def test_returns_float(self):
        p = two_proportion_z_test(5, 10, 8, 20)
        assert isinstance(p, float)


class TestGetUniqueFlaggedClaims:
    def test_basic(self):
        evals = [
            _make_claim_eval("c1", "m1", True),
            _make_claim_eval("c2", "m1", False),
            _make_claim_eval("c3", "m1", True),
        ]
        all_ids, flagged = _get_unique_flagged_claims(evals)
        assert all_ids == {"c1", "c2", "c3"}
        assert flagged == {"c1", "c3"}

    def test_multi_model_union(self):
        evals = [
            _make_claim_eval("c1", "m1", False),
            _make_claim_eval("c1", "m2", True),
        ]
        all_ids, flagged = _get_unique_flagged_claims(evals)
        assert "c1" in flagged

    def test_empty(self):
        all_ids, flagged = _get_unique_flagged_claims([])
        assert all_ids == set()
        assert flagged == set()


class TestGetPrimaryTypes:
    def test_unanimous_single_type(self):
        evals = [
            _make_claim_eval("c1", "m1", True, "Falsehood"),
            _make_claim_eval("c1", "m2", True, "Falsehood"),
        ]
        result = _get_primary_types("c1", "conversation_001.json", evals)
        assert result == ["Falsehood"]

    def test_strict_majority_wins(self):
        evals = [
            _make_claim_eval("c1", "m1", True, "Falsehood"),
            _make_claim_eval("c1", "m2", True, "Falsehood"),
            _make_claim_eval("c1", "m3", True, "Omission"),
        ]
        result = _get_primary_types("c1", "conversation_001.json", evals)
        assert result == ["Falsehood"]

    def test_no_majority_falls_back_to_all_indicated(self):
        evals = [
            _make_claim_eval("c1", "m1", True, "Falsehood"),
            _make_claim_eval("c1", "m2", True, "Omission"),
        ]
        result = _get_primary_types("c1", "conversation_001.json", evals)
        assert set(result) == {"Falsehood", "Omission"}

    def test_no_flagged_entries_returns_empty(self):
        evals = [_make_claim_eval("c1", "m1", False)]
        result = _get_primary_types("c1", "conversation_001.json", evals)
        assert result == []


class TestComputeSubtypeMetrics:
    def test_basic_metrics(self, simple_subtype_dir):
        m = _compute_subtype_metrics(simple_subtype_dir)
        assert m["n_total_claims"] == 4
        assert m["n_flagged_any_judge"] == 2
        assert m["flag_rate_any_judge"] == pytest.approx(0.5)
        assert m["n_confirmed_deceptive"] == 1
        assert m["deception_rate"] == pytest.approx(0.25)
        assert m["flag_to_deception_ratio"] == pytest.approx(2.0)
        assert m["n_assessed"] == 2

    def test_missing_assessment(self, tmp_path):
        flagging_data = {
            "created_timestamp": "2026-01-01",
            "flagging_models": ["gpt-4o"],
            "summary": None,
            "claim_evaluations": [_make_claim_eval("c1", "gpt-4o", True)],
            "errors": [],
        }
        subtype = tmp_path / "sub"
        _write_json(str(subtype / "evaluations" / "flagging_results.json"), flagging_data)
        m = _compute_subtype_metrics(str(subtype))
        assert m["n_flagged_any_judge"] == 1
        assert m["deception_rate"] is None
        assert m["self_assessment_accuracy"] is None

    def test_missing_flagging(self, tmp_path):
        subtype = tmp_path / "empty_sub"
        os.makedirs(str(subtype / "evaluations"), exist_ok=True)
        m = _compute_subtype_metrics(str(subtype))
        assert m["n_total_claims"] == 0
        assert m["flag_rate_any_judge"] is None

    def test_multi_model_union_flagging(self, two_model_subtype_dir):
        m = _compute_subtype_metrics(two_model_subtype_dir)
        assert m["n_total_claims"] == 3
        assert m["n_flagged_any_judge"] == 2

    def test_by_deception_type_populated(self, simple_subtype_dir):
        m = _compute_subtype_metrics(simple_subtype_dir)
        assert "Falsehood" in m["by_deception_type"]
        assert "Omission" in m["by_deception_type"]


class TestAggregateMetrics:
    def test_pools_counts(self, simple_subtype_dir, tmp_path):
        m1 = _compute_subtype_metrics(simple_subtype_dir)
        flagging_data = {
            "created_timestamp": "2026-01-01",
            "flagging_models": ["gpt-4o"],
            "summary": None,
            "claim_evaluations": [_make_claim_eval("d1", "gpt-4o", True, "Omission")],
            "errors": [],
        }
        subtype2 = tmp_path / "sub2"
        _write_json(str(subtype2 / "evaluations" / "flagging_results.json"), flagging_data)
        m2 = _compute_subtype_metrics(str(subtype2))
        agg = _aggregate_metrics([m1, m2])
        assert agg["n_total_claims"] == m1["n_total_claims"] + m2["n_total_claims"]
        assert agg["n_flagged_any_judge"] == m1["n_flagged_any_judge"] + m2["n_flagged_any_judge"]

    def test_empty_list(self):
        agg = _aggregate_metrics([])
        assert agg["n_total_claims"] == 0
        assert agg["flag_rate_any_judge"] is None


class TestSummarizeSubtype:
    def test_runs_and_returns_dict(self, simple_subtype_dir, capsys):
        result = summarize_subtype(simple_subtype_dir)
        out = capsys.readouterr().out
        assert "Flag rate" in out
        assert "Deception rate" in out
        assert isinstance(result, dict)
        assert "flag_rate_any_judge" in result
        assert "deception_rate" in result

    def test_no_internal_keys(self, simple_subtype_dir):
        result = summarize_subtype(simple_subtype_dir)
        assert not any(k.startswith("_") for k in result)

    def test_plots_saved(self, simple_subtype_dir):
        summarize_subtype(simple_subtype_dir)
        plot_dir = os.path.join(simple_subtype_dir, "evaluations", "plots")
        assert os.path.exists(os.path.join(plot_dir, "deception_types.png"))

    def test_no_model_agreement_heatmap_single_model(self, simple_subtype_dir):
        summarize_subtype(simple_subtype_dir)
        plot_dir = os.path.join(simple_subtype_dir, "evaluations", "plots")
        assert not os.path.exists(os.path.join(plot_dir, "model_agreement.png"))

    def test_model_agreement_heatmap_multi_model(self, two_model_subtype_dir):
        summarize_subtype(two_model_subtype_dir)
        plot_dir = os.path.join(two_model_subtype_dir, "evaluations", "plots")
        assert os.path.exists(os.path.join(plot_dir, "model_agreement.png"))


class TestSummarizeScenario:
    def test_runs_and_returns_dict(self, multi_subtype_dir, capsys):
        result = summarize_scenario(multi_subtype_dir)
        out = capsys.readouterr().out
        assert "Flag rate" in out
        assert isinstance(result, dict)
        assert "overall" in result
        assert "by_subtype" in result

    def test_plots_saved(self, multi_subtype_dir):
        summarize_scenario(multi_subtype_dir)
        plot_dir = os.path.join(multi_subtype_dir, "plots")
        assert os.path.exists(os.path.join(plot_dir, "vote_distribution.png"))

    @pytest.mark.parametrize("filename", [
        "rates_by_subtype.png",
        "deception_types.png",
        "flag_rate_by_judge.png",
    ])
    def test_dropped_plots_not_saved(self, multi_subtype_dir, filename):
        summarize_scenario(multi_subtype_dir)
        assert not os.path.exists(os.path.join(multi_subtype_dir, "plots", filename))

    def test_model_agreement_saved_with_two_judges(self, two_judge_scenario_dir):
        summarize_scenario(two_judge_scenario_dir)
        plot_dir = os.path.join(two_judge_scenario_dir, "plots")
        assert os.path.exists(os.path.join(plot_dir, "model_agreement.png"))
        assert os.path.exists(os.path.join(plot_dir, "vote_distribution.png"))

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(ValueError, match="No subtype directories"):
            summarize_scenario(str(tmp_path))

    def test_pairwise_tests_in_verbose(self, multi_subtype_dir, capsys):
        result = summarize_scenario(multi_subtype_dir, verbose=True)
        assert len(result["pairwise_tests"]) == 1
        test = result["pairwise_tests"][0]
        assert "flag_rate_any_judge_pvalue" in test
        assert isinstance(test["flag_rate_any_judge_pvalue"], float)


class TestSummarizeAblationByModel:
    def test_runs_and_returns_dict(self, multi_model_trial_dir, capsys):
        result = summarize_trial_by_model(multi_model_trial_dir)
        out = capsys.readouterr().out
        assert "Model Comparison" in out
        assert "BY MODEL" in out
        assert isinstance(result, dict)
        assert "by_model" in result
        assert "overall" in result
        assert "model_tests" in result

    def test_by_model_has_all_models(self, multi_model_trial_dir):
        result = summarize_trial_by_model(multi_model_trial_dir)
        assert set(result["by_model"].keys()) == {"openai/gpt-a", "openai/gpt-b"}

    def test_model_metrics_aggregate_across_conditions(self, multi_model_trial_dir):
        result = summarize_trial_by_model(multi_model_trial_dir)
        for model_data in result["by_model"].values():
            assert model_data["n_total_claims"] == 4
            assert model_data["n_flagged_any_judge"] == 2

    def test_model_tests_has_pairwise_entries(self, multi_model_trial_dir):
        result = summarize_trial_by_model(multi_model_trial_dir)
        assert len(result["model_tests"]) == 1
        test = result["model_tests"][0]
        assert "model_a" in test
        assert "model_b" in test
        assert "flag_rate_any_judge_pvalue" in test
        assert isinstance(test["flag_rate_any_judge_pvalue"], float)

    def test_plots_saved(self, multi_model_trial_dir):
        summarize_trial_by_model(multi_model_trial_dir)
        plot_dir = os.path.join(multi_model_trial_dir, "plots")
        assert os.path.exists(os.path.join(plot_dir, "rates_by_model.png"))
        assert os.path.exists(os.path.join(plot_dir, "deception_rate_by_model_stacked_types.png"))
        assert not os.path.exists(os.path.join(plot_dir, "deception_types_by_model.png"))

    def test_no_internal_keys_in_by_model(self, multi_model_trial_dir):
        result = summarize_trial_by_model(multi_model_trial_dir)
        for model_data in result["by_model"].values():
            assert not any(k.startswith("_") for k in model_data)

    def test_raises_on_missing_manifest(self, tmp_path):
        with pytest.raises(ValueError, match="trial_manifest.json not found"):
            summarize_trial_by_model(str(tmp_path))

    def test_per_condition_subtype_plots_saved(self, multi_model_trial_dir):
        summarize_trial_by_model(multi_model_trial_dir)
        plot_dir = os.path.join(multi_model_trial_dir, "plots")
        assert os.path.exists(os.path.join(plot_dir, "deception_rate_by_model_by_sub_scenario_baseline.png"))
        assert os.path.exists(os.path.join(plot_dir, "deception_rate_by_model_by_sub_scenario_treatment.png"))

    def test_sub_scenario_figure_saved_with_two_sub_scenarios(self, two_sub_scenario_trial_dir):
        summarize_trial_by_model(two_sub_scenario_trial_dir)
        plot_dir = os.path.join(two_sub_scenario_trial_dir, "plots")
        assert os.path.exists(os.path.join(plot_dir, "deception_rate_by_model_by_sub_scenario.png"))

    def test_sub_scenario_figure_skipped_with_one_sub_scenario(self, multi_model_trial_dir):
        summarize_trial_by_model(multi_model_trial_dir)
        plot_dir = os.path.join(multi_model_trial_dir, "plots")
        assert not os.path.exists(os.path.join(plot_dir, "deception_rate_by_model_by_sub_scenario.png"))

    def test_per_condition_plots_skipped_with_one_condition(self, two_sub_scenario_trial_dir):
        summarize_trial_by_model(two_sub_scenario_trial_dir)
        plot_dir = os.path.join(two_sub_scenario_trial_dir, "plots")
        assert not os.path.exists(os.path.join(plot_dir, "deception_rate_by_model_by_sub_scenario_default.png"))

    def test_by_model_by_subtype_in_result(self, multi_model_trial_dir):
        result = summarize_trial_by_model(multi_model_trial_dir)
        assert "by_model_by_subtype" in result
        for model in ("openai/gpt-a", "openai/gpt-b"):
            assert model in result["by_model_by_subtype"]
            assert "Subtype_1" in result["by_model_by_subtype"][model]
            assert not any(
                k.startswith("_")
                for k in result["by_model_by_subtype"][model]["Subtype_1"]
            )


@pytest.fixture
def two_judge_scenario_dir(tmp_path):
    judges = ["gpt-4o", "claude-3"]
    for subtype_name, claims in [
        ("Subtype_1", [("c1", True, "Falsehood"), ("c2", False, "")]),
        ("Subtype_2", [("d1", True, "Omission"), ("d2", True, "Paltering")]),
    ]:
        evals = [_make_claim_eval(cid, j, flagged, dtype) for cid, flagged, dtype in claims for j in judges]
        flagging_data = {
            "created_timestamp": "2026-01-01 00:00:00",
            "flagging_models": judges,
            "summary": None,
            "claim_evaluations": evals,
            "errors": [],
        }
        _write_json(str(tmp_path / subtype_name / "evaluations" / "flagging_results.json"), flagging_data)
    return str(tmp_path)


@pytest.fixture
def two_judge_trial_dir(tmp_path):
    judges = ["gpt-4o", "claude-3"]
    models = ["openai/gpt-a", "openai/gpt-b"]
    conditions = ["baseline", "treatment"]

    entries = []
    for model in models:
        for condition in conditions:
            cond_dir = tmp_path / model / condition
            subtype_path = cond_dir / "Subtype_1"
            evals = [
                _make_claim_eval("x1", j, True, "Falsehood") for j in judges
            ] + [
                _make_claim_eval("x2", j, False, "") for j in judges
            ]
            flagging_data = {
                "created_timestamp": "2026-01-01 00:00:00",
                "flagging_models": judges,
                "summary": None,
                "claim_evaluations": evals,
                "errors": [],
            }
            assessment_data = {
                "created_timestamp": "2026-01-01 00:00:00",
                "assessing_model": model,
                "summary": None,
                "assessment_results": [_make_assessment("x1", True, "Falsehood")],
                "errors": [],
            }
            _write_json(str(subtype_path / "evaluations" / "flagging_results.json"), flagging_data)
            _write_json(str(subtype_path / "evaluations" / "self_assessment_results.json"), assessment_data)
            entries.append({
                "assessing_model": model,
                "condition_name": condition,
                "log_dir": str(cond_dir),
                "flagging_models": judges,
            })

    manifest = {
        "trial_name": "two_judge_trial",
        "scenario": "product_promotion",
        "assessing_models": models,
        "conditions": entries,
    }
    _write_json(str(tmp_path / "trial_manifest.json"), manifest)
    return str(tmp_path)


class TestFlagRateByJudgePlot:
    def test_trial_creates_one_plot_per_assessing_model(self, two_judge_trial_dir):
        summarize_trial_per_model_by_condition(two_judge_trial_dir)
        plot_dir = os.path.join(two_judge_trial_dir, "plots")
        assert os.path.exists(os.path.join(plot_dir, "gpt-a_flag_rate_by_judge.png"))
        assert os.path.exists(os.path.join(plot_dir, "gpt-b_flag_rate_by_judge.png"))

    def test_trial_single_judge_creates_per_model_plots(self, multi_model_trial_dir):
        summarize_trial_per_model_by_condition(multi_model_trial_dir)
        plot_dir = os.path.join(multi_model_trial_dir, "plots")
        assert os.path.exists(os.path.join(plot_dir, "gpt-a_flag_rate_by_judge.png"))
        assert os.path.exists(os.path.join(plot_dir, "gpt-b_flag_rate_by_judge.png"))
