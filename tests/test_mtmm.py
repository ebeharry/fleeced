import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from ordinalcorr import tetrachoric

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.mtmm import (
    CONV_THRESHOLD,
    SUBTYPES,
    _find_flagging_files,
    _get_heterotrait_monomethod_block,
    _htmonm_local_values,
    _hthm_row_col_neighbours,
    _merge_flagging_results,
    _parse_column,
    assess_construct_validity,
    build_claim_matrix,
    check_trait_pattern_consistency,
    classify_mtmm_correlations,
    compute_mtmm_correlation_matrix,
    run_mtmm,
    save_mtmm_results,
)



def _make_flagging_results(
    judges: list[str],
    claim_ids: list[str],
    flag_fn=None,
) -> dict:
    """
    Build a minimal flagging_results dict with controlled per-cell flag values.

    :param flag_fn: callable(judge, claim_id, subtype) -> bool. Defaults all False.
    """
    evals = []
    for judge in judges:
        for claim_id in claim_ids:
            indicators = {}
            for s in SUBTYPES:
                indicators[s] = bool(flag_fn(judge, claim_id, s)) if flag_fn else False
            evals.append({
                "flagging_model": judge,
                "conversation_file": "conv_001",
                "claim_id": claim_id,
                "claim_type": "product",
                "turn": 1,
                "claim": "Test claim",
                "deception_indicators": indicators,
                "reasons": {s: "" for s in SUBTYPES},
                "is_flagged": any(indicators.values()),
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


def _write_flagging_results(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _minimal_corr_matrix(judges: list[str] | None = None, n_claims: int = 20) -> pd.DataFrame:
    """
    Build a minimal valid corr_matrix for tests that need the structural parameter.
    """
    if judges is None:
        judges = ["j1", "j2"]
    fr = _make_flagging_results(judges, [f"c{i}" for i in range(n_claims)])
    df = build_claim_matrix(fr)
    return compute_mtmm_correlation_matrix(df)



class TestFindFlaggingFiles:
    def test_finds_files_in_nested_dirs(self, tmp_path):
        for subdir in ["a/evaluations", "b/evaluations", "c/other"]:
            os.makedirs(tmp_path / subdir, exist_ok=True)
        (tmp_path / "a/evaluations/flagging_results.json").write_text("{}")
        (tmp_path / "b/evaluations/flagging_results.json").write_text("{}")
        (tmp_path / "c/other/not_flagging.json").write_text("{}")
        found = _find_flagging_files(str(tmp_path))
        assert len(found) == 2
        assert all(os.path.basename(f) == "flagging_results.json" for f in found)

    def test_returns_sorted_paths(self, tmp_path):
        for subdir in ["z/evaluations", "a/evaluations"]:
            os.makedirs(tmp_path / subdir, exist_ok=True)
            (tmp_path / subdir / "flagging_results.json").write_text("{}")
        found = _find_flagging_files(str(tmp_path))
        assert found == sorted(found)

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert _find_flagging_files(str(tmp_path)) == []

    def test_raises_for_nonexistent_dir(self, tmp_path):
        with pytest.raises(ValueError, match="Directory not found"):
            _find_flagging_files(str(tmp_path / "nonexistent"))



class TestMergeFlaggingResults:
    def _write_fr(self, path: str, judges: list[str], claim_ids: list[str]) -> None:
        fr = _make_flagging_results(judges, claim_ids)
        _write_flagging_results(path, fr)

    def test_merges_claim_evaluations(self, tmp_path):
        p1 = str(tmp_path / "f1.json")
        p2 = str(tmp_path / "f2.json")
        self._write_fr(p1, ["j1"], ["c1", "c2"])
        self._write_fr(p2, ["j1"], ["c1", "c2"])
        merged = _merge_flagging_results([p1, p2])
        assert len(merged["claim_evaluations"]) == 4

    def test_prefixes_claim_ids_to_avoid_collision(self, tmp_path):
        p1 = str(tmp_path / "f1.json")
        p2 = str(tmp_path / "f2.json")
        self._write_fr(p1, ["j1"], ["c1"])
        self._write_fr(p2, ["j1"], ["c1"])
        merged = _merge_flagging_results([p1, p2])
        ids = {e["claim_id"] for e in merged["claim_evaluations"]}
        assert len(ids) == 2
        assert any(id_.startswith("0_") for id_ in ids)
        assert any(id_.startswith("1_") for id_ in ids)

    def test_unions_flagging_models(self, tmp_path):
        p1 = str(tmp_path / "f1.json")
        p2 = str(tmp_path / "f2.json")
        self._write_fr(p1, ["j1"], ["c1"])
        self._write_fr(p2, ["j2"], ["c1"])
        merged = _merge_flagging_results([p1, p2])
        assert set(merged["flagging_models"]) == {"j1", "j2"}

    def test_raises_on_empty_paths(self):
        with pytest.raises(ValueError, match="No flagging_results"):
            _merge_flagging_results([])



class TestParseColumn:
    def test_basic_split(self):
        judge, subtype = _parse_column("gpt-4o::Falsehood")
        assert judge == "gpt-4o"
        assert subtype == "Falsehood"

    def test_judge_with_slash(self):
        judge, subtype = _parse_column("openai/gpt-4o::Omission")
        assert judge == "openai/gpt-4o"
        assert subtype == "Omission"

    def test_all_subtypes(self):
        for s in SUBTYPES:
            _, subtype = _parse_column(f"judge::{s}")
            assert subtype == s



class TestBuildClaimMatrix:
    def test_shape_two_judges(self):
        fr = _make_flagging_results(["j1", "j2"], ["c1", "c2", "c3"])
        df = build_claim_matrix(fr)
        assert df.shape == (3, 8)

    def test_binary_values(self):
        fr = _make_flagging_results(["j1"], ["c1"])
        df = build_claim_matrix(fr)
        assert set(df.values.flatten()) <= {0, 1}

    def test_column_naming_convention(self):
        fr = _make_flagging_results(["j1"], ["c1"])
        df = build_claim_matrix(fr)
        expected = {f"j1::{s}" for s in SUBTYPES}
        assert set(df.columns) == expected

    def test_flag_fn_respected(self):
        def flag_fn(judge, claim_id, subtype):
            return judge == "j1" and subtype == "Falsehood"

        fr = _make_flagging_results(["j1", "j2"], ["c1"], flag_fn=flag_fn)
        df = build_claim_matrix(fr)
        assert df.loc["c1", "j1::Falsehood"] == 1
        assert df.loc["c1", "j1::Omission"] == 0
        assert df.loc["c1", "j2::Falsehood"] == 0

    def test_nan_for_missing_judge(self):
        fr = _make_flagging_results(["j1", "j2"], ["c1", "c2"])
        fr["claim_evaluations"] = [
            e for e in fr["claim_evaluations"]
            if not (e["flagging_model"] == "j2" and e["claim_id"] == "c1")
        ]
        df = build_claim_matrix(fr)
        assert all(pd.isna(df.loc["c1", f"j2::{s}"]) for s in SUBTYPES)
        assert not any(pd.isna(df.loc["c2", f"j2::{s}"]) for s in SUBTYPES)

    def test_index_name_is_claim_id(self):
        fr = _make_flagging_results(["j1"], ["c1"])
        df = build_claim_matrix(fr)
        assert df.index.name == "claim_id"

    def test_empty_evaluations_returns_empty_df(self):
        fr = _make_flagging_results([], [])
        df = build_claim_matrix(fr)
        assert df.empty



class TestComputeMtmmCorrelationMatrix:
    def test_square_shape(self):
        fr = _make_flagging_results(["j1", "j2"], ["c1", "c2", "c3"])
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        assert corr.shape == (8, 8)

    def test_symmetric(self):
        fr = _make_flagging_results(["j1", "j2"], ["c1", "c2", "c3"])
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(1)
        data = {f"j{j}::{s}": rng.integers(0, 2, size=20).tolist()
                for j in range(2) for s in SUBTYPES}
        df = pd.DataFrame(data)
        corr = compute_mtmm_correlation_matrix(df)
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_identical_columns_use_zero_cell_correction(self):
        data = {"j1::Falsehood": [1, 0, 1, 0], "j2::Falsehood": [1, 0, 1, 0]}
        df = pd.DataFrame(data)
        corr = compute_mtmm_correlation_matrix(df)
        doubled = [4, 1, 1, 4]
        expected = tetrachoric(np.repeat([0, 0, 1, 1], doubled), np.repeat([0, 1, 0, 1], doubled))
        assert corr.loc["j1::Falsehood", "j2::Falsehood"] == pytest.approx(expected)

    def test_opposite_columns_use_zero_cell_correction(self):
        data = {"j1::Falsehood": [1, 0, 1, 0], "j2::Falsehood": [0, 1, 0, 1]}
        df = pd.DataFrame(data)
        corr = compute_mtmm_correlation_matrix(df)
        doubled = [1, 4, 4, 1]
        expected = tetrachoric(np.repeat([0, 0, 1, 1], doubled), np.repeat([0, 1, 0, 1], doubled))
        assert corr.loc["j1::Falsehood", "j2::Falsehood"] == pytest.approx(expected)

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="empty"):
            compute_mtmm_correlation_matrix(pd.DataFrame())



class TestClassifyMtmmCorrelations:
    def _make_corr(self, n_judges=2, n_subtypes=4):
        """
        Fully-crossed random-flag matrix for classification tests.
        """
        rng = np.random.default_rng(0)
        n_claims = 20
        cols = [f"j{i}::{s}" for i in range(n_judges) for s in SUBTYPES[:n_subtypes]]
        data = rng.integers(0, 2, size=(n_claims, len(cols)))
        df = pd.DataFrame(data, columns=cols)
        return compute_mtmm_correlation_matrix(df)

    def test_correct_counts_two_judges(self):
        corr = self._make_corr(n_judges=2, n_subtypes=4)
        classified = classify_mtmm_correlations(corr)
        assert len(classified["heterotrait_monomethod"]) == 12
        assert len(classified["monotrait_heteromethod"]) == 4
        assert len(classified["heterotrait_heteromethod"]) == 12

    def test_correct_counts_three_judges(self):
        corr = self._make_corr(n_judges=3, n_subtypes=4)
        classified = classify_mtmm_correlations(corr)
        assert len(classified["heterotrait_monomethod"]) == 18
        assert len(classified["monotrait_heteromethod"]) == 12
        assert len(classified["heterotrait_heteromethod"]) == 36

    def test_no_diagonal_in_any_block(self):
        corr = self._make_corr()
        classified = classify_mtmm_correlations(corr)
        total = sum(len(v) for v in classified.values())
        n = len(corr)
        assert total == n * (n - 1) // 2

    def test_returns_floats(self):
        corr = self._make_corr()
        classified = classify_mtmm_correlations(corr)
        for values in classified.values():
            for v in values:
                assert isinstance(v, float)

    def test_single_judge_no_mthm_no_hthm(self):
        data = {"j1::Falsehood": [1, 0, 1], "j1::Omission": [0, 1, 0]}
        df = pd.DataFrame(data)
        corr = compute_mtmm_correlation_matrix(df)
        classified = classify_mtmm_correlations(corr)
        assert classified["monotrait_heteromethod"] == []
        assert classified["heterotrait_heteromethod"] == []
        assert len(classified["heterotrait_monomethod"]) == 1



class TestAssessConstructValidity:
    def _corr(self, judges=None, n_claims=20):
        return _minimal_corr_matrix(judges, n_claims)

    def test_mthm_exceeds_hthm_and_htmonm_when_mthm_highest(self):
        classified = {
            "monotrait_heteromethod": [0.8, 0.7, 0.9],
            "heterotrait_monomethod": [0.3, 0.2, 0.4],
            "heterotrait_heteromethod": [0.1, 0.15, 0.05],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_exceeds_hthm"] is True
        assert result["mthm_exceeds_htmonm"] is True

    def test_mthm_does_not_exceed_when_mthm_lower(self):
        classified = {
            "monotrait_heteromethod": [0.1, 0.2],
            "heterotrait_monomethod": [0.5, 0.6],
            "heterotrait_heteromethod": [0.4, 0.3],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_exceeds_hthm"] is False
        assert result["mthm_exceeds_htmonm"] is False

    def test_known_mean_values(self):
        classified = {
            "monotrait_heteromethod": [0.6, 0.8],
            "heterotrait_monomethod": [0.2, 0.4],
            "heterotrait_heteromethod": [0.0, 0.2],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mean_mthm"] == pytest.approx(0.7)
        assert result["mean_htmonm"] == pytest.approx(0.3)
        assert result["mean_hthm"] == pytest.approx(0.1)

    def test_pct_above_hthm_mean(self):
        classified = {
            "monotrait_heteromethod": [0.5, 0.2, 0.8],
            "heterotrait_monomethod": [0.3],
            "heterotrait_heteromethod": [0.3, 0.3],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_pct_above_hthm_mean"] == pytest.approx(2 / 3)

    def test_empty_mthm_returns_nan(self):
        classified = {
            "monotrait_heteromethod": [],
            "heterotrait_monomethod": [0.3],
            "heterotrait_heteromethod": [0.1],
        }
        result = assess_construct_validity(classified, self._corr())
        assert math.isnan(result["mean_mthm"])

    def test_all_required_keys_present(self):
        corr = self._corr()
        classified = classify_mtmm_correlations(corr)
        result = assess_construct_validity(classified, corr)
        expected_keys = {
            "mean_mthm", "mean_htmonm", "mean_hthm",
            "mthm_is_positive", "mthm_all_positive", "mthm_pct_positive",
            "mthm_exceeds_hthm", "mthm_exceeds_htmonm",
            "mthm_pct_above_hthm_mean",
            "mthm_meets_threshold", "mthm_significant_gt_zero",
            "mthm_all_exceed_row_col_hthm", "mthm_pct_exceed_row_col_hthm",
            "mthm_all_exceed_local_htmonm", "mthm_pct_exceed_local_htmonm",
            "block_ordering_holds",
        }
        assert set(result.keys()) == expected_keys

    def test_mthm_is_positive_false_when_mthm_negative(self):
        classified = {
            "monotrait_heteromethod": [-0.05, -0.03],
            "heterotrait_monomethod": [-0.20, -0.15],
            "heterotrait_heteromethod": [-0.10, -0.12],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_is_positive"] is False

    def test_mthm_all_positive_true_when_all_coeffs_positive(self):
        classified = {
            "monotrait_heteromethod": [0.4, 0.6, 0.5],
            "heterotrait_monomethod": [0.2],
            "heterotrait_heteromethod": [0.1],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_all_positive"] is True

    def test_mthm_all_positive_false_when_one_coeff_negative(self):
        classified = {
            "monotrait_heteromethod": [0.5, -0.1, 0.4],
            "heterotrait_monomethod": [0.2],
            "heterotrait_heteromethod": [0.1],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_all_positive"] is False

    def test_mthm_meets_threshold_when_above_conv_threshold(self):
        classified = {
            "monotrait_heteromethod": [0.4, 0.5, 0.6],
            "heterotrait_monomethod": [0.1],
            "heterotrait_heteromethod": [0.05],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_meets_threshold"] is True

    def test_mthm_does_not_meet_threshold_when_below(self):
        classified = {
            "monotrait_heteromethod": [0.1, 0.2],
            "heterotrait_monomethod": [0.05],
            "heterotrait_heteromethod": [0.01],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_meets_threshold"] is False

    def test_mthm_significant_gt_zero_with_clearly_positive_values(self):
        rng = np.random.default_rng(42)
        mthm = (0.7 + rng.uniform(0.0, 0.2, size=20)).tolist()
        classified = {
            "monotrait_heteromethod": mthm,
            "heterotrait_monomethod": [0.1],
            "heterotrait_heteromethod": [0.05],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_significant_gt_zero"] is True

    def test_mthm_not_significant_gt_zero_with_near_zero_values(self):
        classified = {
            "monotrait_heteromethod": [0.05, -0.03],
            "heterotrait_monomethod": [0.1],
            "heterotrait_heteromethod": [0.05],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["mthm_significant_gt_zero"] is False

    def test_mthm_all_exceed_row_col_hthm_when_judges_agree(self):
        rng = np.random.default_rng(9)
        flags = rng.integers(0, 2, size=(30, 4))

        def flag_fn(judge, claim_id, subtype):
            idx = int(claim_id[1:])
            return bool(flags[idx, SUBTYPES.index(subtype)])

        fr = _make_flagging_results(["j1", "j2"], [f"c{i}" for i in range(30)], flag_fn=flag_fn)
        corr = compute_mtmm_correlation_matrix(build_claim_matrix(fr))
        classified = classify_mtmm_correlations(corr)
        result = assess_construct_validity(classified, corr)
        assert isinstance(result["mthm_all_exceed_row_col_hthm"], bool)
        assert isinstance(result["mthm_pct_exceed_row_col_hthm"], float)

    def test_mthm_all_exceed_local_htmonm_structural(self):
        rng = np.random.default_rng(11)
        flags = rng.integers(0, 2, size=(30, 4))

        def flag_fn(judge, claim_id, subtype):
            idx = int(claim_id[1:])
            return bool(flags[idx, SUBTYPES.index(subtype)])

        fr = _make_flagging_results(["j1", "j2"], [f"c{i}" for i in range(30)], flag_fn=flag_fn)
        corr = compute_mtmm_correlation_matrix(build_claim_matrix(fr))
        classified = classify_mtmm_correlations(corr)
        result = assess_construct_validity(classified, corr)
        assert isinstance(result["mthm_all_exceed_local_htmonm"], bool)
        assert isinstance(result["mthm_pct_exceed_local_htmonm"], float)

    def test_structural_checks_true_when_judges_identical(self):
        rng = np.random.default_rng(5)
        flags = rng.integers(0, 2, size=(40, 4))

        def flag_fn(judge, claim_id, subtype):
            idx = int(claim_id[1:])
            return bool(flags[idx, SUBTYPES.index(subtype)])

        fr = _make_flagging_results(["j1", "j2"], [f"c{i}" for i in range(40)], flag_fn=flag_fn)
        corr = compute_mtmm_correlation_matrix(build_claim_matrix(fr))
        classified = classify_mtmm_correlations(corr)
        result = assess_construct_validity(classified, corr)
        assert result["mthm_all_exceed_row_col_hthm"] is True
        assert result["mthm_all_exceed_local_htmonm"] is True
        assert result["mthm_pct_exceed_row_col_hthm"] == pytest.approx(1.0)
        assert result["mthm_pct_exceed_local_htmonm"] == pytest.approx(1.0)

    def test_block_ordering_holds_when_mthm_gt_htmonm_gt_hthm(self):
        classified = {
            "monotrait_heteromethod": [0.7, 0.8],
            "heterotrait_monomethod": [0.4, 0.5],
            "heterotrait_heteromethod": [0.1, 0.2],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["block_ordering_holds"] is True

    def test_block_ordering_fails_when_htmonm_exceeds_mthm(self):
        classified = {
            "monotrait_heteromethod": [0.3, 0.4],
            "heterotrait_monomethod": [0.5, 0.6],
            "heterotrait_heteromethod": [0.1, 0.2],
        }
        result = assess_construct_validity(classified, self._corr())
        assert result["block_ordering_holds"] is False



class TestGetHeterotraitMonomethodBlock:
    def test_block_shape(self):
        fr = _make_flagging_results(["j1", "j2"], [f"c{i}" for i in range(10)])
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        block = _get_heterotrait_monomethod_block(corr, "j1", SUBTYPES)
        assert block.shape == (4, 4)

    def test_diagonal_is_one(self):
        rng = np.random.default_rng(2)
        data = {f"j1::{s}": rng.integers(0, 2, size=20).tolist() for s in SUBTYPES}
        df = pd.DataFrame(data)
        corr = compute_mtmm_correlation_matrix(df)
        block = _get_heterotrait_monomethod_block(corr, "j1", SUBTYPES)
        np.testing.assert_allclose(np.diag(block), 1.0, atol=1e-10)



class TestCheckTraitPatternConsistency:
    def _identical_judge_fr(self, n_claims=20):
        """
        Both judges always agree, so their HTMonoM blocks are identical.
        """
        rng = np.random.default_rng(7)
        flags = rng.integers(0, 2, size=(n_claims, 4))

        def flag_fn(judge, claim_id, subtype):
            idx = int(claim_id[1:])
            s_idx = SUBTYPES.index(subtype)
            return bool(flags[idx, s_idx])

        return _make_flagging_results(
            ["j1", "j2"],
            [f"c{i}" for i in range(n_claims)],
            flag_fn=flag_fn,
        )

    def test_identical_judges_high_consistency(self):
        fr = self._identical_judge_fr()
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        result = check_trait_pattern_consistency(corr, ["j1", "j2"], SUBTYPES)
        assert result["mean_consistency"] == pytest.approx(1.0, abs=1e-6)

    def test_pair_count_two_judges(self):
        fr = self._identical_judge_fr()
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        result = check_trait_pattern_consistency(corr, ["j1", "j2"], SUBTYPES)
        assert len(result["pairwise_spearman"]) == 1

    def test_pair_count_three_judges(self):
        rng = np.random.default_rng(3)
        n_claims = 20
        flags = rng.integers(0, 2, size=(n_claims, 4))

        def flag_fn(judge, claim_id, subtype):
            idx = int(claim_id[1:])
            s_idx = SUBTYPES.index(subtype)
            return bool(flags[idx, s_idx])

        fr = _make_flagging_results(
            ["j1", "j2", "j3"],
            [f"c{i}" for i in range(n_claims)],
            flag_fn=flag_fn,
        )
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        result = check_trait_pattern_consistency(corr, ["j1", "j2", "j3"], SUBTYPES)
        assert len(result["pairwise_spearman"]) == 3

    def test_mean_consistency_in_range(self):
        fr = self._identical_judge_fr()
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        result = check_trait_pattern_consistency(corr, ["j1", "j2"], SUBTYPES)
        assert -1.0 <= result["mean_consistency"] <= 1.0

    def test_single_judge_returns_nan_mean(self):
        fr = _make_flagging_results(["j1"], [f"c{i}" for i in range(10)])
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        result = check_trait_pattern_consistency(corr, ["j1"], SUBTYPES)
        assert math.isnan(result["mean_consistency"])
        assert result["pairwise_spearman"] == {}

    def test_result_keys_present(self):
        fr = self._identical_judge_fr()
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        result = check_trait_pattern_consistency(corr, ["j1", "j2"], SUBTYPES)
        assert "pairwise_spearman" in result
        assert "mean_consistency" in result

    def test_pattern_meets_threshold_key_present_and_boolean(self):
        fr = self._identical_judge_fr()
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        result = check_trait_pattern_consistency(corr, ["j1", "j2"], SUBTYPES)
        assert "pattern_meets_threshold" in result
        assert isinstance(result["pattern_meets_threshold"], bool)
        assert result["pattern_meets_threshold"] is True



class TestSaveMtmmResults:
    def _make_results(self):
        fr = _make_flagging_results(["j1", "j2"], [f"c{i}" for i in range(10)])
        df = build_claim_matrix(fr)
        corr = compute_mtmm_correlation_matrix(df)
        classified = classify_mtmm_correlations(corr)
        validity = assess_construct_validity(classified, corr)
        judges = ["j1", "j2"]
        consistency = check_trait_pattern_consistency(corr, judges, SUBTYPES)
        return {
            "claim_matrix": df,
            "corr_matrix": corr,
            "classified": classified,
            "validity": validity,
            "consistency": consistency,
            "judges": judges,
            "subtypes": SUBTYPES,
        }

    def test_json_file_created(self, tmp_path):
        results = self._make_results()
        save_mtmm_results(results, str(tmp_path))
        assert os.path.isfile(tmp_path / "mtmm_results.json")

    def test_csv_file_created(self, tmp_path):
        results = self._make_results()
        save_mtmm_results(results, str(tmp_path))
        assert os.path.isfile(tmp_path / "mtmm_corr_matrix.csv")

    def test_json_has_expected_keys(self, tmp_path):
        results = self._make_results()
        save_mtmm_results(results, str(tmp_path))
        with open(tmp_path / "mtmm_results.json") as f:
            data = json.load(f)
        assert "judges" in data
        assert "subtypes" in data
        assert "validity" in data
        assert "consistency" in data
        assert "classified_means" in data

    def test_csv_recoverable_as_dataframe(self, tmp_path):
        results = self._make_results()
        save_mtmm_results(results, str(tmp_path))
        recovered = pd.read_csv(tmp_path / "mtmm_corr_matrix.csv", index_col=0)
        assert recovered.shape == results["corr_matrix"].shape

    def test_creates_output_dir(self, tmp_path):
        results = self._make_results()
        new_dir = str(tmp_path / "nested" / "output")
        save_mtmm_results(results, new_dir)
        assert os.path.isdir(new_dir)



class TestRunMtmm:
    def _write_fr(self, tmp_path, judges, n_claims=15):
        fr = _make_flagging_results(
            judges,
            [f"c{i}" for i in range(n_claims)],
        )
        path = str(tmp_path / "flagging_results.json")
        _write_flagging_results(path, fr)
        return path

    def test_output_files_created(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        out = str(tmp_path / "mtmm_out")
        run_mtmm(path, out)
        assert os.path.isfile(os.path.join(out, "mtmm_results.json"))
        assert os.path.isfile(os.path.join(out, "mtmm_corr_matrix.csv"))
        assert os.path.isfile(os.path.join(out, "mtmm_heatmap.png"))
        assert os.path.isfile(os.path.join(out, "mtmm_summary.png"))

    def test_result_keys_present(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        results = run_mtmm(path, str(tmp_path / "out"))
        for key in ("claim_matrix", "corr_matrix", "classified",
                    "validity", "consistency", "judges", "subtypes"):
            assert key in results

    def test_judges_extracted_correctly(self, tmp_path):
        path = self._write_fr(tmp_path, ["alpha", "beta", "gamma"])
        results = run_mtmm(path, str(tmp_path / "out"))
        assert sorted(results["judges"]) == ["alpha", "beta", "gamma"]

    def test_subtypes_match_constants(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        results = run_mtmm(path, str(tmp_path / "out"))
        assert set(results["subtypes"]) == set(SUBTYPES)

    def test_raises_for_single_judge(self, tmp_path):
        fr = _make_flagging_results(["only_j"], [f"c{i}" for i in range(5)])
        path = str(tmp_path / "flagging_results.json")
        _write_flagging_results(path, fr)
        with pytest.raises(ValueError, match="at least 2 judges"):
            run_mtmm(path, str(tmp_path / "out"))

    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            run_mtmm(str(tmp_path / "nonexistent.json"), str(tmp_path / "out"))

    def test_verbose_runs_without_error(self, tmp_path, capsys):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        run_mtmm(path, str(tmp_path / "out"), verbose=True)
        captured = capsys.readouterr()
        assert "Loaded" in captured.out
        assert "Claim matrix" in captured.out
        assert "Correlation matrix" in captured.out

    def test_directory_input_merges_files(self, tmp_path):
        results_dir = tmp_path / "pilot"
        for subdir in ["modelA/eval", "modelB/eval"]:
            os.makedirs(results_dir / subdir)
            fr = _make_flagging_results(["j1", "j2"], [f"c{i}" for i in range(10)])
            _write_flagging_results(str(results_dir / subdir / "flagging_results.json"), fr)
        results = run_mtmm(str(results_dir), str(tmp_path / "out"))
        assert results["claim_matrix"].shape[0] == 20

    def test_directory_raises_when_no_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No flagging_results.json files found"):
            run_mtmm(str(empty_dir), str(tmp_path / "out"))
