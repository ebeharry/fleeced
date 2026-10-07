import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analysis.efa import (
    _build_efa_correlation_matrix,
    _check_simple_loading_structure,
    _compute_factor_importance,
    _drop_constant_cols,
    _ensure_positive_definite,
    _fit_efa_loadings,
    _run_parallel_analysis,
    max_abs_factor_correlation,
    run_efa,
    save_efa_results,
    type_purity,
)
from src.analysis.mtmm import SUBTYPES, compute_mtmm_correlation_matrix



def _make_flagging_results(
    judges: list[str],
    claim_ids: list[str],
    flag_fn=None,
) -> dict:
    """
    Minimal flagging_results dict with controlled flag values.
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


def _make_binary_df(n_rows: int = 20, n_cols: int = 4, seed: int = 0) -> pd.DataFrame:
    """
    Random binary DataFrame — all columns vary so none are dropped.
    """
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=(n_rows, n_cols))
    cols = [f"j{i // len(SUBTYPES)}::{SUBTYPES[i % len(SUBTYPES)]}" for i in range(n_cols)]
    return pd.DataFrame(data, columns=cols)


def _make_structured_df(n_claims: int = 40, seed: int = 42) -> pd.DataFrame:
    """
    Two-factor binary matrix: cols 0-3 load on factor 1, cols 4-7 on factor 2.
    Designed so parallel analysis retains at least 1 factor.
    """
    rng = np.random.default_rng(seed)
    f1 = rng.standard_normal(n_claims)
    f2 = rng.standard_normal(n_claims)
    cols = {}
    for i in range(4):
        prob = 1 / (1 + np.exp(-(f1 + 0.3 * rng.standard_normal(n_claims))))
        cols[f"j0::{SUBTYPES[i % 4]}"] = (rng.uniform(size=n_claims) < prob).astype(float)
    for i in range(4):
        prob = 1 / (1 + np.exp(-(f2 + 0.3 * rng.standard_normal(n_claims))))
        cols[f"j1::{SUBTYPES[i % 4]}"] = (rng.uniform(size=n_claims) < prob).astype(float)
    return pd.DataFrame(cols)



class TestDropConstantCols:
    def test_drops_all_zero_column(self):
        df = pd.DataFrame({"a": [1, 0, 1, 0], "b": [0, 0, 0, 0]})
        result, cols = _drop_constant_cols(df)
        assert "b" not in result.columns
        assert "a" in result.columns

    def test_drops_all_one_column(self):
        df = pd.DataFrame({"a": [1, 0, 1], "b": [1, 1, 1]})
        result, cols = _drop_constant_cols(df)
        assert "b" not in result.columns

    def test_keeps_varying_columns(self):
        df = _make_binary_df(n_rows=10, n_cols=4)
        result, cols = _drop_constant_cols(df)
        assert result.shape[1] == len(cols)
        assert len(cols) > 0

    def test_returns_tuple(self):
        df = _make_binary_df()
        result = _drop_constant_cols(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_retained_col_names_match_df_columns(self):
        df = _make_binary_df(n_rows=10)
        filtered_df, col_names = _drop_constant_cols(df)
        assert list(filtered_df.columns) == col_names

    def test_no_constant_cols_returns_all(self):
        df = _make_binary_df(n_rows=10, n_cols=4)
        filtered_df, col_names = _drop_constant_cols(df)
        assert filtered_df.shape[1] == df.shape[1]



class TestEnsurePositiveDefinite:
    def test_positive_definite_input_unchanged(self):
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        np.testing.assert_array_equal(_ensure_positive_definite(corr), corr)

    def test_non_positive_definite_is_repaired(self):
        corr = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        assert np.linalg.eigvalsh(corr).min() < 0
        fixed = _ensure_positive_definite(corr)
        assert np.linalg.eigvalsh(fixed).min() > 0

    def test_repaired_matrix_has_unit_diagonal_and_is_symmetric(self):
        corr = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
        fixed = _ensure_positive_definite(corr)
        np.testing.assert_allclose(np.diag(fixed), 1.0, atol=1e-10)
        np.testing.assert_allclose(fixed, fixed.T, atol=1e-12)


class TestBuildEfaCorrelationMatrix:
    def test_shape_and_unit_diagonal(self):
        df = _make_structured_df(n_claims=60)
        corr = _build_efa_correlation_matrix(df)
        assert corr.shape == (df.shape[1], df.shape[1])
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-8)

    def test_symmetric_and_positive_definite(self):
        df = _make_structured_df(n_claims=60)
        corr = _build_efa_correlation_matrix(df)
        np.testing.assert_allclose(corr, corr.T, atol=1e-10)
        assert np.linalg.eigvalsh(corr).min() > 0

    def test_matches_mtmm_tetrachoric_off_diagonal(self):
        df = _make_structured_df(n_claims=80)
        corr = _build_efa_correlation_matrix(df)
        expected = compute_mtmm_correlation_matrix(df).to_numpy()
        np.testing.assert_allclose(corr, expected, atol=1e-6)

    def test_nan_cells_do_not_produce_nan(self):
        df = _make_structured_df(n_claims=60)
        df.iloc[:10, 0] = np.nan
        corr = _build_efa_correlation_matrix(df)
        assert not np.any(np.isnan(corr))


class TestRunParallelAnalysis:
    def test_returns_expected_keys(self):
        df = _make_binary_df(n_rows=30, n_cols=4)
        result = _run_parallel_analysis(df, n_random=10, seed=0)
        assert "observed_eigenvalues" in result
        assert "simulated_mean_eigenvalues" in result
        assert "n_factors_retained" in result

    def test_observed_eigenvalues_length(self):
        n_cols = 4
        df = _make_binary_df(n_rows=30, n_cols=n_cols)
        result = _run_parallel_analysis(df, n_random=10, seed=0)
        assert len(result["observed_eigenvalues"]) == n_cols

    def test_simulated_mean_length(self):
        n_cols = 4
        df = _make_binary_df(n_rows=30, n_cols=n_cols)
        result = _run_parallel_analysis(df, n_random=10, seed=0)
        assert len(result["simulated_mean_eigenvalues"]) == n_cols

    def test_n_factors_nonnegative(self):
        df = _make_binary_df(n_rows=20, n_cols=4)
        result = _run_parallel_analysis(df, n_random=10, seed=0)
        assert result["n_factors_retained"] >= 0

    def test_n_factors_leq_n_cols(self):
        n_cols = 4
        df = _make_binary_df(n_rows=30, n_cols=n_cols)
        result = _run_parallel_analysis(df, n_random=10, seed=0)
        assert result["n_factors_retained"] <= n_cols

    def test_eigenvalues_descending(self):
        df = _make_binary_df(n_rows=40, n_cols=6)
        result = _run_parallel_analysis(df, n_random=10, seed=0)
        evals = result["observed_eigenvalues"]
        assert all(evals[i] >= evals[i + 1] for i in range(len(evals) - 1))

    def test_seed_reproducibility(self):
        df = _make_binary_df(n_rows=30, n_cols=4)
        r1 = _run_parallel_analysis(df, n_random=10, seed=7)
        r2 = _run_parallel_analysis(df, n_random=10, seed=7)
        assert r1["n_factors_retained"] == r2["n_factors_retained"]
        assert r1["observed_eigenvalues"] == r2["observed_eigenvalues"]
        assert r1["simulated_mean_eigenvalues"] == r2["simulated_mean_eigenvalues"]

    def test_structured_data_retains_factors(self):
        df = _make_structured_df(n_claims=60, seed=42)
        df, _ = _drop_constant_cols(df)
        result = _run_parallel_analysis(df, n_random=30, seed=42)
        assert result["n_factors_retained"] >= 1

    def test_eigenvalues_are_list(self):
        df = _make_binary_df(n_rows=20, n_cols=3)
        result = _run_parallel_analysis(df, n_random=10, seed=0)
        assert isinstance(result["observed_eigenvalues"], list)
        assert isinstance(result["simulated_mean_eigenvalues"], list)


class TestFitEfaLoadings:
    def _make_corr(self, n_claims=60, seed=42):
        df = _make_structured_df(n_claims=n_claims, seed=seed)
        return _build_efa_correlation_matrix(df)

    def test_loadings_shape_one_factor(self):
        corr = self._make_corr()
        loadings, phi = _fit_efa_loadings(corr, n_factors=1)
        assert loadings.shape == (8, 1)
        assert phi.shape == (1, 1)

    def test_loadings_shape_two_factors(self):
        corr = self._make_corr()
        loadings, phi = _fit_efa_loadings(corr, n_factors=2)
        assert loadings.shape == (8, 2)
        assert phi.shape == (2, 2)

    def test_returns_numpy_array(self):
        corr = self._make_corr()
        loadings, phi = _fit_efa_loadings(corr, n_factors=1)
        assert isinstance(loadings, np.ndarray)
        assert isinstance(phi, np.ndarray)

    def test_deterministic(self):
        corr = self._make_corr()
        l1, phi1 = _fit_efa_loadings(corr, n_factors=2)
        l2, phi2 = _fit_efa_loadings(corr, n_factors=2)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(phi1, phi2)

    def test_factor_correlation_diagonal_is_one(self):
        corr = self._make_corr()
        _, phi = _fit_efa_loadings(corr, n_factors=2)
        np.testing.assert_allclose(np.diag(phi), 1.0, atol=1e-8)

    def test_factor_correlation_symmetric(self):
        corr = self._make_corr()
        _, phi = _fit_efa_loadings(corr, n_factors=2)
        np.testing.assert_allclose(phi, phi.T, atol=1e-8)

    def test_single_factor_correlation_is_identity(self):
        corr = self._make_corr()
        _, phi = _fit_efa_loadings(corr, n_factors=1)
        np.testing.assert_array_equal(phi, np.ones((1, 1)))

    def test_recovers_two_block_structure(self):
        corr = np.full((8, 8), 0.1)
        corr[:4, :4] = 0.6
        corr[4:, 4:] = 0.5
        np.fill_diagonal(corr, 1.0)
        loadings, phi = _fit_efa_loadings(corr, n_factors=2)
        primary = np.argmax(np.abs(loadings), axis=1)
        assert len(set(primary[:4])) == 1
        assert len(set(primary[4:])) == 1
        assert primary[0] != primary[4]
        assert abs(phi[0, 1]) < 0.6


class TestComputeFactorImportance:
    def test_orthogonal_factors_match_sum_of_squares(self):
        loadings = np.array([[0.8, 0.0], [0.6, 0.0], [0.0, 0.7]])
        result = _compute_factor_importance(loadings, np.eye(2), ["a", "b", "c"])
        assert result["factor_importance"] == pytest.approx([1.0, 0.49], abs=1e-4)
        assert result["total_variance"] == 3.0

    def test_oblique_importance_uses_squared_structure_coefficients(self):
        loadings = np.array([[0.8, 0.0], [0.0, 0.7]])
        phi = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = _compute_factor_importance(loadings, phi, ["a", "b"])
        assert result["factor_importance"] == pytest.approx([0.64 + 0.1225, 0.16 + 0.49], abs=1e-4)

    def test_total_common_pct_from_communalities(self):
        loadings = np.array([[0.8, 0.0], [0.0, 0.7]])
        phi = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = _compute_factor_importance(loadings, phi, ["a", "b"])
        assert result["total_common_pct"] == pytest.approx(
            sum(result["communalities"].values()) / 2 * 100, abs=0.01
        )

    def test_communalities_keyed_by_variable(self):
        loadings = np.array([[0.8], [0.6]])
        result = _compute_factor_importance(loadings, np.ones((1, 1)), ["a", "b"])
        assert result["communalities"]["a"] == pytest.approx(0.64, abs=1e-4)
        assert result["communalities"]["b"] == pytest.approx(0.36, abs=1e-4)


class TestCheckSimpleLoadingStructure:
    def test_returns_expected_keys(self):
        loadings = np.eye(3)
        result = _check_simple_loading_structure(loadings, ["a", "b", "c"], 0.4)
        assert "per_variable" in result
        assert "n_simple" in result
        assert "n_vars" in result
        assert "simple_structure_fraction" in result

    def test_identity_loadings_all_simple(self):
        loadings = np.eye(3)
        result = _check_simple_loading_structure(loadings, ["a", "b", "c"], 0.4)
        assert result["n_simple"] == 3
        assert result["simple_structure_fraction"] == pytest.approx(1.0)

    def test_no_salient_loading_not_simple(self):
        loadings = np.array([[0.1, 0.05], [0.7, 0.1]])
        result = _check_simple_loading_structure(loadings, ["a", "b"], 0.4)
        per_var = {v["variable"]: v for v in result["per_variable"]}
        assert per_var["a"]["is_simple"] is False
        assert per_var["b"]["is_simple"] is True

    def test_cross_loaded_not_simple(self):
        loadings = np.array([[0.7, 0.7], [0.9, 0.0]])
        result = _check_simple_loading_structure(loadings, ["a", "b"], 0.4)
        per_var = {v["variable"]: v for v in result["per_variable"]}
        assert per_var["a"]["is_simple"] is False

    def test_simple_structure_fraction_in_range(self):
        loadings = np.random.default_rng(0).standard_normal((6, 2))
        result = _check_simple_loading_structure(loadings, [str(i) for i in range(6)], 0.4)
        assert 0.0 <= result["simple_structure_fraction"] <= 1.0

    def test_per_variable_count_matches_n_vars(self):
        n_vars = 5
        loadings = np.eye(n_vars)[:, :3]
        col_names = [f"v{i}" for i in range(n_vars)]
        result = _check_simple_loading_structure(loadings, col_names, 0.4)
        assert len(result["per_variable"]) == n_vars
        assert result["n_vars"] == n_vars

    def test_per_variable_entry_has_expected_keys(self):
        loadings = np.eye(2)
        result = _check_simple_loading_structure(loadings, ["x", "y"], 0.4)
        required = {"variable", "primary_factor", "primary_loading", "max_cross_loading", "is_simple"}
        for entry in result["per_variable"]:
            assert set(entry.keys()) == required

    def test_n_simple_consistent_with_per_variable(self):
        loadings = np.array([[0.9, 0.05], [0.6, 0.6], [0.0, 0.8]])
        result = _check_simple_loading_structure(loadings, ["a", "b", "c"], 0.4)
        n_simple_from_per_var = sum(1 for v in result["per_variable"] if v["is_simple"])
        assert result["n_simple"] == n_simple_from_per_var

    @pytest.mark.parametrize("threshold", [0.2, 0.4, 0.6])
    def test_stricter_threshold_not_more_simple(self, threshold):
        rng = np.random.default_rng(5)
        loadings = rng.standard_normal((8, 3))
        col_names = [f"v{i}" for i in range(8)]
        result_strict = _check_simple_loading_structure(loadings, col_names, 0.2)
        result_loose = _check_simple_loading_structure(loadings, col_names, 0.6)
        assert result_strict["n_simple"] <= result_loose["n_simple"]

    def test_primary_factor_one_indexed(self):
        loadings = np.array([[0.9, 0.1], [0.1, 0.9]])
        result = _check_simple_loading_structure(loadings, ["a", "b"], 0.4)
        factors = {v["variable"]: v["primary_factor"] for v in result["per_variable"]}
        assert factors["a"] == 1
        assert factors["b"] == 2



class TestTypePurity:
    @pytest.fixture
    def loadings(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Factor 1": [0.8, 0.5, 0.45, 0.1], "Factor 2": [0.0, 0.1, 0.42, 0.9]},
            index=["j1::Falsehood", "j2::Falsehood", "j1::Omission", "j2::Omission"],
        )

    def test_largest_type_group_per_factor(self, loadings):
        # Factor 1 salient: F, F, O (largest group 2); Factor 2 salient: O, O (largest 2).
        assert type_purity(loadings, 0.40) == pytest.approx(100 * 4 / 5)

    def test_cross_loaded_variable_counts_on_each_factor(self):
        loadings = pd.DataFrame(
            {"Factor 1": [0.6, 0.5], "Factor 2": [0.5, 0.0]},
            index=["j1::Falsehood", "j2::Omission"],
        )
        assert type_purity(loadings, 0.40) == pytest.approx(100 * 2 / 3)

    @pytest.mark.parametrize("threshold, expected", [(0.46, 100.0), (0.85, 100.0), (0.40, 80.0)])
    def test_threshold_controls_salience(self, loadings, threshold, expected):
        assert type_purity(loadings, threshold) == pytest.approx(expected)

    def test_negative_loadings_are_salient(self):
        loadings = pd.DataFrame({"Factor 1": [-0.7, 0.6]}, index=["j1::Falsehood", "j2::Omission"])
        assert type_purity(loadings, 0.40) == pytest.approx(50.0)

    def test_none_when_nothing_salient(self, loadings):
        assert type_purity(loadings, 0.95) is None


class TestMaxAbsFactorCorrelation:
    @pytest.mark.parametrize("matrix, expected", [
        ([[1.0, -0.6], [-0.6, 1.0]], 0.6),
        ([[1.0, 0.2, 0.4], [0.2, 1.0, -0.3], [0.4, -0.3, 1.0]], 0.4),
    ])
    def test_largest_off_diagonal_magnitude(self, matrix, expected):
        assert max_abs_factor_correlation(pd.DataFrame(matrix)) == pytest.approx(expected)

    def test_none_for_one_factor(self):
        assert max_abs_factor_correlation(pd.DataFrame([[1.0]])) is None


class TestSaveEfaResults:
    def _make_results(self, include_loadings=True):
        df = _make_binary_df(n_rows=20, n_cols=4)
        parallel = {
            "observed_eigenvalues": [2.0, 1.0, 0.5, 0.3],
            "simulated_mean_eigenvalues": [1.5, 1.2, 0.9, 0.7],
            "n_factors_retained": 1,
        }
        loadings_df = None
        simple_structure = None
        if include_loadings:
            loadings_df = pd.DataFrame([[0.8], [0.5], [0.2], [-0.1]], columns=["Factor 1"])
            simple_structure = {
                "n_simple": 3,
                "n_vars": 4,
                "simple_structure_fraction": 0.75,
                "per_variable": [],
            }
        return {
            "n_claims": 20,
            "n_vars": 4,
            "parallel": parallel,
            "loadings_df": loadings_df,
            "simple_structure": simple_structure,
            "col_names": list(df.columns),
        }

    def test_creates_json_file(self, tmp_path):
        results = self._make_results()
        save_efa_results(results, str(tmp_path))
        assert os.path.isfile(tmp_path / "efa_results.json")

    def test_creates_csv_when_loadings_present(self, tmp_path):
        results = self._make_results(include_loadings=True)
        save_efa_results(results, str(tmp_path))
        assert os.path.isfile(tmp_path / "efa_loadings.csv")

    def test_no_csv_when_no_loadings(self, tmp_path):
        results = self._make_results(include_loadings=False)
        save_efa_results(results, str(tmp_path))
        assert not os.path.isfile(tmp_path / "efa_loadings.csv")

    def test_json_has_required_keys(self, tmp_path):
        results = self._make_results()
        save_efa_results(results, str(tmp_path))
        with open(tmp_path / "efa_results.json") as f:
            data = json.load(f)
        for key in ("n_claims", "n_vars", "n_factors_retained",
                    "observed_eigenvalues", "simulated_mean_eigenvalues", "simple_structure"):
            assert key in data

    def test_json_n_factors_matches_parallel(self, tmp_path):
        results = self._make_results()
        save_efa_results(results, str(tmp_path))
        with open(tmp_path / "efa_results.json") as f:
            data = json.load(f)
        assert data["n_factors_retained"] == results["parallel"]["n_factors_retained"]

    def test_creates_output_dir_if_missing(self, tmp_path):
        results = self._make_results()
        nested = str(tmp_path / "deep" / "nested")
        save_efa_results(results, nested)
        assert os.path.isdir(nested)

    def test_csv_is_recoverable(self, tmp_path):
        results = self._make_results(include_loadings=True)
        save_efa_results(results, str(tmp_path))
        recovered = pd.read_csv(tmp_path / "efa_loadings.csv", index_col=0)
        assert recovered.shape == results["loadings_df"].shape



class TestRunEfa:
    def _write_fr(self, tmp_path, judges, n_claims=20) -> str:
        fr = _make_flagging_results(
            judges,
            [f"c{i}" for i in range(n_claims)],
            flag_fn=lambda j, c, s: int(c[1:]) % 2 == 0,
        )
        path = str(tmp_path / "flagging_results.json")
        _write_flagging_results(path, fr)
        return path

    def test_returns_expected_keys(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        results = run_efa(path, str(tmp_path / "out"))
        for key in ("claim_matrix", "parallel", "loadings_df", "simple_structure",
                    "n_claims", "n_vars", "col_names"):
            assert key in results

    def test_type_purity_and_max_phi_saved(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        out_dir = str(tmp_path / "out")
        results = run_efa(path, out_dir)
        with open(os.path.join(out_dir, "efa_results.json")) as f:
            saved = json.load(f)
        for key in ("type_purity", "max_abs_factor_correlation"):
            assert key in results
            assert saved[key] == results[key]

    def test_n_claims_set(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"], n_claims=15)
        results = run_efa(path, str(tmp_path / "out"))
        assert results["n_claims"] > 0

    def test_claim_matrix_binary(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        results = run_efa(path, str(tmp_path / "out"))
        vals = results["claim_matrix"].fillna(0).values.flatten()
        assert set(vals).issubset({0, 1})

    def test_output_json_created(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        out_dir = str(tmp_path / "out")
        run_efa(path, out_dir)
        assert os.path.isfile(os.path.join(out_dir, "efa_results.json"))

    def test_output_scree_plot_created(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        out_dir = str(tmp_path / "out")
        run_efa(path, out_dir)
        assert os.path.isfile(os.path.join(out_dir, "efa_scree_plot.png"))

    def test_raises_on_empty_directory(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No flagging_results.json"):
            run_efa(str(empty), str(tmp_path / "out"))

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            run_efa(str(tmp_path / "nonexistent.json"), str(tmp_path / "out"))

    def test_raises_on_empty_evaluations(self, tmp_path):
        path = str(tmp_path / "flagging_results.json")
        _write_flagging_results(path, {"flagging_models": [], "claim_evaluations": []})
        with pytest.raises(ValueError, match="No claim evaluations"):
            run_efa(path, str(tmp_path / "out"))

    def test_verbose_runs_without_error(self, tmp_path, capsys):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        run_efa(path, str(tmp_path / "out"), verbose=True)
        out = capsys.readouterr().out
        assert "Loaded" in out

    def test_directory_input_merges_files(self, tmp_path):
        results_dir = tmp_path / "pilot"
        for model in ["modelA", "modelB"]:
            os.makedirs(results_dir / model / "eval")
            fr = _make_flagging_results(
                ["j1", "j2"],
                [f"c{i}" for i in range(10)],
                flag_fn=lambda j, c, s: int(c[1:]) % 2 == 0,
            )
            _write_flagging_results(
                str(results_dir / model / "eval" / "flagging_results.json"), fr
            )
        results = run_efa(str(results_dir), str(tmp_path / "out"))
        assert results["n_claims"] == 20

    def test_seed_reproducibility(self, tmp_path):
        fr = _make_flagging_results(
            ["j1", "j2"],
            [f"c{i}" for i in range(30)],
            flag_fn=lambda j, c, s: int(c[1:]) % 3 == 0,
        )
        path = str(tmp_path / "flagging_results.json")
        _write_flagging_results(path, fr)
        r1 = run_efa(path, str(tmp_path / "out1"), seed=42)
        r2 = run_efa(path, str(tmp_path / "out2"), seed=42)
        assert r1["parallel"]["n_factors_retained"] == r2["parallel"]["n_factors_retained"]

    def test_constant_columns_removed(self, tmp_path):
        def flag_fn(j, c, s):
            if s == "Paltering":
                return False
            return int(c[1:]) % 2 == 0
        fr = _make_flagging_results(
            ["j1", "j2"],
            [f"c{i}" for i in range(20)],
            flag_fn=flag_fn,
        )
        path = str(tmp_path / "flagging_results.json")
        _write_flagging_results(path, fr)
        results = run_efa(path, str(tmp_path / "out"))
        assert not any("Paltering" in c for c in results["col_names"])

    def test_col_names_count_matches_n_vars(self, tmp_path):
        path = self._write_fr(tmp_path, ["j1", "j2"])
        results = run_efa(path, str(tmp_path / "out"))
        assert len(results["col_names"]) == results["n_vars"]

    def test_zero_factors_gives_none_loadings(self, tmp_path):
        rng = np.random.default_rng(99)
        n_claims = 8
        claim_ids = [f"c{i}" for i in range(n_claims)]
        fr = _make_flagging_results(
            ["j1", "j2"],
            claim_ids,
            flag_fn=lambda j, c, s: bool(rng.integers(0, 2)),
        )
        path = str(tmp_path / "flagging_results.json")
        _write_flagging_results(path, fr)
        results = run_efa(path, str(tmp_path / "out"), n_random=500, seed=0)
        if results["parallel"]["n_factors_retained"] == 0:
            assert results["loadings_df"] is None
            assert results["simple_structure"] is None
