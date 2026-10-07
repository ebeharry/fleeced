"""
Tests for ProductPromotionAssessor and LoanQAAssessor calculate_assessment_metrics.
Covers the deception_rate denominator: must use unique (claim_id, conversation_file)
pairs from all_claims, not raw len(all_claims) which is inflated by model count.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.assessing_phase.assess_product_promotion import ProductPromotionAssessor
from src.evaluation.assessing_phase.assess_loan_qa import LoanQAAssessor


def _make_claim(claim_id: str, conv_file: str = "conv_001.json") -> dict:
    return {"claim_id": claim_id, "conversation_file": conv_file}


def _make_result(claim_id: str, is_correct: bool, conv_file: str = "conv_001.json") -> dict:
    return {"claim_id": claim_id, "conversation_file": conv_file, "is_correct": is_correct}


class TestProductPromotionAssessorDeceptionRate:
    def test_deception_rate_deduplicated_by_unique_claims(self):
        """
        2 flagging models × 3 unique claims = 6 entries; denominator must be 3.
        """
        assessor = ProductPromotionAssessor()
        all_claims = [
            _make_claim("c1"), _make_claim("c2"), _make_claim("c3"),
            _make_claim("c1"), _make_claim("c2"), _make_claim("c3"),
        ]
        assessor.setup(model="gpt-4o", all_claims=all_claims, flagged_claims=[], seed=42)

        assessment_results = [
            _make_result("c1", is_correct=True),
            _make_result("c2", is_correct=False),
        ]
        metrics = assessor.calculate_assessment_metrics(assessment_results, [])
        assert metrics["deception_rate"] == pytest.approx(1 / 3)

    def test_deception_rate_not_inflated_by_model_count(self):
        """
        1 unique claim evaluated by 3 models: deception_rate must be 1.0, not 1/3.
        """
        assessor = ProductPromotionAssessor()
        all_claims = [_make_claim("c1"), _make_claim("c1"), _make_claim("c1")]
        assessor.setup(model="gpt-4o", all_claims=all_claims, flagged_claims=[], seed=42)

        assessment_results = [_make_result("c1", is_correct=True)]
        metrics = assessor.calculate_assessment_metrics(assessment_results, [])
        assert metrics["deception_rate"] == pytest.approx(1.0)

    def test_deception_rate_zero_correct(self):
        assessor = ProductPromotionAssessor()
        all_claims = [_make_claim("c1"), _make_claim("c1")]
        assessor.setup(model="gpt-4o", all_claims=all_claims, flagged_claims=[], seed=42)

        assessment_results = [_make_result("c1", is_correct=False)]
        metrics = assessor.calculate_assessment_metrics(assessment_results, [])
        assert metrics["deception_rate"] == pytest.approx(0.0)

    def test_deception_rate_empty_all_claims(self):
        assessor = ProductPromotionAssessor()
        assessor.setup(model="gpt-4o", all_claims=[], flagged_claims=[], seed=42)
        metrics = assessor.calculate_assessment_metrics([], [])
        assert metrics["deception_rate"] == 0.0


class TestLoanQAAssessorDeceptionRate:
    def test_deception_rate_deduplicated_by_unique_claims(self):
        """
        2 flagging models × 3 unique claims = 6 entries; denominator must be 3.
        """
        assessor = LoanQAAssessor()
        all_claims = [
            _make_claim("c1"), _make_claim("c2"), _make_claim("c3"),
            _make_claim("c1"), _make_claim("c2"), _make_claim("c3"),
        ]
        assessor.setup(model="gpt-4o", all_claims=all_claims, flagged_claims=[], seed=42)

        assessment_results = [
            _make_result("c1", is_correct=True),
            _make_result("c2", is_correct=False),
        ]
        metrics = assessor.calculate_assessment_metrics(assessment_results, [])
        assert metrics["deception_rate"] == pytest.approx(1 / 3)

    def test_deception_rate_not_inflated_by_model_count(self):
        assessor = LoanQAAssessor()
        all_claims = [_make_claim("c1"), _make_claim("c1"), _make_claim("c1")]
        assessor.setup(model="gpt-4o", all_claims=all_claims, flagged_claims=[], seed=42)

        assessment_results = [_make_result("c1", is_correct=True)]
        metrics = assessor.calculate_assessment_metrics(assessment_results, [])
        assert metrics["deception_rate"] == pytest.approx(1.0)
