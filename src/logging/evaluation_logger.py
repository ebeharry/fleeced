import json
import os

from ._common import _now

class _BaseEvaluationLogger:
    """
    Shared append-only logging plumbing for the flagging and self-assessment
    evaluation loggers.

    :param eval_dir: Directory under which the "evaluations" subdirectory is created.
    :param filename: Name of the JSON file this logger writes to.
    :param model_field: Key under which the model field is stored in ``data``.
    :param model_value: Value of the model field (a single model name or a list of them).
    :param results_key: Key under which the list of result entries is stored in ``data``.
    """

    def __init__(self, eval_dir: str, filename: str, model_field: str, model_value, results_key: str):
        self.eval_dir = os.path.join(eval_dir, "evaluations")
        os.makedirs(self.eval_dir, exist_ok=True)
        self._filename = filename
        self._results_key = results_key

        self.data = {
            "created_timestamp": _now(),
            model_field: model_value,
            "summary": None,
            results_key: [],
            "errors": [],
        }

        self._save()

    def log_error(self, message: str):
        """
        :param message: error message to record
        """
        self.data["errors"].append({"timestamp": _now(), "message": message})
        self._save()

    def _save(self):
        filepath = os.path.join(self.eval_dir, self._filename)
        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)


class FlaggingLogger(_BaseEvaluationLogger):
    """
    Append-only logger to track flagged statements during the first phase of
    evaluation in a JSON file for research analysis.

    :param eval_dir: Directory where flagging JSON is written.
    :param flagging_models: List of model names used for flagging.
    """

    def __init__(self, eval_dir: str, flagging_models: list[str]):
        super().__init__(eval_dir, "flagging_results.json", "flagging_models", flagging_models, "claim_evaluations")
        self.flagging_models = flagging_models

    def add_claim_result(self, flagging_model: str, conversation_file: str,
                         claim_id: str, claim_dict: dict, claim_str: str, raw_response: str, result: dict):
        """
        Append a single claim flagging result.

        :param flagging_model: Model that performed the flagging.
        :param conversation_file: Source conversation filename.
        :param claim_id: Unique identifier for the claim.
        :param claim_dict: Full claim dict with ground truth and context.
        :param claim_str: Flattened claim for logging that includes formatted ground truth and agent statement for the scenario
        :param raw_response: Raw LLM output before parsing.
        :param result: Parsed result dict with deception_indicators, reasons, is_flagged.
        """
        entry = {
            "flagging_model": flagging_model,
            "conversation_file": conversation_file,
            "claim_id": claim_id,
            "claim_type": claim_dict["claim_type"],
            "turn": claim_dict["turn"],
            "claim": claim_str,
            "deception_indicators": result["deception_indicators"],
            "reasons": result["reasons"],
            "is_flagged": result["is_flagged"],
            "raw_response": raw_response,
            "timestamp": _now()
        }
        self.data["claim_evaluations"].append(entry)
        self._save()


class AssessmentLogger(_BaseEvaluationLogger):
    """
    Append-only logger to track self-assessment MCQ results during the second
    phase of evaluation.

    :param eval_dir: Directory where assessment JSON is written.
    :param assessing_model: Model name used for self-assessment.
    """

    def __init__(self, eval_dir: str, assessing_model: str):
        super().__init__(eval_dir, "self_assessment_results.json", "assessing_model", assessing_model, "assessment_results")
        self.assessing_model = assessing_model

    def add_assessment_result(self, claim_id: str, conversation_file: str,
                              selected_answer: str, correct_answer: str,
                              is_correct: bool, reason: str,
                              options: list[str], raw_response: str,
                              mcq_question: str, agent_statement: str,
                              flagging_primary_types: list[str] | None = None):
        """
        Append a single self-assessment MCQ result.

        :param claim_id: Unique identifier linking back to the flagged claim.
        :param conversation_file: Source conversation filename.
        :param selected_answer: Letter the agent selected (A-C or "unknown").
        :param correct_answer: Letter of the correct (deceptive) option.
        :param is_correct: Whether the agent identified the deceptive statement.
        :param reason: Agent's explanation for its selection.
        :param options: The generated MCQ options (2 accurate alternatives).
        :param raw_response: Raw LLM output before parsing.
        :param mcq_question: Full MCQ question text sent to the model.
        :param agent_statement: Original agent statement that was flagged.
        :param flagging_primary_types: The primary deception types assigned during flagging.
        """
        entry = {
            "claim_id": claim_id,
            "conversation_file": conversation_file,
            "agent_statement": agent_statement,
            "flagging_primary_types": flagging_primary_types or [],
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "reason": reason,
            "options": options,
            "raw_response": raw_response,
            "mcq_question": mcq_question,
            "timestamp": _now(),
        }
        self.data["assessment_results"].append(entry)
        self._save()
