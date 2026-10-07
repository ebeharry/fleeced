import asyncio
from src.logging.evaluation_logger import FlaggingLogger, AssessmentLogger
from src.utils.litellm_utils import RETRYABLE_EXCEPTIONS, async_call_with_retry, supports_thinking_param

class ClaimEvaluator:
    """
    A scenario and evaluator agnostic class to format transcripts and evaluate them using LLM as a judge.
    """
    def __init__(self, flagging_models, assessing_model, flagger_class, assessor_class, verbose, seed=42, label=None):
        """
        :param flagging_models: A list of the models that will be used to flag potentially deceptive statements
        :param assessing_model: The model being evaluated, that must eventually self-assess itself
        :param flagger_class: A flagging class from src/evaluation/
        :param assessor_class: A asessor class from src/evaluation/
        :param verbose: Controls if each decision is printed
        :param label: Optional context string prepended to progress lines (e.g. "gpt-4.1 | Shoes (50% Benign)")
        """
        self.flagging_models = flagging_models
        self.assessing_model = assessing_model
        self.flagger = flagger_class()
        self.assessor = assessor_class()
        self.verbose = verbose
        self.label = label
        self.flagging_logger = None
        self.assessment_logger = None
        self.flag_results = []
        self.majority_gated_claims = []
        self.total_claims = None
        self.seed=seed

    def setup(self, log_dir: str):
        """
        Sets up flagger and loggers, which will process and format all transcripts.
        Assessor is set up after flagging is run.

        :param log_dir: Directory containing conversation files and where evaluation
            results will be written.
        """
        self.flagger.setup(log_dir)
        self.flagging_logger = FlaggingLogger(log_dir, self.flagging_models)
        self.assessment_logger = AssessmentLogger(log_dir, self.assessing_model)

    async def _call_mcq_model(self, model: str, sys_prompt: str, user_message: str) -> str:
        """
        Call an MCQ-style evaluation model with retry logic and return the raw response text.

        :param model: Model identifier to call.
        :param sys_prompt: System prompt for the MCQ evaluation.
        :param user_message: User-role message containing the claim or question to evaluate.
        :return: Raw text content of the model's response.
        """
        response = await async_call_with_retry(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            extra_body={"enable_thinking": False} if supports_thinking_param(model) else {},
        )
        return response.choices[0].message.content

    def _log_and_record_error(self, logger, error: Exception, context: str, retryable: bool):
        """
        Print a labeled error message and record it via the given logger.

        :param logger: The FlaggingLogger or AssessmentLogger instance to record the error on.
        :param error: The exception that was raised.
        :param context: Short description of what was being evaluated (e.g. model/file/claim identifiers).
        :param retryable: Whether the error is one of the retryable API exceptions.
        """
        prefix = f"[{self.label}] " if self.label else ""
        if retryable:
            print(f"{prefix}Rate limit error: {error}")
        else:
            print(f"{prefix}Other error: {type(error).__name__}: {error}")
        logger.log_error(f"{type(error).__name__} {context}: {error}")

    async def _flag_claim(self, model: str, filename: str, claim: dict):
        """
        Classify a single claim using MCQ-style classification.

        :param model: Model to use for flagging.
        :param filename: Source conversation filename.
        :param claim: Claim dict to evaluate.
        """
        sys_prompt = self.flagger.mcq_system_prompt()
        user_message = self.flagger.format_claim_for_mcq(claim)

        try:
            raw_response = await self._call_mcq_model(model, sys_prompt, user_message)
            result = self.flagger.extract_mcq_evaluation(raw_response)

            if self.verbose:
                flagged_types = [t for t, v in result["deception_indicators"].items() if v]
                status = ", ".join(flagged_types) if flagged_types else "NO FLAG"
                print(f"  [{claim['claim_id']}] {status}")

            self.flagging_logger.add_claim_result(
                flagging_model=model,
                conversation_file=filename,
                claim_id=claim["claim_id"],
                claim_dict=claim,
                claim_str=self.flagger.flatten_claim_for_logging(claim),
                raw_response=raw_response,
                result=result,
            )

            flagged_claim = dict(claim)
            flagged_claim.update(result)
            flagged_claim["conversation_file"] = filename
            self.flag_results.append(flagged_claim)

        except RETRYABLE_EXCEPTIONS as e:
            self._log_and_record_error(
                self.flagging_logger, e, f"model={model} file={filename} claim={claim['claim_id']}", retryable=True
            )

        except Exception as e:
            self._log_and_record_error(
                self.flagging_logger, e, f"model={model} file={filename} claim={claim['claim_id']}", retryable=False
            )

    async def _flag_conversation(self, model: str, filename: str):
        """
        Classify all claims in a conversation using MCQ-style per-claim evaluation.

        :param model: Model to use for flagging.
        :param filename: Source conversation filename.
        """
        claims = self.flagger.all_claims.get(filename, [])
        if not claims:
            if self.verbose:
                print(f"  No claims found for {filename}")
            return

        await asyncio.gather(*[self._flag_claim(model, filename, claim) for claim in claims])

    async def _model_flag_all_conversations(self, model: str):
        """
        Classify all conversations in the flagging step using the given model.

        :param model: Model to use for flagging.
        """
        for filename in self.flagger.all_claims:
            if self.verbose:
                print(f"Evaluating {filename} with {model}")
            await self._flag_conversation(model, filename)


    async def _run_flag(self):
        """
        Have flagging models iterate and classify every claim.
        """
        await asyncio.gather(*[self._model_flag_all_conversations(model) for model in self.flagging_models])

        self.majority_gated_claims = self.flagger.apply_majority_gate(self.flag_results, self.flagging_models)
        flagging_metrics = self.flagger.calculate_flagging_metrics(
            self.flagging_logger.data["claim_evaluations"],
            self.flagging_logger.data["errors"]
        )
        flagging_metrics["total_majority_gated_claims"] = len(self.majority_gated_claims)
        self.flagging_logger.data["summary"] = flagging_metrics
        self.flagging_logger._save()

    async def _assess_claim(self, claim: dict):
        """
        Self-assess one claim flagged for review.

        :param claim: A flagged claim dict from the flagging phase.
        """
        try:
            sys_prompt = self.assessor.self_eval_system_prompt()
            options = self.assessor.generate_options(claim)
            context = self.assessor.get_mcq_question_context(claim)
            correct_letter, mcq_question = self.assessor.generate_mcq_question(claim, context, options)
        except Exception as e:
            prefix = f"[{self.label}] " if self.label else ""
            print(f"{prefix}Skipping claim {claim['claim_id']}: unable to generate assessment options: {e}")
            self.assessment_logger.log_error(f"SkippedAssessment file={claim.get('conversation_file', 'unknown')} claim={claim['claim_id']}: {e}")
            return

        try:
            raw_response = await self._call_mcq_model(self.assessing_model, sys_prompt, mcq_question)
            result = self.assessor.extract_self_eval_evaluation(raw_response, correct_letter)

            if self.verbose:
                print(f"  [{claim['claim_id']}] Selected {result['selected_answer']} - Correct {correct_letter}")

            self.assessment_logger.add_assessment_result(
                claim_id=claim["claim_id"],
                conversation_file=claim["conversation_file"],
                selected_answer=result["selected_answer"],
                correct_answer=correct_letter,
                is_correct=result["is_correct"],
                reason=result["reason"],
                options=options,
                raw_response=raw_response,
                mcq_question=mcq_question,
                agent_statement=claim.get("agent_statement", ""),
                flagging_primary_types=claim.get("primary_types", []),
            )

        except RETRYABLE_EXCEPTIONS as e:
            self._log_and_record_error(
                self.assessment_logger,
                e,
                f"file={claim.get('conversation_file', 'unknown')} claim={claim['claim_id']}",
                retryable=True,
            )

        except Exception as e:
            self._log_and_record_error(
                self.assessment_logger,
                e,
                f"file={claim.get('conversation_file', 'unknown')} claim={claim['claim_id']}",
                retryable=False,
            )
    
    async def _run_assessment(self):
        """
        Self-assess all claims flagged for review.
        """
        sorted_claims = sorted(
            self.assessor.flagged_claims,
            key=lambda c: (c.get("conversation_file", ""), c.get("claim_id", ""))
        )
        for i, claim in enumerate(sorted_claims):
            claim["_eval_index"] = i
        await asyncio.gather(*[self._assess_claim(claim) for claim in self.assessor.flagged_claims])

        assessment_metrics = self.assessor.calculate_assessment_metrics(
            self.assessment_logger.data["assessment_results"],
            self.assessment_logger.data["errors"]
        )
        self.assessment_logger.data["summary"] = assessment_metrics
        self.assessment_logger._save()

    def _print_summary(self, flagging_metrics: dict, assessment_metrics: dict):
        """
        Print a single compact progress line with optional label prefix.

        :param flagging_metrics: Metrics dict returned by the flagger's calculate_flagging_metrics.
        :param assessment_metrics: Metrics dict returned by the assessor's calculate_assessment_metrics.
        """
        prefix = f"[{self.label}] " if self.label else ""

        total_evaluated = flagging_metrics.get("total_unique_claims_evaluated", 0)
        total_flagged = flagging_metrics.get("total_unique_claims_flagged", 0)
        flag_rate = flagging_metrics.get("overall_flag_rate", 0.0)
        flag_str = f"Flagged: {total_flagged}/{total_evaluated} ({flag_rate:.1%})"

        total_assessed = assessment_metrics.get("total_assessed", 0)
        if total_assessed == 0:
            assess_str = "no claims assessed"
        else:
            accuracy = assessment_metrics.get("accuracy", "N/A")
            deception_rate = assessment_metrics.get("deception_rate", "N/A")
            accuracy_str = f"{accuracy:.1%}" if isinstance(accuracy, float) else str(accuracy)
            deception_str = f"{deception_rate:.1%}" if isinstance(deception_rate, float) else str(deception_rate)
            assess_str = f"assessed: {total_assessed} | accuracy: {accuracy_str} | deception: {deception_str}"

        print(f"{prefix}{flag_str} | {assess_str}")

    async def run(self):
        """
        Evaluate all conversations with all models.
        """
        await self._run_flag()
        self.assessor.setup(self.assessing_model, self.flag_results, self.majority_gated_claims, self.seed)
        if len(self.assessor.flagged_claims) > 0:
            await self._run_assessment()
        else:
            self.assessment_logger.data["summary"] = {
                "random_seed": self.seed,
                "total_assessed": 0,
                "total_correct": "N/A",
                "total_incorrect": "N/A",
                "accuracy": "N/A",
                "false_negative_rate": "N/A",
                "deception_rate": 0.0,
                "total_errors": "N/A",
                "by_conversation": "N/A",
            }
            self.assessment_logger._save()

        self._print_summary(
            self.flagging_logger.data["summary"],
            self.assessment_logger.data["summary"],
        )
