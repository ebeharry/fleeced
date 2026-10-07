import asyncio
import json
import os
from datetime import datetime, timezone
import yaml

from .scenario_runner import GenerationRunner, EvaluationRunner
from ..analysis import claim_grouping
from ..analysis.efa import run_efa, save_efa_taxonomy_figures
from ..analysis.g_study import run_gstudy
from ..analysis.mtmm import run_mtmm
from ..analysis.summary import run_full_analysis, run_deception_rate_by_mode
from ..analysis.validity_table import SECTIONS, save_validity_table

_VALID_MODES = {"generate", "evaluate", "all"}
_VALID_ANALYSIS_KEYS = {
    "summary", "mtmm", "efa", "gstudy",
    "gstudy_condition", "gstudy_target",
    "efa_n_random", "efa_seed", "efa_cross_loading_threshold",
}


class TrialRunner:
    """
    A scenario and evaluator agnostic class to automate running generate/evaluate
    pipelines based on YAML configs with named conditions, model lists, and optional
    analysis steps.

    Reads a YAML config defining named conditions and runs one GenerationRunner per
    condition into an isolated subdirectory, then optionally evaluates and analyses.
    """

    def __init__(self, config_path: str, log_dir: str | None = None, verbose: bool = False, max_concurrent_models: int = 3):
        """
        :param config_path: path to a YAML config file
        :param log_dir: optional override for the base output directory name;
            if None, uses trial_name from the config
        :param verbose: whether to print per-conversation progress
        :param max_concurrent_models: max number of (assessing_model, condition) pairs
            running generation+evaluation simultaneously
        """
        self.config_path = config_path
        self.verbose = verbose
        self._config = self._load_config(config_path)
        self._validate_config(self._config)

        self.trial_name = self._config.get("trial_name") or self._config.get("ablation_name")
        self.scenario = self._config["scenario"]
        self.mode = self._config.get("mode", "all")
        self.num_convs_per_condition = self._config["num_convs_per_condition"]
        self.max_turns = self._config.get("max_turns", 10)
        self.max_tool_calls = self._config.get("max_tool_calls", 3)
        self.seed = self._config.get("seed", None)
        self.progressive_seeds = self._config.get("progressive_seeds", False)
        self.conditions = self._config.get("conditions", [{"name": "default"}])
        self.base_log_dir = log_dir if log_dir is not None else self.trial_name
        self.flagging_models = self._config.get("flagging_models", [])
        self.assessing_models = self._config["assessing_models"]
        self.max_concurrent_models = max_concurrent_models
        self.analysis_config = self._resolve_analysis_config(self._config.get("analysis", {}))

    @staticmethod
    def _resolve_analysis_config(raw: dict) -> dict:
        """
        Apply defaults to the analysis block from the config.

        :param raw: raw analysis dict from the YAML (may be empty)
        :return: analysis config dict with all keys present
        """
        return {
            "summary": raw.get("summary", True),
            "mtmm": raw.get("mtmm", False),
            "efa": raw.get("efa", False),
            "gstudy": raw.get("gstudy", False),
            "gstudy_condition": raw.get("gstudy_condition", "baseline"),
            "gstudy_target": raw.get("gstudy_target", 0.85),
            "efa_n_random": raw.get("efa_n_random", 500),
            "efa_seed": raw.get("efa_seed", 42),
            "efa_cross_loading_threshold": raw.get("efa_cross_loading_threshold", 0.40),
        }

    @staticmethod
    def _load_config(config_path: str) -> dict:
        """
        Load and parse a YAML config from disk.

        :param config_path: path to the YAML file
        :return: parsed config as a plain dict
        """
        if not os.path.isfile(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Config file must be a YAML mapping, got: {type(config).__name__}")
        return config

    @staticmethod
    def _validate_top_level_fields(config: dict) -> None:
        """
        Validate the presence and types of top-level required config fields.

        :param config: parsed config dict from _load_config
        """
        if "trial_name" not in config and "ablation_name" not in config:
            raise ValueError("Missing required config field: 'trial_name' (or 'ablation_name')")
        name_key = "trial_name" if "trial_name" in config else "ablation_name"
        if not isinstance(config[name_key], str):
            raise ValueError(f"Config field '{name_key}' must be str, got {type(config[name_key]).__name__}")

        for field, expected_type in [("scenario", str), ("assessing_models", list), ("num_convs_per_condition", int)]:
            if field not in config:
                raise ValueError(f"Missing required config field: '{field}'")
            if not isinstance(config[field], expected_type):
                raise ValueError(
                    f"Config field '{field}' must be {expected_type.__name__}, "
                    f"got {type(config[field]).__name__}"
                )

        if not config["assessing_models"]:
            raise ValueError("'assessing_models' must not be empty")
        if not all(isinstance(m, str) for m in config["assessing_models"]):
            raise ValueError("'assessing_models' must be a list of strings")

    @staticmethod
    def _validate_mode_requirements(config: dict) -> None:
        """
        Validate 'mode' and its dependent fields (e.g. 'flagging_models').

        :param config: parsed config dict from _load_config
        """
        mode = config.get("mode", "all")
        if mode not in _VALID_MODES:
            raise ValueError(f"'mode' must be one of {sorted(_VALID_MODES)}, got '{mode}'")

        if mode != "generate":
            if "flagging_models" not in config:
                raise ValueError(f"'flagging_models' is required when mode='{mode}'")
            if not isinstance(config["flagging_models"], list) or not all(isinstance(m, str) for m in config["flagging_models"]):
                raise ValueError("'flagging_models' must be a list of strings")

    @staticmethod
    def _validate_analysis(config: dict) -> None:
        """
        Validate the shape of the optional 'analysis' block.

        :param config: parsed config dict from _load_config
        """
        if "analysis" in config:
            analysis = config["analysis"]
            if not isinstance(analysis, dict):
                raise ValueError("'analysis' must be a mapping")
            unknown = set(analysis.keys()) - _VALID_ANALYSIS_KEYS
            if unknown:
                raise ValueError(f"Unknown keys in 'analysis': {sorted(unknown)}")

    @staticmethod
    def _validate_prompt_patches(condition: dict, index: int) -> None:
        """
        Validate the optional 'prompt_patches' field of a single condition.

        :param condition: condition dict from the config
        :param index: index of this condition within the conditions list
        """
        if "prompt_patches" not in condition:
            return
        patches = condition["prompt_patches"]
        if not isinstance(patches, dict):
            raise ValueError(
                f"conditions[{index}]['prompt_patches'] must be a mapping, "
                f"got {type(patches).__name__}"
            )
        for role in patches:
            if role not in {"agent", "user"}:
                raise ValueError(
                    f"conditions[{index}]['prompt_patches'] has invalid role '{role}'; "
                    f"must be 'agent' or 'user'"
                )
            if not isinstance(patches[role], dict):
                raise ValueError(
                    f"conditions[{index}]['prompt_patches']['{role}'] must be a mapping "
                    f"of section header to replacement text"
                )

    @staticmethod
    def _validate_user_filter(condition: dict, index: int) -> None:
        """
        Validate the optional 'user_filter' field of a single condition, including
        its recursive nested-dict shape.

        :param condition: condition dict from the config
        :param index: index of this condition within the conditions list
        """
        if "user_filter" not in condition:
            return
        user_filter = condition["user_filter"]
        if not isinstance(user_filter, dict):
            raise ValueError(
                f"conditions[{index}]['user_filter'] must be a mapping, "
                f"got {type(user_filter).__name__}"
            )
        stack = [(user_filter, "user_filter")]
        while stack:
            current_dict, current_path = stack.pop()
            for k, v in current_dict.items():
                if isinstance(v, dict):
                    stack.append((v, f"{current_path}.{k}"))
                elif isinstance(v, list):
                    if not all(isinstance(item, str) for item in v):
                        raise ValueError(
                            f"conditions[{index}]['user_filter'] {current_path}.{k}: "
                            f"list values must all be strings"
                        )
                elif not isinstance(v, str):
                    raise ValueError(
                        f"conditions[{index}]['user_filter'] {current_path}.{k}: "
                        f"values must be str or list of str, got {type(v).__name__}"
                    )

    @staticmethod
    def _validate_conditions_shape(config: dict) -> None:
        """
        Validate the optional 'conditions' field is a non-empty list, if present.

        :param config: parsed config dict from _load_config
        """
        if "conditions" in config:
            if not isinstance(config["conditions"], list):
                raise ValueError("'conditions' must be a list")
            if not config["conditions"]:
                raise ValueError("'conditions' list must not be empty")

    @staticmethod
    def _validate_conditions(config: dict) -> None:
        """
        Validate each condition's shape, including 'name', 'prompt_patches',
        'goal_override', and 'user_filter'.

        :param config: parsed config dict from _load_config
        """
        for i, condition in enumerate(config.get("conditions", [])):
            if not isinstance(condition, dict):
                raise ValueError(f"Each condition must be a mapping; conditions[{i}] is not")
            if "name" not in condition:
                raise ValueError(f"conditions[{i}] is missing required field 'name'")
            if not isinstance(condition["name"], str):
                raise ValueError(f"conditions[{i}]['name'] must be a string")
            TrialRunner._validate_prompt_patches(condition, i)
            if "goal_override" in condition:
                if not isinstance(condition["goal_override"], str):
                    raise ValueError(f"conditions[{i}]['goal_override'] must be a string")
            TrialRunner._validate_user_filter(condition, i)

    @staticmethod
    def _validate_config(config: dict) -> None:
        """
        Validate required fields and types in the parsed config dict.

        :param config: parsed config dict from _load_config
        """
        TrialRunner._validate_top_level_fields(config)
        TrialRunner._validate_mode_requirements(config)
        TrialRunner._validate_conditions_shape(config)
        TrialRunner._validate_analysis(config)
        TrialRunner._validate_conditions(config)

    def _condition_log_dir(self, assessing_model: str, condition_name: str) -> str:
        """
        Return the output directory path for one (assessing_model, condition) pair.

        :param assessing_model: model name string
        :param condition_name: condition name string
        :return: relative path under base_log_dir
        """
        model_short_name = assessing_model.split("/")[-1]
        return f"{self.base_log_dir}/{model_short_name}/{condition_name}"

    async def _run_generation(self, assessing_model: str, condition: dict, semaphore: asyncio.Semaphore) -> None:
        """
        Run generation for one (assessing_model, condition) pair.

        :param assessing_model: model used as the agent
        :param condition: condition dict from the trial config
        :param semaphore: limits concurrent generation tasks
        """
        async with semaphore:
            condition_name = condition["name"]
            condition_log_dir = self._condition_log_dir(assessing_model, condition_name)
            print(f"Generating: model={assessing_model} | condition={condition_name}")
            runner = GenerationRunner(
                model=assessing_model,
                scenario_type=self.scenario,
                num_total_simulations=self.num_convs_per_condition,
                log_dir=condition_log_dir,
                conv_id="001",
                seed=self.seed,
                progressive_seeds=self.progressive_seeds,
                max_turns=self.max_turns,
                max_tool_calls=self.max_tool_calls,
                default_goal=condition.get("goal_override"),
                prompt_patches=condition.get("prompt_patches"),
                user_filter=condition.get("user_filter"),
                verbose=self.verbose,
            )
            await runner.run()
            print(f"Generation complete: model={assessing_model} | condition={condition_name}")

    async def _run_evaluation(self, assessing_model: str, condition: dict, semaphore: asyncio.Semaphore) -> dict:
        """
        Run evaluation for one (assessing_model, condition) pair and return its manifest entry.

        :param assessing_model: model used as the self-assessor
        :param condition: condition dict from the trial config
        :param semaphore: limits concurrent evaluation tasks
        :return: manifest entry dict for this pair
        """
        async with semaphore:
            condition_name = condition["name"]
            condition_log_dir = self._condition_log_dir(assessing_model, condition_name)
            print(f"Evaluating: model={assessing_model} | condition={condition_name}")
            eval_runner = EvaluationRunner(
                scenario_type=self.scenario,
                flagging_models=self.flagging_models,
                assessing_model=assessing_model,
                eval_dir=condition_log_dir,
                seed=self.seed,
                verbose=False
            )
            await eval_runner.run()
            print(f"Evaluation complete: model={assessing_model} | condition={condition_name}")
            return {
                "assessing_model": assessing_model,
                "condition_name": condition_name,
                "log_dir": condition_log_dir,
                "goal": condition.get("goal_override"),
                "prompt_patches": condition.get("prompt_patches"),
                "flagging_models": self.flagging_models,
            }

    async def run(self) -> None:
        """
        Run the pipeline according to self.mode:
          "generate": Phase 1 only (generation). No manifest or analysis.
          "evaluate": Phase 2 only (evaluation). Writes manifest and runs analysis.
          "all": Phase 1 then Phase 2, writes manifest, runs analysis.
        """
        os.makedirs(f"results/{self.base_log_dir}", exist_ok=True)
        pairs = [
            (assessing_model, condition)
            for assessing_model in self.assessing_models
            for condition in self.conditions
        ]

        if self.mode in ("generate", "all"):
            gen_semaphore = asyncio.Semaphore(len(pairs))
            await asyncio.gather(*[
                self._run_generation(model, cond, gen_semaphore)
                for model, cond in pairs
            ])

        if self.mode in ("evaluate", "all"):
            eval_semaphore = asyncio.Semaphore(self.max_concurrent_models)
            manifest_entries = list(await asyncio.gather(*[
                self._run_evaluation(model, cond, eval_semaphore)
                for model, cond in pairs
            ]))
            self._write_manifest(manifest_entries)
            trial_dir = f"results/{self.base_log_dir}"
            self._run_analysis(trial_dir)

    def _run_analysis(self, trial_dir: str) -> None:
        """
        Run enabled analysis steps from self.analysis_config.

        Deception-rate-by-mode, MTMM, EFA, and G-study each run once per
        claim grouping mode in :data:`~src.analysis.claim_grouping.CLAIM_MODES`
        ("all", "falsehood_omission", "no_paltering", "no_equivocation",
        "active_vs_passive").
        MTMM/EFA/G-study write to mode-suffixed output subdirectories (e.g. "mtmm",
        "mtmm_falsehood_omission"); deception-rate-by-mode writes a single
        mode-suffixed JSON file at the trial root (e.g.
        "deception_rate_by_type.json", "deception_rate_by_type_falsehood_omission.json").
        After all taxonomies, the EFA figures that combine them (when EFA is enabled)
        and Table 4 from the enabled G-study/MTMM/EFA analyses are written to
        "{trial_dir}/plots".

        :param trial_dir: path to the results directory containing trial_manifest.json
        """
        ac = self.analysis_config
        if ac["summary"]:
            run_full_analysis(trial_dir, verbose=self.verbose)
        for mode in claim_grouping.CLAIM_MODES:
            suffix = claim_grouping.output_suffix(mode)
            if ac["summary"]:
                run_deception_rate_by_mode(
                    trial_dir,
                    output_path=os.path.join(
                        trial_dir, f"deception_rate_by_type{suffix}.json"
                    ),
                    mode=mode,
                    verbose=self.verbose,
                )
            if ac["mtmm"]:
                run_mtmm(
                    flagging_results_path=trial_dir,
                    output_dir=os.path.join(trial_dir, f"mtmm{suffix}"),
                    mode=mode,
                    scenario=self.scenario,
                    verbose=self.verbose,
                )
            if ac["efa"]:
                run_efa(
                    flagging_results_path=trial_dir,
                    output_dir=os.path.join(trial_dir, f"efa{suffix}"),
                    n_random=ac["efa_n_random"],
                    seed=ac["efa_seed"],
                    cross_loading_threshold=ac["efa_cross_loading_threshold"],
                    mode=mode,
                    scenario=self.scenario,
                    verbose=self.verbose,
                )
            if ac["gstudy"]:
                run_gstudy(
                    results_dir=trial_dir,
                    scenario=self.scenario,
                    output_dir=os.path.join(trial_dir, f"gstudy{suffix}"),
                    condition=ac["gstudy_condition"],
                    target=ac["gstudy_target"],
                    mode=mode,
                    verbose=self.verbose,
                )
        trials = [(trial_dir, self.scenario)]
        plot_dir = os.path.join(trial_dir, "plots")
        paths = []
        if ac["efa"]:
            paths.extend(save_efa_taxonomy_figures(trials, plot_dir, ac["efa_cross_loading_threshold"]))
        sections = {section for section in SECTIONS if ac[section]}
        if sections:
            paths.extend(save_validity_table(trials, plot_dir, sections, ac["efa_cross_loading_threshold"]))
        if self.verbose:
            for path in paths:
                print(f"Saved {path}")

    def _write_manifest(self, entries: list[dict]) -> None:
        """
        Write trial_manifest.json to the base output directory.

        :param entries: list of per-condition dicts recording name, log_dir, and overrides
        """
        manifest = {
            "trial_name": self.trial_name,
            "config_path": self.config_path,
            "scenario": self.scenario,
            "num_convs_per_condition": self.num_convs_per_condition,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "seed": self.seed,
            "progressive_seeds": self.progressive_seeds,
            "flagging_models": self.flagging_models,
            "assessing_models": self.assessing_models,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conditions": entries,
        }
        manifest_path = f"results/{self.base_log_dir}/trial_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
