import argparse
import asyncio
import json
import os

from .analysis.efa import run_efa, save_efa_taxonomy_figures
from .analysis.g_study import run_gstudy
from .analysis.mtmm import run_mtmm, save_mtmm_matrix_figures
from .analysis.summary import run_full_analysis
from .analysis.validity_table import SECTIONS, save_validity_table
from .engine.scenario_runner import GenerationRunner, EvaluationRunner
from .engine.trial_runner import TrialRunner


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the shared --verbose flag to a subparser.

    :param parser: the argparse subparser to add the argument to
    :return: None
    """
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbosity"
    )


def _add_flagging_results_arg(parser: argparse.ArgumentParser) -> None:
    """
    Add the shared --flagging_results argument to a subparser.

    :param parser: the argparse subparser to add the argument to
    :return: None
    """
    parser.add_argument(
        "--flagging_results",
        type=str,
        required=True,
        help="Path to a flagging_results.json file, or a directory to search recursively for all such files"
    )


def _trial_scenario(trial_dir: str) -> str:
    """
    Read the scenario of a finished trial from its manifest.

    :param trial_dir: Trial results directory containing trial_manifest.json.
    :return: Scenario type identifier.
    """
    with open(os.path.join(trial_dir, "trial_manifest.json")) as f:
        return json.load(f)["scenario"]


def _run_paper(trial_dirs: list[str], output_dir: str, salient_threshold: float, verbose: bool) -> None:
    """
    Lay out finished trials side by side: the EFA figures that combine all taxonomies,
    the MTMM matrices, and Table 4.

    Reads each trial's saved analysis outputs; nothing is recomputed.

    :param trial_dirs: Trial results directories, in display order.
    :param output_dir: Directory to write the figures and validity_table.csv/.tex.
    :param salient_threshold: Salient loading cutoff for loading outlines and type purity.
    :param verbose: If True, print each saved path.
    """
    trials = [(trial_dir, _trial_scenario(trial_dir)) for trial_dir in trial_dirs]
    paths = save_efa_taxonomy_figures(trials, output_dir, salient_threshold)
    paths.extend(save_mtmm_matrix_figures(trials, output_dir))
    paths.extend(save_validity_table(trials, output_dir, set(SECTIONS), salient_threshold))
    if verbose:
        for path in paths:
            print(f"Saved {path}")


def main():
    """
    There are two main modes: generate mode that will generate scenarios, and evaluate mode
    that will evaluate the scenarios
    """
    parser = argparse.ArgumentParser(description="Run or evaluate dialogue scenarios")
    subparsers = parser.add_subparsers(dest="mode", required=True)


    run_parser = subparsers.add_parser("run", help="Run a YAML-configured generate/evaluate/analyse pipeline")
    run_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to a YAML run config file"
    )
    run_parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Optional override for the base output directory name"
    )
    _add_common_args(run_parser)

    analyse_parser = subparsers.add_parser("analyse", help="Run full analysis on an existing trial directory")
    analyse_parser.add_argument(
        "--trial_dir",
        type=str,
        required=True,
        help="Path to trial results directory containing trial_manifest.json"
    )
    _add_common_args(analyse_parser)

    gstudy_parser = subparsers.add_parser("gstudy", help="Run G-study and D-study reliability analysis")
    gstudy_parser.add_argument(
        "--scenario",
        choices=["product_promotion", "loan_qa"],
        required=True,
        help="Scenario type to analyse"
    )
    gstudy_parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to trial results directory (contains {model}/{condition}/{subtype}/ subdirs)"
    )
    gstudy_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gstudy",
        help="Directory to write g_study JSON, d_study CSV, and plots"
    )
    gstudy_parser.add_argument(
        "--condition",
        type=str,
        default="baseline",
        help="Condition subdirectory to read (default: baseline)"
    )
    gstudy_parser.add_argument(
        "--target",
        type=float,
        default=0.85,
        help="G-coefficient reliability target for D-study reference line"
    )
    _add_common_args(gstudy_parser)

    mtmm_parser = subparsers.add_parser("mtmm", help="Run Multi-Trait Multi-Method validity analysis")
    _add_flagging_results_arg(mtmm_parser)
    mtmm_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/mtmm",
        help="Directory to write MTMM outputs (JSON, CSV, plots)"
    )
    _add_common_args(mtmm_parser)

    efa_parser = subparsers.add_parser("efa", help="Run EFA and parallel analysis on flagging results")
    _add_flagging_results_arg(efa_parser)
    efa_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/efa",
        help="Directory to write EFA outputs (JSON, CSV, plot)"
    )
    efa_parser.add_argument(
        "--n_random",
        type=int,
        default=500,
        help="Number of random matrices for parallel analysis simulation"
    )
    efa_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for parallel analysis simulation"
    )
    efa_parser.add_argument(
        "--cross_loading_threshold",
        type=float,
        default=0.40,
        help="Maximum permitted cross-loading for simple structure classification"
    )
    _add_common_args(efa_parser)

    paper_parser = subparsers.add_parser(
        "paper", help="Build the cross-scenario paper figures and Table 4 from finished trials"
    )
    paper_parser.add_argument(
        "--trial_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Trial results directories (each with trial_manifest.json), in display order"
    )
    paper_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/paper_figures",
        help="Directory to write the figures and validity_table.csv/.tex"
    )
    paper_parser.add_argument(
        "--salient_threshold",
        type=float,
        default=0.40,
        help="Salient loading cutoff for loading outlines and type purity"
    )
    _add_common_args(paper_parser)

    args = parser.parse_args()

    if args.mode == "run":
        runner = TrialRunner(
            config_path=args.config_path,
            log_dir=args.log_dir,
            verbose=args.verbose
        )
        asyncio.run(runner.run())
    
    elif args.mode == "analyse":
        run_full_analysis(args.trial_dir, verbose=args.verbose)

    elif args.mode == "gstudy":
        run_gstudy(
            results_dir=args.results_dir,
            scenario=args.scenario,
            output_dir=args.output_dir,
            condition=args.condition,
            target=args.target,
            verbose=args.verbose,
        )

    elif args.mode == "mtmm":
        run_mtmm(
            flagging_results_path=args.flagging_results,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    elif args.mode == "efa":
        run_efa(
            flagging_results_path=args.flagging_results,
            output_dir=args.output_dir,
            n_random=args.n_random,
            seed=args.seed,
            cross_loading_threshold=args.cross_loading_threshold,
            verbose=args.verbose,
        )

    elif args.mode == "paper":
        _run_paper(
            trial_dirs=args.trial_dirs,
            output_dir=args.output_dir,
            salient_threshold=args.salient_threshold,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
