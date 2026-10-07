import functools
import os

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams["font.size"] = 12

_PAPER_FONTSIZE = 12
_PAPER_SMALL_FONTSIZE = 11
_PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": _PAPER_FONTSIZE,
}
_STACKED_FIGSIZE = (4.5, 3.4)
_COLUMN_FIGSIZE = (3.3, 2.5)
_COLUMN_RC = {**_PAPER_RC, "font.size": 8}
_GRID_TYPE_ORDER = [
    "Falsehood", "Omission", "Equivocation", "Paltering", "Active", "Passive",
]
_GRID_TYPE_CODES = {
    "Falsehood": "F", "Omission": "O", "Equivocation": "E",
    "Paltering": "P", "Active": "Act", "Passive": "Pas",
}
_GRID_FONTSIZE = 8
_GRID_SMALL_FONTSIZE = 6.5

_DECEPTION_TYPE_ORDER = [
    "Falsehood",
    "Omission",
    "Equivocation",
    "Paltering",
]

_TYPE_COLORS = {
    "Falsehood": "#D81B60",
    "Omission": "#1E88E5",
    "Equivocation": "#FFC107",
    "Paltering": "#004D40",
}

_MODEL_COLORS = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

_MODEL_DISPLAY_NAMES = {
    "gpt-4.1": "GPT-4.1",
    "gpt-4-1": "GPT-4.1",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "deepseek-chat": "DeepSeek-V4-Flash",
    "mistral-small-2506": "Mistral-Small-2506",
    "llama-4-scout": "Llama-4-Scout",
    "gemini-2.5-flash-lite": "Gemini-2.5-Flash-Lite",
    "claude-3-haiku": "Claude-3-Haiku",
    "gpt-4o": "GPT-4o",
}


_SCENARIO_DISPLAY_NAMES = {
    "product_promotion": "Product Promotion",
    "loan_qa": "Loan Q&A",
}


PAPER_JUDGE_ORDER = [
    "gpt-5.5",
    "gpt-5.6-sol",
    "claude-sonnet-5",
    "deepseek-v4.1-flash",
    "mistral-medium-3-5",
    "llama-4-maverick",
]
PAPER_JUDGE_CODES = dict(zip(PAPER_JUDGE_ORDER, ["G5.5", "G5.6", "Cl", "DS", "Mi", "Ll"]))
PAPER_MODEL_NAMES = dict(zip(PAPER_JUDGE_ORDER, [
    "GPT-5.5",
    "GPT-5.6-sol",
    "Claude-Sonnet-5",
    "DeepSeek-V4.1-Flash",
    "Mistral-Medium-3.5",
    "Llama-4-Maverick",
]))


def _model_short_id(model: str) -> str:
    """
    Strip any provider prefix from a model identifier.

    :param model: Model identifier, e.g. "openrouter/anthropic/claude-sonnet-5".
    :return: Last path segment, e.g. "claude-sonnet-5".
    """
    return model.split("/")[-1]


def _paper_order_index(judge: str) -> int:
    """
    Return a known judge's position in :data:`PAPER_JUDGE_ORDER`.

    :param judge: Judge identifier whose short id is in :data:`PAPER_JUDGE_ORDER`.
    :return: Index in :data:`PAPER_JUDGE_ORDER`.
    """
    return PAPER_JUDGE_ORDER.index(_model_short_id(judge))


def paper_judge_order(judges: list[str]) -> list[str]:
    """
    Order judges for paper figures: judges whose short id is in :data:`PAPER_JUDGE_ORDER`
    in that order, then any others in their given order. Provider prefixes are ignored.

    :param judges: Judge identifiers, with or without provider prefixes.
    :return: Judge identifiers in display order.
    """
    unique = list(dict.fromkeys(judges))
    known = sorted((j for j in unique if _model_short_id(j) in PAPER_JUDGE_ORDER), key=_paper_order_index)
    return known + [j for j in unique if _model_short_id(j) not in PAPER_JUDGE_ORDER]


def paper_judge_labels(judges: list[str]) -> dict[str, str]:
    """
    Map judges to their short paper codes from :data:`PAPER_JUDGE_CODES`, ignoring provider prefixes.

    :param judges: Judge identifiers, with or without provider prefixes.
    :return: Judge identifier -> code, for judges with a paper code.
    """
    return {j: PAPER_JUDGE_CODES[_model_short_id(j)] for j in judges if _model_short_id(j) in PAPER_JUDGE_CODES}


def paper_model_name(model: str) -> str:
    """
    Return a model's paper display name from :data:`PAPER_MODEL_NAMES`, ignoring provider prefixes.

    :param model: Model identifier, with or without a provider prefix.
    :return: Display name, or the short id when the model has none.
    """
    return PAPER_MODEL_NAMES.get(_model_short_id(model), _model_short_id(model))


def format_scenario_name(scenario: str | None) -> str:
    """
    Map a scenario type identifier to its display name.

    :param scenario: Scenario type identifier (e.g. "product_promotion"), or
        None.
    :return: Display name from :data:`_SCENARIO_DISPLAY_NAMES` if known,
        otherwise ``scenario`` with underscores replaced and title-cased;
        empty string when ``scenario`` is None.
    """
    if not scenario:
        return ""
    return _SCENARIO_DISPLAY_NAMES.get(scenario, scenario.replace("_", " ").title())


def _titled(base_title: str, scenario: str | None) -> str:
    """
    Prefix a plot title with a scenario display name when provided.

    :param base_title: Title text without the scenario prefix.
    :param scenario: Scenario type identifier, or None to skip prefixing.
    :return: ``"{scenario display name} {base_title}"``, or ``base_title``
        unchanged when ``scenario`` is None.
    """
    display = format_scenario_name(scenario)
    return f"{display} {base_title}" if display else base_title


def _format_model_name(model: str) -> str:
    """
    Map a raw model identifier to its display name.

    :param model: Raw model identifier (may include a provider prefix).
    :return: Display name from :data:`_MODEL_DISPLAY_NAMES` if a pattern
        matches, otherwise the last path segment of ``model``; always
        lowercased.
    """
    key = model.lower()
    for pattern, display in _MODEL_DISPLAY_NAMES.items():
        if pattern in key:
            return display.lower()
    return model.split("/")[-1].lower()


def _efa_sort_key(var: str) -> tuple:
    """
    Compute a sort key for an EFA variable in ``"{judge}::{subtype}"`` format.

    :param var: Variable name in ``"{judge}::{subtype}"`` format.
    :return: Tuple of (deception type order index, formatted judge name)
        used to sort variables by subtype then judge.
    """
    parts = var.split("::", 1)
    judge = parts[0] if parts else var
    subtype = parts[1] if len(parts) > 1 else ""
    type_order = (
        _DECEPTION_TYPE_ORDER.index(subtype)
        if subtype in _DECEPTION_TYPE_ORDER
        else len(_DECEPTION_TYPE_ORDER)
    )
    return (type_order, _format_model_name(judge))


def _efa_var_label(var: str) -> str:
    """
    Build a display label for an EFA variable.

    :param var: Variable name in ``"{judge}::{subtype}"`` format.
    :return: Label formatted as ``"{judge display name} · {subtype}"``.
    """
    parts = var.split("::", 1)
    judge = parts[0] if parts else var
    subtype = parts[1] if len(parts) > 1 else var
    return f"{_format_model_name(judge)} · {subtype}"


def _legend_ncol(n: int) -> int:
    """
    Determine the number of legend columns for ``n`` entries.

    :param n: Number of legend entries.
    :return: ``n`` if ``n`` is 5 or fewer, otherwise ``ceil(n / 2)``.
    """
    import math
    return math.ceil(n / 2) if n > 5 else n


def _legend_bottom(ax: Axes, ncol: int, y_offset: float = -0.4) -> None:
    """
    Draw a horizontal legend centred below the given axes.

    :param ax: Axes to draw the legend on.
    :param ncol: Number of legend columns; skipped entirely when 0.
    :param y_offset: Vertical anchor offset below the axes.
    """
    if ncol == 0:
        return
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        frameon=False,
        fontsize=12,
    )


def _annotate_bar(
    ax: Axes,
    bar,
    color: str,
    err_hi: float = 0.0,
    n: int | None = None,
) -> None:
    """
    Annotate a single bar with its percentage value and, optionally, ``n``.

    :param ax: Axes the bar belongs to.
    :param bar: Bar patch returned by ``ax.bar`` / ``ax.barh``.
    :param color: Text color for the percentage annotation.
    :param err_hi: Height of the upper error bar, added to the bar top when
        positioning the annotation.
    :param n: Sample count to render above the percentage when provided.
    """
    bar_top = bar.get_height() + err_hi
    x_center = bar.get_x() + bar.get_width() / 2
    pct_text = f"{bar.get_height():.1f}%"
    if n is not None:
        ax.text(
            x_center, bar_top + 1.0, f"(n={n})",
            ha="center", va="bottom", fontsize=12, color="black",
        )
        ax.text(
            x_center, bar_top + 3.2, pct_text,
            ha="center", va="bottom", fontsize=12, color=color,
            fontweight="bold",
        )
    else:
        ax.text(
            x_center, bar_top + 0.5, pct_text,
            ha="center", va="bottom", fontsize=12, color=color,
            fontweight="bold",
        )


def _append_n(labels: list[str], n_per_label: list[int] | None) -> list[str]:
    """
    Append ``(n=X)`` sample counts to each label.

    :param labels: Base labels.
    :param n_per_label: Sample count per label; when None, ``labels`` is
        returned unchanged.
    :return: Labels with ``"\\n(n=X)"`` appended, or ``labels`` unchanged.
    """
    if n_per_label is None:
        return labels
    return [f"{lbl}\n(n={n})" for lbl, n in zip(labels, n_per_label)]


def _save_figure(
    fig,
    output_path: str,
    rect: list[float] | None = None,
    apply_tight_layout: bool = True,
    bbox_inches: str | None = "tight",
    dpi: int = 300,
) -> None:
    """
    Finalize and write a figure to disk.

    Creates the output directory if needed, applies tight layout, saves the
    figure as a PNG, and closes it to free memory.

    :param fig: Figure to save.
    :param output_path: Full file path (including filename) for the PNG.
    :param rect: Optional normalized rect passed to ``fig.tight_layout``.
    :param apply_tight_layout: If False, skips the ``fig.tight_layout`` call;
        used when the caller has already applied tight layout before adding
        further artists (e.g. a suptitle) that must not shift it.
    :param bbox_inches: Passed to ``fig.savefig``; None keeps the exact
        figure size instead of cropping to the drawn content.
    :param dpi: Resolution of the saved PNG.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if apply_tight_layout:
        fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def _ci_to_yerr(
    rates: list[float | None],
    cis: list[tuple[float, float] | None],
) -> tuple[list[float], list[float], list[float]]:
    """
    Clean None entries and compute matplotlib error-bar heights in
    percentage points.

    :param rates: Rate per group, in [0, 1]; None entries render as 0.0.
    :param cis: (lower, upper) CI per group; None entries render as
        (0.0, 0.0), producing zero-length error bars.
    :return: Tuple of (cleaned rates, lower yerr in percentage points, upper
        yerr in percentage points).
    """
    rates_clean = [r if r is not None else 0.0 for r in rates]
    cis_clean = [ci if ci is not None else (0.0, 0.0) for ci in cis]
    yerr_low = [
        100 * max(0.0, r - ci[0]) for r, ci in zip(rates_clean, cis_clean)
    ]
    yerr_high = [
        100 * max(0.0, ci[1] - r) for r, ci in zip(rates_clean, cis_clean)
    ]
    return rates_clean, yerr_low, yerr_high


def _ordered_deception_types(type_dicts: list[dict]) -> list[str]:
    """
    Order deception types across groups: canonical types first, then any
    others alphabetically.

    :param type_dicts: List of {type: count} dicts to collect types from.
    :return: Deception types present across ``type_dicts``, in
        :data:`_DECEPTION_TYPE_ORDER` order, followed by unrecognized types
        sorted alphabetically.
    """
    present = {t for g in type_dicts for t in g}
    canonical = [t for t in _DECEPTION_TYPE_ORDER if t in present]
    extra = sorted(present - set(_DECEPTION_TYPE_ORDER))
    return canonical + extra


def save_deception_type_bar(
    by_type: dict,
    n_flagged: int,
    plot_dir: str,
    title: str,
    n_total: int | None = None,
) -> None:
    """
    Horizontal bar chart of aggregated deception type counts.

    :param by_type: Dict mapping deception type name to count (already
        aggregated).
    :param n_flagged: Total flagged claims (for percentage labels).
    :param plot_dir: Directory to write deception_types.png.
    :param title: Chart title.
    :param n_total: Total claims evaluated; appended to title as (n=X) when
        provided.
    """
    labels = _ordered_deception_types([by_type])

    counts = [by_type.get(t, 0) for t in labels]
    colors = [_TYPE_COLORS.get(t, "#000000") for t in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.7)))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_ylabel("Deception Type")
    bar_title = f"{title} (n={n_total})" if n_total is not None else title
    ax.set_title(bar_title, fontweight="bold")

    for bar, count in zip(bars, counts):
        pct = f"{100 * count / n_flagged:.1f}%" if n_flagged > 0 else "—"
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{count} ({pct})",
            va="center",
            fontsize=12
        )

    _legend_bottom(ax, len(labels))
    ax.set_xlim(0, max(counts) * 1.3 if counts else 1)

    _save_figure(fig, os.path.join(plot_dir, "deception_types.png"))


def save_rates_bar(
    labels: list[str],
    flag_rates: list[float],
    flag_cis: list[tuple[float, float]],
    deception_rates: list[float | None],
    deception_cis: list[tuple[float, float] | None],
    plot_dir: str,
    title: str,
    filename: str,
    n_per_label: list[int] | None = None,
    xlabel: str | None = None,
) -> None:
    """
    Side-by-side bar chart of flag rate and deception rate per label with
    CI error bars.

    :param labels: Group labels (subtypes or conditions).
    :param flag_rates: Flag rate per group.
    :param flag_cis: (lower, upper) CI for flag rate per group.
    :param deception_rates: Deception rate per group; None entries are skipped.
    :param deception_cis: (lower, upper) CI for deception rate; None entries
        are skipped.
    :param plot_dir: Directory to write {filename}.
    :param title: Chart title.
    :param filename: Output filename (e.g. rates_by_model.png).
    :param n_per_label: Sample count per group; appended to x-axis labels as
        (n=X) when provided.
    :param xlabel: X-axis label; omitted when None.
    """
    n = len(labels)
    x = np.arange(n)
    width = 0.35

    flag_yerr_low = [
        max(0.0, r - ci[0]) for r, ci in zip(flag_rates, flag_cis)
    ]
    flag_yerr_high = [
        max(0.0, ci[1] - r) for r, ci in zip(flag_rates, flag_cis)
    ]

    has_deception = any(r is not None for r in deception_rates)
    if has_deception:
        dec_rates_clean, dec_yerr_low, dec_yerr_high = _ci_to_yerr(
            deception_rates, deception_cis
        )

    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), 5))

    offset = -width / 2 if has_deception else 0
    flag_bars = ax.bar(
        x + offset, [r * 100 for r in flag_rates], width,
        yerr=[
            [e * 100 for e in flag_yerr_low],
            [e * 100 for e in flag_yerr_high],
        ],
        label="Flag rate", color="#000000", capsize=4,
        error_kw={"elinewidth": 1.2}, zorder=3,
    )

    if has_deception:
        dec_bars = ax.bar(
            x + width / 2, [r * 100 for r in dec_rates_clean], width,
            yerr=[dec_yerr_low, dec_yerr_high],
            label="Deception rate", color="#D55E00", capsize=4,
            error_kw={"elinewidth": 1.2}, zorder=3,
        )

    max_bar_top = 0.0
    for i, bar in enumerate(flag_bars):
        max_bar_top = max(
            max_bar_top, bar.get_height() + flag_yerr_high[i] * 100
        )
        _annotate_bar(ax, bar, "#000000", err_hi=flag_yerr_high[i] * 100)
    if has_deception:
        for i, bar in enumerate(dec_bars):
            max_bar_top = max(
                max_bar_top, bar.get_height() + dec_yerr_high[i]
            )
            if deception_rates[i] is not None:
                _annotate_bar(ax, bar, "#D55E00", err_hi=dec_yerr_high[i])

    ax.set_xticks(x)
    ax.set_xticklabels(
        _append_n(labels, n_per_label), rotation=20, ha="right", fontsize=12
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, max(40.0, max_bar_top + 12.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    n_legend_cols = 2 if has_deception else 1
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=n_legend_cols,
        frameon=False, fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    _save_figure(
        fig, os.path.join(plot_dir, filename), apply_tight_layout=False
    )


def save_model_agreement_heatmap(
    claim_evaluations: list[dict],
    flagging_models: list[str],
    plot_dir: str,
    n_total: int | None = None,
) -> None:
    """
    Heatmap of pairwise model agreement rates on flagging decisions.
    Skipped silently if fewer than 2 flagging models.

    :param claim_evaluations: List of claim evaluation dicts (with claim_id,
        flagging_model, is_flagged).
    :param flagging_models: Ordered list of model names.
    :param plot_dir: Directory to write model_agreement.png.
    :param n_total: Total claims evaluated; appended to title as (n=X) when
        provided.
    """
    if len(flagging_models) < 2:
        return

    claim_flags: dict[str, dict[str, bool]] = {}
    for entry in claim_evaluations:
        cid = entry["claim_id"]
        model = entry["flagging_model"]
        if cid not in claim_flags:
            claim_flags[cid] = {}
        claim_flags[cid][model] = entry["is_flagged"]

    n = len(flagging_models)
    matrix = np.zeros((n, n))
    for i, m1 in enumerate(flagging_models):
        for j, m2 in enumerate(flagging_models):
            if i == j:
                matrix[i][j] = 1.0
                continue
            shared = [cid for cid, flags in claim_flags.items()
                      if m1 in flags and m2 in flags]
            if not shared:
                matrix[i][j] = float("nan")
                continue
            agree = sum(
                1 for cid in shared
                if claim_flags[cid][m1] == claim_flags[cid][m2]
            )
            matrix[i][j] = agree / len(shared)

    short_labels = [_format_model_name(m) for m in flagging_models]
    cell_size = 1.0
    fig_size = max(4.0, n * cell_size + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(matrix, annot=True, fmt=".2f", vmin=0, vmax=1,
                xticklabels=short_labels, yticklabels=short_labels,
                cmap="YlGnBu", ax=ax, linewidths=0.5, square=True,
                annot_kws={"size": max(9, 14 - n)})
    base_title = "Model Agreement Rate on Flagging Decisions"
    heatmap_title = (
        f"{base_title} (n={n_total})" if n_total is not None else base_title
    )
    ax.set_title(heatmap_title, fontweight="bold", pad=16)
    _save_figure(fig, os.path.join(plot_dir, "model_agreement.png"))


def save_vote_distribution_bar(
    vote_distribution: dict,
    n_models: int,
    plot_dir: str,
    title: str = "Claims by Number of Flagging Models",
    n_total: int | None = None,
) -> None:
    """
    Vertical bar chart showing how many unique claims received each vote count.
    Bars above the majority threshold are colored to indicate passing the gate.

    :param vote_distribution: Dict mapping vote count (int) to number of
        claims.
    :param n_models: Total number of flagging models (defines x-axis range and
        threshold).
    :param plot_dir: Directory to write vote_distribution.png.
    :param title: Chart title.
    :param n_total: Total claims; appended to title when provided.
    """
    threshold = n_models / 2
    xs = list(range(n_models + 1))
    ys = [vote_distribution.get(k, 0) for k in xs]

    colors = ["#009E73" if k > threshold else "#56B4E9" for k in xs]

    fig, ax = plt.subplots(figsize=(max(5, n_models * 1.2 + 2), 4))
    bars = ax.bar(xs, ys, color=colors, edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, ys):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )

    ax.axvline(
        x=threshold, color="black", linestyle="--", linewidth=1.2, alpha=0.6
    )
    ylim = ax.get_ylim()
    ax.text(
        threshold + 0.05, ylim[1] * 0.95,
        "majority\nthreshold",
        fontsize=12, color="black", alpha=0.7, va="top",
    )

    legend_handles = [
        Patch(facecolor="#009E73", label="Passed majority gate"),
        Patch(facecolor="#56B4E9", label="Filtered out"),
    ]
    ax.legend(handles=legend_handles, fontsize=12, frameon=False)

    ax.set_xlabel(
        "Number of flagging models that flagged the claim", fontsize=12
    )
    ax.set_ylabel("Number of unique claims", fontsize=12)
    ax.set_xticks(xs)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    base_title = title
    vote_title = (
        f"{base_title} (n={n_total})" if n_total is not None else base_title
    )
    ax.set_title(vote_title, fontweight="bold")
    _save_figure(fig, os.path.join(plot_dir, "vote_distribution.png"))


def save_flag_rate_by_judge(
    group_labels: list[str],
    judge_labels: list[str],
    flag_rates: list[list[float]],
    flag_cis: list[list[tuple[float, float] | None]],
    plot_dir: str,
    title: str,
    filename: str,
    n_per_label: list[int] | None = None,
    xlabel: str | None = None,
) -> None:
    """
    Grouped bar chart of per-judge flag rates across groups (conditions or
    subtypes). Skipped silently when judge_labels is empty.

    :param group_labels: X-axis labels (conditions or subtypes).
    :param judge_labels: Short name per flagging model; defines bar clusters.
    :param flag_rates: flag_rates[judge_idx][group_idx] in [0, 1].
    :param flag_cis: flag_cis[judge_idx][group_idx] as (lo, hi) or None.
    :param plot_dir: Directory to write the output file.
    :param title: Chart title.
    :param filename: Output filename.
    :param n_per_label: Total claims per group; appended to x-axis labels as
        (n=X) when provided.
    :param xlabel: X-axis label; omitted when None.
    """
    if not judge_labels:
        return
    n_groups = len(group_labels)
    n_judges = len(judge_labels)
    x = np.arange(n_groups)
    step = 0.82 / max(n_judges, 1)
    bar_width = step * 0.72

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 1.4), 5))

    max_bar_top = 0.0
    for ji, judge in enumerate(judge_labels):
        rates = flag_rates[ji]
        cis = flag_cis[ji]
        rates_clean, yerr_low, yerr_high = _ci_to_yerr(rates, cis)
        offset = (ji - (n_judges - 1) / 2) * step
        color = _MODEL_COLORS[ji % len(_MODEL_COLORS)]
        bars = ax.bar(
            x + offset,
            [r * 100 for r in rates_clean],
            bar_width,
            yerr=[yerr_low, yerr_high],
            label=_format_model_name(judge),
            color=color,
            capsize=4,
            error_kw={"elinewidth": 1.2},
            zorder=3,
        )
        for bar, err_hi in zip(bars, yerr_high):
            max_bar_top = max(max_bar_top, bar.get_height() + err_hi)
            _annotate_bar(ax, bar, color, err_hi=err_hi)

    tick_labels = list(group_labels)
    if n_per_label is not None:
        tick_labels = _append_n(tick_labels, n_per_label)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel("Flag rate (%)")
    ax.set_ylim(0, max(40.0, max_bar_top + 12.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=_legend_ncol(max(1, n_judges)),
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    _save_figure(
        fig, os.path.join(plot_dir, filename), apply_tight_layout=False
    )


def _format_percent_tick(value: float, _position: int) -> str:
    """
    Format a percent-scale tick value with only the decimals it needs.

    :param value: Tick value already on a 0-100 scale.
    :param _position: Tick index supplied by matplotlib; unused.
    :return: Label such as ``"2%"`` or ``"0.5%"``.
    """
    return f"{value:g}%"


def save_grouped_model_bar(
    group_labels: list[str],
    model_labels: list[str],
    deception_rates: list[list[float | None]],
    deception_cis: list[list[tuple[float, float] | None]],
    plot_dir: str,
    title: str,
    filename: str,
    xlabel: str | None = None,
    n_per_label: list[int] | None = None,
    n_per_model_per_label: list[list[int]] | None = None,
    figsize: tuple[float, float] | None = None,
    y_max: float | None = None,
    annotate_bars: bool = True,
) -> None:
    """
    Grouped bar chart of deception rate per group across models.

    :param group_labels: X-axis labels (e.g. subtype names + "Overall").
    :param model_labels: One label per model (defines bar groups).
    :param deception_rates: deception_rates[model_idx][group_idx]; None renders
        as 0.
    :param deception_cis: deception_cis[model_idx][group_idx]; None skips error
        bars.
    :param plot_dir: Directory to save the plot.
    :param title: Chart title.
    :param filename: Output filename.
    :param xlabel: X-axis label; omitted when None.
    :param n_per_label: Sample count per group; appended to x-axis labels as
        (n=X) when provided.
    :param n_per_model_per_label: Per-model sample counts per group; shown as
        annotations above bars when provided.
    :param figsize: Override figure size; defaults to (max(10, n_groups * 2.2)
        - 1, 5.5).
    :param y_max: Override y-axis upper limit; defaults to max(max_bar_top * 1.15,
        max_bar_top + 2.0), where max_bar_top includes error bars.
    :param annotate_bars: If False, suppresses per-bar percentage and n
        annotations.
    """
    n_groups = len(group_labels)
    n_models = len(model_labels)
    x = np.arange(n_groups)
    step = 0.82 / max(n_models, 1)
    bar_width = step * 0.72

    default_figsize = (max(10, n_groups * 2.2) - 1, 5.5)
    fig, ax = plt.subplots(figsize=figsize or default_figsize)

    max_bar_top = 0.0
    max_bar_height = 0.0
    for mi, model in enumerate(model_labels):
        rates = deception_rates[mi]
        cis = deception_cis[mi]
        rates_clean, yerr_low, yerr_high = _ci_to_yerr(rates, cis)
        offset = (mi - (n_models - 1) / 2) * step
        color = _MODEL_COLORS[mi % len(_MODEL_COLORS)]
        bars = ax.bar(
            x + offset,
            [r * 100 for r in rates_clean],
            bar_width,
            yerr=[yerr_low, yerr_high],
            label=_format_model_name(model),
            color=color,
            capsize=4,
            error_kw={"elinewidth": 1.2},
            zorder=3,
        )
        bar_group = zip(bars, rates_clean, yerr_high)
        for gi, (bar, rate, err_hi) in enumerate(bar_group):
            max_bar_top = max(max_bar_top, bar.get_height() + err_hi)
            max_bar_height = max(max_bar_height, bar.get_height())
            if rates[gi] is None or not annotate_bars:
                continue
            n_val = (
                n_per_model_per_label[mi][gi]
                if n_per_model_per_label is not None else None
            )
            _annotate_bar(ax, bar, color, err_hi=err_hi, n=n_val)

    tick_labels = list(group_labels)
    if n_per_label is not None:
        tick_labels = _append_n(tick_labels, n_per_label)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=20, ha="right", fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel("Deception rate (%)")
    _y_max = y_max if y_max is not None else max(max_bar_top * 1.15, max_bar_top + 2.0)
    ax.set_ylim(0, _y_max)
    ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_percent_tick))
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=_legend_ncol(max(1, n_models)),
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    _save_figure(
        fig, os.path.join(plot_dir, filename), apply_tight_layout=False
    )


def save_column_grouped_model_bar(
    group_labels: list[str],
    legend_labels: list[str],
    deception_rates: list[list[float]],
    deception_cis: list[list[tuple[float, float]]],
    output_path: str,
    title: str,
    xlabel: str,
) -> None:
    """
    Save a grouped bar chart of deception rate per group across models, sized
    for one column of a two-column paper.

    :param group_labels: X-axis group labels.
    :param legend_labels: Legend label per model, used verbatim.
    :param deception_rates: deception_rates[model_idx][group_idx], in [0, 1].
    :param deception_cis: deception_cis[model_idx][group_idx] as (lower, upper).
    :param output_path: Full file path of the PNG.
    :param title: Figure title.
    :param xlabel: X-axis label.
    """
    n_models = len(legend_labels)
    x = np.arange(len(group_labels))
    step = 0.84 / n_models
    with plt.rc_context(_COLUMN_RC):
        fig, ax = plt.subplots(figsize=_COLUMN_FIGSIZE, layout="constrained")
        max_bar_top = 0.0
        for mi, label in enumerate(legend_labels):
            rates, yerr_low, yerr_high = _ci_to_yerr(deception_rates[mi], deception_cis[mi])
            heights = [r * 100 for r in rates]
            ax.bar(
                x + (mi - (n_models - 1) / 2) * step,
                heights,
                step * 0.8,
                yerr=[yerr_low, yerr_high],
                label=label,
                color=_MODEL_COLORS[mi % len(_MODEL_COLORS)],
                capsize=1.5,
                error_kw={"elinewidth": 0.6, "capthick": 0.6},
                zorder=3,
            )
            max_bar_top = max([max_bar_top] + [h + e for h, e in zip(heights, yerr_high)])
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Deception rate (%)")
        ax.set_ylim(0, max_bar_top * 1.08)
        ax.yaxis.set_major_locator(MaxNLocator(nbins="auto", steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_percent_tick))
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
            fontsize=7,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=0.9,
        )
        fig.suptitle(title, fontweight="bold")
        _save_figure(fig, output_path, apply_tight_layout=False, bbox_inches=None)


def save_stacked_deception_type_bar(
    group_labels: list[str],
    by_type_per_group: list[dict],
    n_flagged_per_group: list[int],
    plot_dir: str,
    title: str,
    filename: str = "deception_types.png",
    n_total_per_group: list[int] | None = None,
    xlabel: str | None = None,
) -> None:
    """
    Stacked bar chart of deception type breakdown across multiple groups
    (subtypes or conditions).

    :param group_labels: Group labels.
    :param by_type_per_group: List of {type: count} dicts, one per group.
    :param n_flagged_per_group: Total flagged per group (for normalisation).
    :param plot_dir: Directory to write the output file.
    :param title: Chart title.
    :param filename: Output filename (default: deception_types.png).
    :param n_total_per_group: Total claims per group; appended to x-axis labels
        as (n=X) when provided.
    """
    all_types = _ordered_deception_types(by_type_per_group)

    x = np.arange(len(group_labels))
    fig, ax = plt.subplots(figsize=(max(6, len(group_labels) * 1.4), 5))
    bottoms = np.zeros(len(group_labels))

    for dtype in all_types:
        fractions = []
        for i, g in enumerate(by_type_per_group):
            total = n_flagged_per_group[i]
            frac = g.get(dtype, 0) / total * 100 if total > 0 else 0.0
            fractions.append(frac)
        color = _TYPE_COLORS.get(dtype, "#000000")
        ax.bar(x, fractions, bottom=bottoms, label=dtype, color=color)
        bottoms += np.array(fractions)

    ax.set_xticks(x)
    ax.set_xticklabels(
        _append_n(group_labels, n_total_per_group), rotation=20, ha="right"
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("% of flagged claims")
    ax.set_ylim(0, 110)
    ax.set_title(title, fontweight="bold")
    _legend_bottom(ax, len(all_types))
    _save_figure(fig, os.path.join(plot_dir, filename))


def save_deception_type_by_model_by_condition(
    model_labels: list[str],
    condition_labels: list[str],
    by_type_per_model_per_cond: list[list[dict]],
    n_flagged_per_model_per_cond: list[list[int]],
    plot_dir: str,
    title: str,
    filename: str = "deception_types_by_model_by_condition.png",
    n_total_per_cond: list[int] | None = None,
) -> None:
    """
    Grid of grouped deception type bar charts: one subplot per model,
    conditions on x-axis.

    :param model_labels: Short model name per subplot.
    :param condition_labels: Condition names for x-axis (shared across all
        subplots).
    :param by_type_per_model_per_cond: by_type_per_model_per_cond
        [model_idx][cond_idx] = {type: count}.
    :param n_flagged_per_model_per_cond: n_flagged_per_model_per_cond
        [model_idx][cond_idx] = int.
    :param plot_dir: Directory to write the output file.
    :param title: Figure-level title.
    :param filename: Output filename.
    :param n_total_per_cond: Total claims per condition; appended to x-axis
        labels as (n=X) when provided.
    """
    n_models = len(model_labels)
    flat_type_dicts = [
        g for model_data in by_type_per_model_per_cond for g in model_data
    ]
    all_types = _ordered_deception_types(flat_type_dicts)

    n_conds = len(condition_labels)
    n_types = len(all_types)
    width = 0.8 / max(n_types, 1)
    x = np.arange(n_conds)
    tick_labels = _append_n(condition_labels, n_total_per_cond)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(max(5, n_conds * 1.4) * n_models, 5),
        sharey=True,
        squeeze=False,
    )

    for mi, (model, ax) in enumerate(zip(model_labels, axes[0])):
        max_bar_top = 0.0
        for ti, dtype in enumerate(all_types):
            fractions = []
            for ci in range(n_conds):
                g = by_type_per_model_per_cond[mi][ci]
                total = n_flagged_per_model_per_cond[mi][ci]
                frac = g.get(dtype, 0) / total * 100 if total > 0 else 0.0
                fractions.append(frac)
            offset = (ti - (n_types - 1) / 2) * width
            color = _TYPE_COLORS.get(dtype, "#CC79A7")
            bars = ax.bar(
                x + offset, fractions, width,
                label=dtype if mi == 0 else None, color=color,
            )
            for bar, frac in zip(bars, fractions):
                max_bar_top = max(max_bar_top, bar.get_height())
                if frac > 0.5:
                    _annotate_bar(ax, bar, color)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.set_xlabel("Condition")
        ax.set_title(_format_model_name(model), fontsize=12)
        if mi == 0:
            ax.set_ylabel("% of flagged claims")
        ax.set_ylim(0, max(40.0, max_bar_top + 12.0))

    _, labels = axes[0][0].get_legend_handles_labels()
    _legend_bottom(ax, len(labels))
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    _save_figure(
        fig, os.path.join(plot_dir, filename), apply_tight_layout=False
    )


def save_deception_rate_bar(
    labels: list[str],
    deception_rates: list[float | None],
    deception_cis: list[tuple[float, float] | None],
    plot_dir: str,
    title: str,
    filename: str,
    n_per_label: list[int] | None = None,
    xlabel: str | None = None,
) -> None:
    """
    Bar chart of deception rate per label with CI error bars.

    :param labels: X-axis labels (e.g. condition names).
    :param deception_rates: Deception rate per label; None renders as 0.
    :param deception_cis: (lower, upper) CI per label; None skips error bars.
    :param plot_dir: Directory to write the output file.
    :param title: Chart title.
    :param filename: Output filename.
    :param n_per_label: Sample count per label; appended to x-axis labels as
        (n=X) when provided.
    """
    n = len(labels)
    x = np.arange(n)

    rates_clean, yerr_low, yerr_high = _ci_to_yerr(
        deception_rates, deception_cis
    )

    fig, ax = plt.subplots(figsize=(max(5, n * 1.4), 5))
    bars = ax.bar(
        x,
        [r * 100 for r in rates_clean],
        0.5,
        yerr=[yerr_low, yerr_high],
        color="#D55E00",
        capsize=5,
        error_kw={"elinewidth": 1.2},
    )
    max_bar_top = 0.0
    for i, bar in enumerate(bars):
        max_bar_top = max(max_bar_top, bar.get_height() + yerr_high[i])
        n_val = n_per_label[i] if n_per_label else None
        _annotate_bar(ax, bar, "#D55E00", err_hi=yerr_high[i], n=n_val)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Deception rate (%)")
    ax.set_ylim(0, max(40.0, max_bar_top + 12.0))
    ax.set_title(title, fontweight="bold")
    _save_figure(
        fig, os.path.join(plot_dir, filename), rect=[0, 0.07, 1, 1]
    )


def save_stacked_deception_rate_by_model_bar(
    model_labels: list[str],
    confirmed_by_type_per_model: list[dict],
    n_total_per_model: list[int],
    plot_dir: str,
    title: str | None = None,
    filename: str = "deception_rate_by_model_stacked_types.png",
    deception_rate_per_model: list[float | None] | None = None,
) -> None:
    """
    Stacked bar chart of deception rate per model, broken down by deception
    subtype.

    Uses fractional allocation: a confirmed claim with k primary types
    contributes 1/k to each of them, so segment heights sum to the model's
    overall deception rate. Sized for a half-width panel in a two-column
    paper.

    :param model_labels: Model labels (defines bars, in x-axis order).
    :param confirmed_by_type_per_model: List of {deception type: fractional
        confirmed count} dicts, one per model.
    :param n_total_per_model: Total claims per model (for normalisation).
    :param plot_dir: Directory to write the output file.
    :param title: Optional chart title; omitted by default so the paper
        caption carries it.
    :param filename: Output filename.
    :param deception_rate_per_model: Overall deception rate per model;
        annotated above each bar as a percentage when provided.
    """
    all_types = _ordered_deception_types(confirmed_by_type_per_model)

    with plt.rc_context(_PAPER_RC):
        x = np.arange(len(model_labels))
        fig, ax = plt.subplots(figsize=_STACKED_FIGSIZE)
        bottoms = np.zeros(len(model_labels))

        for dtype in all_types:
            fractions = []
            for i, g in enumerate(confirmed_by_type_per_model):
                total = n_total_per_model[i]
                frac = g.get(dtype, 0) / total * 100 if total > 0 else 0.0
                fractions.append(frac)
            color = _TYPE_COLORS.get(dtype, "#000000")
            ax.bar(x, fractions, bottom=bottoms, label=dtype, color=color,
                   zorder=3)
            bottoms += np.array(fractions)

        if deception_rate_per_model is not None:
            for i, rate in enumerate(deception_rate_per_model):
                if rate is None:
                    continue
                ax.annotate(
                    f"{rate * 100:.1f}%", xy=(x[i], bottoms[i]),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=_PAPER_SMALL_FONTSIZE,
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_format_model_name(m) for m in model_labels],
            rotation=35, ha="right", rotation_mode="anchor",
            fontsize=_PAPER_SMALL_FONTSIZE,
        )
        ax.set_ylabel("Deception rate (%)")
        max_bar_top = float(bottoms.max()) if len(bottoms) else 0.0
        ax.set_ylim(0, max_bar_top * 1.25 if max_bar_top > 0 else 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(all_types),
            frameon=False,
            fontsize=_PAPER_SMALL_FONTSIZE,
            handlelength=1.0,
            handletextpad=0.4,
            columnspacing=0.8,
        )
        if title:
            fig.suptitle(title, fontsize=_PAPER_FONTSIZE, fontweight="bold")
        fig.tight_layout()
        _save_figure(
            fig, os.path.join(plot_dir, filename), apply_tight_layout=False,
            bbox_inches=None,
        )


def save_dstudy_plot(
    dstudy_df,
    plot_dir: str,
    scenario: str,
    target: float = 0.85,
    filename: str | None = None,
) -> None:
    """
    Three-panel line plot of G-coefficient vs n for each varied facet.

    One subplot per facet (n_judges, n_convs, n_subtypes). A horizontal dashed
    line marks the target reliability threshold.

    :param dstudy_df: DataFrame with columns [varied_facet, n, g_coefficient].
    :param plot_dir: Directory to write the output file.
    :param scenario: Scenario label used in the title and default filename.
    :param target: G-coefficient reference threshold.
    :param filename: Output filename; defaults to dstudy_{scenario}.png.
    """
    facet_labels = {
        "n_judges": "Number of Judges",
        "n_convs": "Conversations per Sub-scenario",
        "n_subtypes": "Number of Sub-scenarios",
        "n_models": "Number of Agent Models",
    }
    facets = [f for f in ["n_judges", "n_convs", "n_subtypes", "n_models"]
              if f in dstudy_df["varied_facet"].values]

    fig, axes = plt.subplots(
        1, len(facets), figsize=(5 * len(facets), 4), sharey=True
    )
    if len(facets) == 1:
        axes = [axes]

    for ax, facet in zip(axes, facets):
        sub = dstudy_df[dstudy_df["varied_facet"] == facet].sort_values("n")
        ax.plot(
            sub["n"], sub["g_coefficient"], marker="o", linewidth=2,
            markersize=5, color="#000000",
        )
        ax.axhline(target, color="#D55E00", linestyle="--", linewidth=1.2,
                   label=f"Target (Eρ² = {target})")
        ax.set_xlabel(facet_labels.get(facet, facet), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if ax is axes[0]:
            ax.set_ylabel("Generalizability Coefficient (Eρ²)", fontsize=12)
            ax.legend(fontsize=12, loc="lower right")

    title = f"D-Study: {format_scenario_name(scenario)}"
    fig.suptitle(title, fontsize=12, fontweight="bold")

    out_filename = filename or f"dstudy_{scenario}.png"
    _save_figure(fig, os.path.join(plot_dir, out_filename))


def save_mtmm_summary_bar(
    classified: dict[str, list[float]],
    output_path: str,
    scenario: str | None = None,
) -> None:
    """
    Save a bar chart comparing mean correlations across the three MTMM
    block types.

    Bar height is the mean correlation; error bars show ±1 SD.

    :param classified: Output of
        :func:`~src.analysis.mtmm.classify_mtmm_correlations`.
    :param output_path: Full file path for the saved PNG.
    :param scenario: Scenario type identifier; prefixed to the title when
        provided.
    """
    keys = [
        "monotrait_heteromethod",
        "heterotrait_monomethod",
        "heterotrait_heteromethod",
    ]
    labels = [
        "Monotrait\nHeteromethod",
        "Heterotrait\nMonomethod",
        "Heterotrait\nHeteromethod",
    ]
    means = [
        float(np.mean(classified[k])) if classified.get(k) else 0.0
        for k in keys
    ]
    sds = [
        float(np.std(classified[k])) if classified.get(k) else 0.0
        for k in keys
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(keys))
    colors = ["#000000", "#D55E00", "#009E73"]
    ax.bar(x, means, yerr=sds, capsize=5, color=colors, alpha=0.85, width=0.5,
           error_kw={"elinewidth": 1.2})
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Mean tetrachoric r", fontsize=12)
    ax.set_title(
        _titled("MTMM Block Correlations", scenario),
        fontweight="bold", fontsize=12,
    )
    ax.set_ylim(-1, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    _save_figure(fig, output_path)


def save_efa_loading_bar_chart(
    loadings_df: pd.DataFrame,
    cross_loading_threshold: float,
    output_path: str,
    scenario: str | None = None,
) -> None:
    """
    Save a per-factor horizontal bar chart of factor loadings.

    Each subplot shows one factor. Bars are coloured by deception subtype
    (the trait dimension of each judge::subtype column). Dashed reference lines
    at ±cross_loading_threshold mark the simple-structure boundary.

    :param loadings_df: Rotated loadings DataFrame, shape (n_vars, n_factors),
        with index entries in ``"{judge}::{subtype}"`` format.
    :param cross_loading_threshold: Threshold value drawn as dashed reference
        lines.
    :param output_path: Full file path for the saved PNG.
    :param scenario: Scenario type identifier; prefixed to the title when
        provided.
    """
    sorted_vars = sorted(loadings_df.index.tolist(), key=_efa_sort_key)
    unique_models = list(dict.fromkeys(
        v.split("::", 1)[0] for v in sorted_vars
    ))
    model_color = {
        m: _MODEL_COLORS[i % len(_MODEL_COLORS)]
        for i, m in enumerate(unique_models)
    }

    factor_cols = list(loadings_df.columns)
    n_factors = len(factor_cols)
    fig_width = max(4.5 * n_factors, 6)
    fig, axes = plt.subplots(
        1, n_factors,
        figsize=(fig_width, max(4, len(loadings_df) * 0.3)),
        sharey=True,
    )
    if n_factors == 1:
        axes = [axes]

    for k, ax in enumerate(axes):
        col = factor_cols[k]
        vals = loadings_df.loc[sorted_vars, col]
        colors = [model_color[v.split("::", 1)[0]] for v in vals.index]
        tick_labels = [_efa_var_label(v) for v in vals.index]
        ax.barh(
            tick_labels, vals.values, color=colors, edgecolor="white",
            height=0.6,
        )
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.axvline(cross_loading_threshold, color="#999999", linewidth=0.8,
                   linestyle="--", alpha=0.7,
                   label=f"|λ| = {cross_loading_threshold}")
        ax.axvline(-cross_loading_threshold, color="#999999", linewidth=0.8,
                   linestyle="--", alpha=0.7)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Loading", fontsize=10)
        ax.set_title(col, fontsize=11, fontweight="bold")
        if k == 0:
            ax.legend(fontsize=7)

    handles = [
        mpatches.Patch(color=model_color[m], label=_format_model_name(m))
        for m in unique_models
    ]
    n_legend_cols = min(len(unique_models), 4)
    fig.legend(
        handles=handles, loc="lower center", ncol=n_legend_cols, fontsize=8,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        _titled("Factor Loadings (promax oblique rotation)", scenario)
        + "\nHigh loading = column captures this factor; "
        "Low cross-loading = discriminant validity",
        fontsize=10,
    )

    _save_figure(fig, output_path)


def _grid_column_key(
    var: str,
    judge_order: list[str],
) -> tuple:
    """
    Compute a sort key that groups EFA grid columns by type, then judge.

    :param var: Variable name in ``"{judge}::{subtype}"`` format.
    :param judge_order: Judge identifiers in display order.
    :return: Tuple of (type order index, judge order index, variable name).
    """
    judge, subtype = var.split("::", 1)
    type_idx = (
        _GRID_TYPE_ORDER.index(subtype)
        if subtype in _GRID_TYPE_ORDER else len(_GRID_TYPE_ORDER)
    )
    judge_idx = (
        judge_order.index(judge) if judge in judge_order else len(judge_order)
    )
    return (type_idx, judge_idx, var)


def _draw_loading_panel(
    ax: Axes,
    loadings_df: pd.DataFrame,
    columns: list[str],
    cross_loading_threshold: float,
    judge_labels: dict[str, str],
):
    """
    Draw one transposed factor-loading heatmap panel onto ``ax``.

    Rows are factors and columns are ``"{judge}::{subtype}"`` variables in the
    given order. Columns absent from ``loadings_df`` (dropped as constant
    before the EFA) are shown in grey. A loading is outlined when it reaches
    ``cross_loading_threshold`` and every other loading of that variable stays
    below it.

    :param ax: Axes to draw on.
    :param loadings_df: Rotated loadings, shape (n_vars, n_factors).
    :param columns: Ordered variable names defining the panel's columns.
    :param cross_loading_threshold: Salience cutoff for outlining loadings.
    :param judge_labels: Map from judge identifier to its short tick label.
    :return: The image artist, for building a shared colorbar.
    """
    factors = list(loadings_df.columns)
    values = np.full((len(factors), len(columns)), np.nan)
    for j, var in enumerate(columns):
        if var in loadings_df.index:
            values[:, j] = loadings_df.loc[var, factors].to_numpy(dtype=float)

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1,
        aspect="auto", interpolation="nearest",
    )

    for j in range(len(columns)):
        col = values[:, j]
        if np.all(np.isnan(col)):
            continue
        salient = np.flatnonzero(np.abs(col) >= cross_loading_threshold)
        if len(salient) != 1:
            continue
        i = int(salient[0])
        ax.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="#000000", linewidth=1.0,
        ))
        ax.text(
            j, i, f"{col[i]:.1f}".replace("0.", "."),
            ha="center", va="center", fontsize=_GRID_SMALL_FONTSIZE - 1.5,
            color="white" if abs(col[i]) >= 0.6 else "black",
        )

    types = [var.split("::", 1)[1] for var in columns]
    boundaries = [j for j in range(1, len(types)) if types[j] != types[j - 1]]
    for b in boundaries:
        ax.axvline(b - 0.5, color="white", linewidth=2.0)
    starts = [0] + boundaries
    ends = boundaries + [len(types)]
    for start, end in zip(starts, ends):
        ax.text(
            (start + end - 1) / 2, -0.62, types[start],
            ha="center", va="bottom", fontsize=_GRID_SMALL_FONTSIZE,
        )

    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(
        [f"F{i + 1}" for i in range(len(factors))],
        fontsize=_GRID_SMALL_FONTSIZE,
    )
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [judge_labels.get(var.split("::", 1)[0], _format_model_name(
            var.split("::", 1)[0])) for var in columns],
        rotation=90, fontsize=_GRID_SMALL_FONTSIZE - 1,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save_efa_loading_grid(
    loadings_grid: list[list[pd.DataFrame]],
    row_labels: list[str],
    col_labels: list[str],
    output_path: str,
    cross_loading_threshold: float = 0.40,
    judge_order: list[str] | None = None,
    judge_labels: dict[str, str] | None = None,
    panel_width: float = 2.9,
    dpi: int = 300,
) -> None:
    """
    Save a grid of transposed EFA loading heatmaps sized for a paper figure.

    Each row is a taxonomy and each column a scenario. Within a row, panels
    share the same column set (the union of their variables, grouped by type
    then judge), so a variable dropped as constant in one scenario appears as
    a grey column rather than shifting the layout.

    :param loadings_grid: Rotated loadings DataFrames indexed
        ``[row][column]``, each with ``"{judge}::{subtype}"`` index entries.
    :param row_labels: Taxonomy label per row.
    :param col_labels: Scenario label per column.
    :param output_path: Full file path for the saved PNG.
    :param cross_loading_threshold: Salience cutoff for outlining loadings.
    :param judge_order: Judge identifiers in column order within each type;
        defaults to :func:`paper_judge_order`.
    :param judge_labels: Map from judge identifier to short tick label;
        defaults to :func:`paper_judge_labels`.
    :param panel_width: Width of each panel in inches.
    :param dpi: Resolution of the saved PNG.
    """
    if len(loadings_grid) != len(row_labels):
        raise ValueError("loadings_grid and row_labels must have equal length")
    all_judges = sorted(
        {v.split("::", 1)[0] for row in loadings_grid for df in row
         for v in df.index},
        key=_format_model_name,
    )
    judge_labels = paper_judge_labels(all_judges) if judge_labels is None else judge_labels
    judge_order = judge_order or paper_judge_order(all_judges)

    row_heights = [max(df.shape[1] for df in row) + 2.5 for row in loadings_grid]
    fig_h = sum(row_heights) * 0.16 + 0.4
    fig_w = panel_width * len(col_labels) + 0.5

    with plt.rc_context(_PAPER_RC | {"font.size": _GRID_FONTSIZE}):
        fig, axes = plt.subplots(
            len(loadings_grid), len(col_labels),
            figsize=(fig_w, fig_h), squeeze=False, layout="constrained",
            gridspec_kw={"height_ratios": row_heights},
        )
        image = None
        for r, row in enumerate(loadings_grid):
            columns = sorted(
                {v for df in row for v in df.index},
                key=functools.partial(_grid_column_key, judge_order=judge_order),
            )
            for c, df in enumerate(row):
                image = _draw_loading_panel(
                    axes[r][c], df, columns, cross_loading_threshold,
                    judge_labels,
                )
                if r == 0:
                    axes[r][c].set_title(
                        col_labels[c], fontsize=_GRID_FONTSIZE,
                        fontweight="bold", pad=10,
                    )
            axes[r][0].set_ylabel(
                row_labels[r], fontsize=_GRID_FONTSIZE, fontweight="bold",
            )

        colorbar = fig.colorbar(
            image, ax=axes, location="right", shrink=0.5, aspect=25,
            pad=0.01,
        )
        colorbar.set_label("Loading", fontsize=_GRID_SMALL_FONTSIZE)
        colorbar.ax.tick_params(labelsize=_GRID_SMALL_FONTSIZE)
        _save_figure(fig, output_path, apply_tight_layout=False, dpi=dpi)


def _salient_factor(loadings: np.ndarray, threshold: float) -> int | None:
    """
    Return the index of the single salient factor for one variable.

    :param loadings: Loadings of one variable across factors.
    :param threshold: Salience cutoff applied to absolute loadings.
    :return: Factor index when exactly one loading reaches ``threshold``,
        otherwise None.
    """
    salient = np.flatnonzero(np.abs(loadings) >= threshold)
    return int(salient[0]) if len(salient) == 1 else None


def _draw_membership_panel(
    ax: Axes,
    loadings_df: pd.DataFrame,
    types: list[str],
    judges: list[str],
    cross_loading_threshold: float,
    judge_labels: dict[str, str],
    show_group_headers: bool,
) -> None:
    """
    Draw salient-loading counts by factor for types and for judges.

    The left block counts, for each factor and type, how many judges' columns
    of that type load saliently on the factor; the right block counts, for
    each factor and judge, how many of that judge's type columns do. Shading
    is the count relative to its maximum (judges for types, types for
    judges).

    :param ax: Axes to draw on.
    :param loadings_df: Rotated loadings, shape (n_vars, n_factors).
    :param types: Types in display order.
    :param judges: Judge identifiers in display order.
    :param cross_loading_threshold: Salience cutoff for counting loadings.
    :param judge_labels: Map from judge identifier to short tick label.
    :param show_group_headers: Draw "Types" / "Judges" headers above blocks.
    """
    factors = list(loadings_df.columns)
    type_counts = np.zeros((len(factors), len(types)))
    judge_counts = np.zeros((len(factors), len(judges)))
    for var in loadings_df.index:
        judge, dtype = var.split("::", 1)
        f = _salient_factor(
            loadings_df.loc[var].to_numpy(dtype=float), cross_loading_threshold
        )
        if f is None:
            continue
        if dtype in types:
            type_counts[f, types.index(dtype)] += 1
        if judge in judges:
            judge_counts[f, judges.index(judge)] += 1

    n_types = len(types)
    judge_x0 = n_types + 1
    blocks = [
        (type_counts, 0, len(judges), "Blues"),
        (judge_counts, judge_x0, n_types, "Oranges"),
    ]
    for counts, x0, max_count, cmap in blocks:
        n_cols = counts.shape[1]
        ax.imshow(
            counts / max_count, cmap=cmap, vmin=0, vmax=1.15, aspect="auto",
            interpolation="nearest",
            extent=(x0 - 0.5, x0 + n_cols - 0.5, len(factors) - 0.5, -0.5),
        )
        for i in range(counts.shape[0]):
            for j in range(n_cols):
                if counts[i, j] > 0:
                    ax.text(
                        x0 + j, i, f"{int(counts[i, j])}",
                        ha="center", va="center",
                        fontsize=_GRID_SMALL_FONTSIZE,
                        color="white" if counts[i, j] / max_count > 0.6
                        else "black",
                    )
        for j in range(n_cols + 1):
            ax.axvline(x0 + j - 0.5, color="white", linewidth=0.8)
        for i in range(len(factors) + 1):
            ax.plot(
                [x0 - 0.5, x0 + n_cols - 0.5], [i - 0.5, i - 0.5],
                color="white", linewidth=0.8,
            )

    if show_group_headers:
        ax.text((n_types - 1) / 2, -0.7, "Types", ha="center", va="bottom",
                fontsize=_GRID_SMALL_FONTSIZE, fontstyle="italic")
        ax.text(judge_x0 + (len(judges) - 1) / 2, -0.7, "Judges",
                ha="center", va="bottom", fontsize=_GRID_SMALL_FONTSIZE,
                fontstyle="italic")

    ax.set_xlim(-0.5, judge_x0 + len(judges) - 0.5)
    ax.set_ylim(len(factors) - 0.5, -0.5)
    ax.set_xticks(list(range(n_types)) + [judge_x0 + j for j in range(len(judges))])
    ax.set_xticklabels(
        [_GRID_TYPE_CODES.get(t, t) for t in types]
        + [judge_labels.get(j, _format_model_name(j)) for j in judges],
        fontsize=_GRID_SMALL_FONTSIZE - 0.5,
    )
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(
        [f"F{i + 1}" for i in range(len(factors))],
        fontsize=_GRID_SMALL_FONTSIZE,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_efa_membership_grid(
    loadings_grid: list[list[pd.DataFrame]],
    row_labels: list[str],
    col_labels: list[str],
    output_path: str,
    cross_loading_threshold: float = 0.40,
    judge_order: list[str] | None = None,
    judge_labels: dict[str, str] | None = None,
    panel_width: float = 2.9,
) -> None:
    """
    Save a grid of factor-membership count panels sized for a paper figure.

    Each row is a taxonomy and each column a scenario. Every panel shows how
    salient loadings distribute over factors by type and by judge: a
    type-aligned solution puts each type on its own factor, whereas a
    judge-driven solution groups judges on factors across types.

    :param loadings_grid: Rotated loadings DataFrames indexed
        ``[row][column]``, each with ``"{judge}::{subtype}"`` index entries.
    :param row_labels: Taxonomy label per row.
    :param col_labels: Scenario label per column.
    :param output_path: Full file path for the saved PNG.
    :param cross_loading_threshold: Salience cutoff for counting loadings.
    :param judge_order: Judge identifiers in display order; defaults to
        :func:`paper_judge_order`.
    :param judge_labels: Map from judge identifier to short tick label;
        defaults to :func:`paper_judge_labels`.
    :param panel_width: Width of each panel in inches.
    """
    if len(loadings_grid) != len(row_labels):
        raise ValueError("loadings_grid and row_labels must have equal length")
    all_judges = sorted(
        {v.split("::", 1)[0] for row in loadings_grid for df in row
         for v in df.index},
        key=_format_model_name,
    )
    judge_labels = paper_judge_labels(all_judges) if judge_labels is None else judge_labels
    judges = [j for j in (judge_order or paper_judge_order(all_judges)) if j in all_judges]

    row_heights = [max(df.shape[1] for df in row) + 1.6 for row in loadings_grid]
    fig_h = sum(row_heights) * 0.17 + 0.3
    fig_w = panel_width * len(col_labels) + 0.3

    with plt.rc_context(_PAPER_RC | {"font.size": _GRID_FONTSIZE}):
        fig, axes = plt.subplots(
            len(loadings_grid), len(col_labels),
            figsize=(fig_w, fig_h), squeeze=False, layout="constrained",
            gridspec_kw={"height_ratios": row_heights},
        )
        for r, row in enumerate(loadings_grid):
            present = {v.split("::", 1)[1] for df in row for v in df.index}
            types = [t for t in _GRID_TYPE_ORDER if t in present]
            for c, df in enumerate(row):
                _draw_membership_panel(
                    axes[r][c], df, types, judges, cross_loading_threshold,
                    judge_labels, show_group_headers=r == 0,
                )
                if r == 0:
                    axes[r][c].set_title(
                        col_labels[c], fontsize=_GRID_FONTSIZE,
                        fontweight="bold", pad=12,
                    )
            axes[r][0].set_ylabel(
                row_labels[r], fontsize=_GRID_FONTSIZE, fontweight="bold",
            )
        _save_figure(fig, output_path, apply_tight_layout=False)


def _draw_corr_panel(
    ax: Axes,
    corr_df: pd.DataFrame,
    judge_order: list[str],
    judge_labels: dict[str, str],
):
    """
    Draw one judge-by-type correlation matrix ordered by type, then judge.

    White lines separate type blocks, so diagonal blocks hold same-type
    correlations across judges (monotrait-heteromethod); thin outlines mark
    same-judge, different-type cells (heterotrait-monomethod).

    :param ax: Axes to draw on.
    :param corr_df: Square correlation matrix with ``"{judge}::{subtype}"``
        index and columns.
    :param judge_order: Judge identifiers in display order within each type.
    :param judge_labels: Map from judge identifier to short tick label.
    :return: The image artist, for building a shared colorbar.
    """
    order = sorted(
        corr_df.index.tolist(),
        key=functools.partial(_grid_column_key, judge_order=judge_order),
    )
    values = corr_df.loc[order, order].to_numpy(dtype=float)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(
        np.ma.masked_invalid(values), cmap=cmap, vmin=-1, vmax=1,
        interpolation="nearest",
    )

    judges = [v.split("::", 1)[0] for v in order]
    types = [v.split("::", 1)[1] for v in order]
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j and judges[i] == judges[j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="#000000", linewidth=0.5,
                ))

    boundaries = [k for k in range(1, len(types)) if types[k] != types[k - 1]]
    for b in boundaries:
        ax.axvline(b - 0.5, color="white", linewidth=1.5)
        ax.axhline(b - 0.5, color="white", linewidth=1.5)
    starts = [0] + boundaries
    ends = boundaries + [len(types)]
    for start, end in zip(starts, ends):
        ax.text(
            (start + end - 1) / 2, -0.8, types[start],
            ha="center", va="bottom", fontsize=_GRID_SMALL_FONTSIZE,
        )

    tick_labels = [judge_labels.get(j, _format_model_name(j)) for j in judges]
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(tick_labels, rotation=90,
                       fontsize=_GRID_SMALL_FONTSIZE - 1.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(tick_labels, fontsize=_GRID_SMALL_FONTSIZE - 1.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def save_mtmm_matrix_grid(
    corr_matrices: list[pd.DataFrame],
    col_labels: list[str],
    output_path: str,
    judge_order: list[str] | None = None,
    judge_labels: dict[str, str] | None = None,
    panel_width: float = 3.0,
) -> None:
    """
    Save side-by-side MTMM correlation matrices sized for a paper figure.

    :param corr_matrices: One square correlation matrix per scenario, with
        ``"{judge}::{subtype}"`` index and columns.
    :param col_labels: Scenario label per matrix.
    :param output_path: Full file path for the saved PNG.
    :param judge_order: Judge identifiers in display order within each type;
        defaults to :func:`paper_judge_order`.
    :param judge_labels: Map from judge identifier to short tick label;
        defaults to :func:`paper_judge_labels`.
    :param panel_width: Width of each panel in inches.
    """
    if len(corr_matrices) != len(col_labels):
        raise ValueError("corr_matrices and col_labels must have equal length")
    all_judges = sorted(
        {v.split("::", 1)[0] for df in corr_matrices for v in df.index},
        key=_format_model_name,
    )
    judge_labels = paper_judge_labels(all_judges) if judge_labels is None else judge_labels
    judge_order = judge_order or paper_judge_order(all_judges)

    with plt.rc_context(_PAPER_RC | {"font.size": _GRID_FONTSIZE}):
        fig, axes = plt.subplots(
            1, len(corr_matrices),
            figsize=(panel_width * len(corr_matrices) + 0.4, panel_width),
            squeeze=False, layout="constrained",
        )
        image = None
        for c, df in enumerate(corr_matrices):
            image = _draw_corr_panel(axes[0][c], df, judge_order, judge_labels)
            axes[0][c].set_title(
                col_labels[c], fontsize=_GRID_FONTSIZE, fontweight="bold",
                pad=12,
            )
        colorbar = fig.colorbar(
            image, ax=axes, location="right", shrink=0.6, aspect=25, pad=0.01,
        )
        colorbar.set_label("Tetrachoric $r$", fontsize=_GRID_SMALL_FONTSIZE)
        colorbar.ax.tick_params(labelsize=_GRID_SMALL_FONTSIZE)
        _save_figure(fig, output_path, apply_tight_layout=False)


def _draw_scree_panel(
    ax: Axes,
    observed: list[float],
    simulated: list[float],
    n_retained: int,
    n_show: int,
) -> None:
    """
    Draw one parallel-analysis scree panel.

    :param ax: Axes to draw on.
    :param observed: Observed eigenvalues in descending order.
    :param simulated: Mean eigenvalues from permuted data, by position.
    :param n_retained: Number of factors retained.
    :param n_show: Number of leading eigenvalues to plot.
    """
    n = min(n_show, len(observed))
    x = np.arange(1, n + 1)
    ax.plot(x, observed[:n], marker="o", markersize=2.5, linewidth=1.0,
            color="#000000", label="Observed")
    ax.plot(x, simulated[:n], linestyle="--", linewidth=1.0,
            color="#D55E00", label="Permuted mean")
    ax.axvline(n_retained + 0.5, color="#999999", linewidth=0.8, linestyle=":")
    ax.text(0.97, 0.95, f"k = {n_retained}", transform=ax.transAxes,
            ha="right", va="top", fontsize=_GRID_SMALL_FONTSIZE)
    ax.set_xticks(x[::2])
    ax.tick_params(labelsize=_GRID_SMALL_FONTSIZE - 0.5, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_scree_grid(
    eigen_grid: list[list[tuple[list[float], list[float], int]]],
    row_labels: list[str],
    col_labels: list[str],
    output_path: str,
    n_show: int = 10,
    panel_width: float = 1.2,
    panel_height: float = 1.0,
) -> None:
    """
    Save a grid of parallel-analysis scree panels sized for a paper figure.

    :param eigen_grid: ``(observed, permuted mean, n_retained)`` tuples
        indexed ``[row][column]``.
    :param row_labels: Label per row.
    :param col_labels: Label per column.
    :param output_path: Full file path for the saved PNG.
    :param n_show: Number of leading eigenvalues to plot per panel.
    :param panel_width: Width of each panel in inches.
    :param panel_height: Height of each panel in inches.
    """
    if len(eigen_grid) != len(row_labels):
        raise ValueError("eigen_grid and row_labels must have equal length")
    with plt.rc_context(_PAPER_RC | {"font.size": _GRID_FONTSIZE}):
        fig, axes = plt.subplots(
            len(eigen_grid), len(col_labels),
            figsize=(panel_width * len(col_labels) + 0.3,
                     panel_height * len(eigen_grid) + 0.4),
            squeeze=False, layout="constrained",
        )
        for r, row in enumerate(eigen_grid):
            for c, (observed, simulated, n_retained) in enumerate(row):
                _draw_scree_panel(
                    axes[r][c], observed, simulated, n_retained, n_show
                )
                if r == 0:
                    axes[r][c].set_title(
                        col_labels[c], fontsize=_GRID_FONTSIZE,
                        fontweight="bold",
                    )
                if r == len(eigen_grid) - 1:
                    axes[r][c].set_xlabel(
                        "Factor", fontsize=_GRID_SMALL_FONTSIZE
                    )
            axes[r][0].set_ylabel(
                f"{row_labels[r]}\nEigenvalue", fontsize=_GRID_SMALL_FONTSIZE,
            )
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="outside lower center", ncol=2,
            frameon=False, fontsize=_GRID_SMALL_FONTSIZE,
        )
        _save_figure(fig, output_path, apply_tight_layout=False)


_SCENARIO_MARKERS = {
    "loan_qa": ("#2a78d6", "o"),
    "product_promotion": ("#eb6834", "D"),
}
_PURITY_TEXT_COLOR = "#3a3a38"


def _label_purity_values(
    ax: Axes, purities: list[list[float | None]], rows: list[int]
) -> None:
    """
    Print each taxonomy's purity values beside its markers, the lowest on the
    left and the highest on the right.

    :param ax: Matplotlib axes holding the dot plot.
    :param purities: purities[scenario_idx][taxonomy_idx] as a percentage, or
        None where no factor was retained.
    :param rows: y position of each taxonomy row.
    """
    for t, y in enumerate(rows):
        values = sorted(p[t] for p in purities if p[t] is not None)
        if not values:
            continue
        ax.text(values[-1] + 2.5, y, f"{values[-1]:.0f}", ha="left",
                va="center", fontsize=7, color=_PURITY_TEXT_COLOR)
        if len(values) > 1:
            ax.text(values[0] - 2.5, y, f"{values[0]:.0f}", ha="right",
                    va="center", fontsize=7, color=_PURITY_TEXT_COLOR)


def save_type_purity_dot_plot(
    purities: list[list[float | None]],
    scenarios: list[str],
    taxonomy_labels: list[str],
    output_path: str,
) -> None:
    """
    Save a compact dot plot of EFA type purity per taxonomy, with one marker
    per scenario, sized for one column of a two-column paper.

    :param purities: purities[scenario_idx][taxonomy_idx] as a percentage, or
        None where no factor was retained (not drawn).
    :param scenarios: Scenario type identifier per row of ``purities``.
    :param taxonomy_labels: Label per taxonomy, drawn top to bottom.
    :param output_path: Full file path of the PNG.
    """
    rows = list(range(len(taxonomy_labels)))[::-1]
    with plt.rc_context(_COLUMN_RC):
        fig, ax = plt.subplots(figsize=(3.3, 1.9))
        for s, scenario in enumerate(scenarios):
            color, marker = _SCENARIO_MARKERS.get(
                scenario, (_MODEL_COLORS[s % len(_MODEL_COLORS)], "s")
            )
            points = [(p, y) for p, y in zip(purities[s], rows) if p is not None]
            ax.scatter(
                [p for p, _ in points], [y for _, y in points], s=26,
                color=color, marker=marker,
                label=format_scenario_name(scenario), zorder=3, linewidths=0,
            )
        _label_purity_values(ax, purities, rows)

        ax.set_yticks(rows)
        ax.set_yticklabels(taxonomy_labels)
        ax.set_xlim(0, 105)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Type purity (%)")
        ax.xaxis.grid(True, color="#e6e6e3", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#9a9a96")
            spine.set_linewidth(0.8)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=_PURITY_TEXT_COLOR)
        ax.set_ylim(-0.6, len(taxonomy_labels) - 0.4)
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
            frameon=False, handletextpad=0.3, columnspacing=1.2, fontsize=7.5,
        )
        ax.set_title(
            "Type purity of EFA factors", fontsize=9, fontweight="bold", pad=22
        )
        _save_figure(fig, output_path, apply_tight_layout=False)
