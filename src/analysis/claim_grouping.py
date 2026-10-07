RAW_SUBTYPES = ["Falsehood", "Omission", "Equivocation", "Paltering"]

TAXONOMIES = [
    ("4-type", "all"),
    ("Binary", "falsehood_omission"),
    ("IDT", "no_paltering"),
    ("Rogers", "no_equivocation"),
    ("Active/Passive", "active_vs_passive"),
]

CLAIM_MODES = [mode for _, mode in TAXONOMIES]

_MODE_SUBTYPES = {
    "all": RAW_SUBTYPES,
    "falsehood_omission": ["Falsehood", "Omission"],
    "no_paltering": ["Falsehood", "Omission", "Equivocation"],
    "active_vs_passive": ["Active", "Passive"],
    "no_equivocation": ["Falsehood", "Omission", "Paltering"],
}

_MODE_OUTPUT_SUFFIX = {
    "all": "",
    "falsehood_omission": "_falsehood_omission",
    "no_paltering": "_no_paltering",
    "active_vs_passive": "_active_vs_passive",
    "no_equivocation": "_no_equivocation",
}


def mode_subtypes(mode: str) -> list[str]:
    """
    Return the ordered subtype/trait labels used as claim-matrix columns for a mode.

    :param mode: One of :data:`CLAIM_MODES`.
    :return: Ordered list of subtype labels.
    :raises ValueError: If mode is not recognised.
    """
    if mode not in _MODE_SUBTYPES:
        raise ValueError(f"Unknown claim grouping mode: {mode!r}. Expected one of {CLAIM_MODES}")
    return _MODE_SUBTYPES[mode]


def taxonomy_label(mode: str) -> str:
    """
    Return the paper label of the taxonomy a claim grouping mode implements.

    :param mode: One of :data:`CLAIM_MODES`.
    :return: Label such as "4-type" or "Binary".
    """
    for label, taxonomy_mode in TAXONOMIES:
        if taxonomy_mode == mode:
            return label
    raise ValueError(f"Unknown claim grouping mode: {mode!r}. Expected one of {CLAIM_MODES}")


def output_suffix(mode: str) -> str:
    """
    Return the output-directory/filename suffix for a claim grouping mode.

    :param mode: One of :data:`CLAIM_MODES`.
    :return: Empty string for "all", else an underscore-prefixed suffix.
    """
    if mode not in _MODE_OUTPUT_SUFFIX:
        raise ValueError(f"Unknown claim grouping mode: {mode!r}. Expected one of {CLAIM_MODES}")
    return _MODE_OUTPUT_SUFFIX[mode]


def indicators_for_mode(raw_indicators: dict, mode: str) -> dict[str, bool]:
    """
    Project a claim's raw 4-way deception_indicators dict onto a grouping mode.

    - "all": the 4 raw indicators, unchanged.
    - "falsehood_omission": only the Falsehood/Omission indicators; Equivocation
      and Paltering are dropped entirely (not merged into anything).
    - "no_paltering": Falsehood, Omission, and Equivocation unchanged; Paltering
      is dropped entirely.
    - "active_vs_passive": Falsehood and Paltering are merged (logical OR) into
      "Active"; Omission becomes "Passive"; Equivocation is dropped entirely.
    - "no_equivocation": Falsehood, Omission, and Paltering unchanged;
      Equivocation is dropped entirely.

    :param raw_indicators: A claim_evaluations entry's ``deception_indicators`` dict.
    :param mode: One of :data:`CLAIM_MODES`.
    :return: Dict of {subtype_label: bool} restricted/merged for the given mode.
    """
    if mode == "all":
        return {s: bool(raw_indicators.get(s, False)) for s in RAW_SUBTYPES}
    if mode == "falsehood_omission":
        return {
            "Falsehood": bool(raw_indicators.get("Falsehood", False)),
            "Omission": bool(raw_indicators.get("Omission", False)),
        }
    if mode == "no_paltering":
        return {s: bool(raw_indicators.get(s, False)) for s in _MODE_SUBTYPES["no_paltering"]}
    if mode == "active_vs_passive":
        return {
            "Active": bool(raw_indicators.get("Falsehood", False))
            or bool(raw_indicators.get("Paltering", False)),
            "Passive": bool(raw_indicators.get("Omission", False)),
        }
    if mode == "no_equivocation":
        return {s: bool(raw_indicators.get(s, False)) for s in _MODE_SUBTYPES["no_equivocation"]}
    raise ValueError(f"Unknown claim grouping mode: {mode!r}. Expected one of {CLAIM_MODES}")


def is_flagged_for_mode(raw_indicators: dict, mode: str) -> bool:
    """
    Whether a claim counts as flagged under a grouping mode (OR over its traits).

    :param raw_indicators: A claim_evaluations entry's ``deception_indicators`` dict.
    :param mode: One of :data:`CLAIM_MODES`.
    :return: True if any trait under this mode's projection is True.
    """
    return any(indicators_for_mode(raw_indicators, mode).values())
