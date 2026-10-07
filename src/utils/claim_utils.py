import re
import math


def normalize(s: str) -> str:
    """
    Normalize a string for matching by lowercasing and collapsing
    whitespace/punctuation spacing differences.

    :param s: The string to normalize.
    :return: Normalized string.
    """
    return re.sub(r'\s+', ' ', re.sub(r'\s*([:\-])\s*', r'\1 ', s.lower())).strip()


def present(value) -> bool:
    """
    Return True if value is a non-null, non-NaN, non-empty string.

    :param value: Any value from a claim record.
    :return: True if the value should be displayed.
    """
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip() not in ("", "nan", "Not applicable", "N/A")


def format_agent_statement(claim: dict) -> str:
    """
    Format the agent statement for display in evaluation prompts and logs.

    :param claim: The full claim dict.
    :return: Formatted string with the turn number and quoted statement.
    """
    return f"(Turn {claim['turn']}):\n  \"{claim['agent_statement'].strip()}\""


def format_user_question(claim: dict) -> str | None:
    """
    Format the user question from the same turn as the agent statement.

    :param claim: The full claim dict.
    :return: Formatted string with the turn number and quoted question, or None if absent.
    """
    q = claim.get("user_question")
    if not q or not q.strip():
        return None
    return f"(Turn {claim['turn']}):\n  \"{q.strip()}\""


def format_prior_statements(claim: dict) -> str | None:
    """
    Format prior agent statements from earlier turns in the same conversation.

    :param claim: The full claim dict.
    :return: Formatted string listing prior statements by turn, or None if none exist.
    """
    prior = claim.get("prior_agent_statements", [])
    if not prior:
        return None
    lines = [f"  Turn {p['turn']}: \"{p['statement'].strip()}\"" for p in prior]
    return "\n".join(lines)


def format_primary_type_reasons(claim: dict) -> str:
    """
    Format the primary deception types and their reasons for the option generation prompt.

    :param claim: A flagged claim dict with primary_types and primary_reasons keys.
    :return: Bullet-list string of each primary type and its reason.
    """
    primary_types = claim.get("primary_types", [])
    primary_reasons = claim.get("primary_reasons", {})
    if not primary_types:
        return "Unknown — no reason provided"
    lines = [
        f"- {dtype}: {primary_reasons.get(dtype, 'No reason provided')}"
        for dtype in primary_types
    ]
    return "\n        ".join(lines)
