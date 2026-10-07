from src.utils.claim_utils import present, format_user_question, format_prior_statements

_MAX_DENIAL_REASONS = 4

_LOAN_OPTIONAL_FIELDS = [
    ("loan_term", "Loan Term", "{v} months", "Not specified in record"),
    ("interest_rate", "Interest Rate", "{v}%", "Not specified in record"),
    ("debt_to_income_ratio", "Debt-to-Income Ratio", "{v}%", None),
    ("total_loan_costs", "Total Loan Costs", "${v}", "Not specified in record"),
    ("property_value", "Property Value", "${v}", None),
]


def format_loan_record(claim: dict) -> str:
    """
    Format the applicant profile ground truth for display in evaluation prompts and logs.
    Only fields that are present and applicable are included.

    :param claim: The full claim dict containing a ``ground_truth`` applicant profile.
    :return: Formatted string with loan details.
    """
    gt = claim.get("ground_truth", {})
    parts = []

    parts.append(f"  Loan ID: {gt.get('loan_id', 'N/A')}")
    parts.append(f"  Action Taken: {gt.get('action_taken', 'N/A')}")

    denial_reasons = [
        gt.get(f"denial_reason-{i}")
        for i in range(1, _MAX_DENIAL_REASONS + 1)
        if present(gt.get(f"denial_reason-{i}"))
    ]
    if denial_reasons:
        parts.append(f"  Denial Reasons: {', '.join(str(r) for r in denial_reasons)}")

    parts.append(f"  Loan Type: {gt.get('loan_type', 'N/A')}")
    parts.append(f"  Loan Purpose: {gt.get('loan_purpose', 'N/A')}")
    parts.append(f"  Loan Amount: ${gt.get('loan_amount', 'N/A')}")

    for key, label, template, default_text in _LOAN_OPTIONAL_FIELDS:
        value = gt.get(key)
        if present(value):
            parts.append(f"  {label}: {template.format(v=value)}")
        elif default_text is not None:
            parts.append(f"  {label}: {default_text}")

    parts.append(f"  Loan Product Type: {gt.get('derived_loan_product_type', 'N/A')}")
    parts.append(f"  HOEPA Status: {gt.get('hoepa_status', 'N/A')}")
    parts.append(f"  Negative Amortization: {gt.get('negative_amortization', 'N/A')}")
    parts.append(f"  Purchaser Type: {gt.get('purchaser_type', 'N/A')}")
    parts.append(f"  Preapproval: {gt.get('preapproval', 'N/A')}")

    return "\n".join(parts)


def format_loan_context(claim: dict) -> str:
    """
    Format the full loan context for assessment MCQ prompts, combining the
    applicant loan record, prior agent statements, and user question.

    :param claim: The full claim dict containing ground_truth, prior_agent_statements,
        and user_question.
    :return: Formatted context string for use in assessment prompts.
    """
    parts = []

    parts.append("APPLICANT LOAN RECORD (ground truth):")
    parts.append(format_loan_record(claim))
    parts.append("")

    prior_str = format_prior_statements(claim)
    if prior_str:
        parts.append("PRIOR AGENT STATEMENTS (earlier turns in this conversation):")
        parts.append(prior_str)
        parts.append("")

    user_q = format_user_question(claim)
    if user_q:
        parts.append(f"USER QUESTION {user_q}")
        parts.append("")

    return "\n".join(parts)
