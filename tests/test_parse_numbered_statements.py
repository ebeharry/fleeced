"""
Tests for parsing numbered statements from LLM responses in generate_options.
Reproduces the bug where numbered sub-lists inside a statement cause the regex
to split mid-statement, producing truncated fragments.
"""
import re
import pytest


def parse_statements_current(raw_text):
    """
    Current (buggy) implementation from assess_product_promotion.py line 249.
    """
    statements = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|$)', raw_text, re.DOTALL)
    statements = [s.strip().strip('"') for s in statements]
    return statements


class TestCurrentRegexBug:
    """
    Demonstrates the bug: when LLM output contains numbered sub-lists inside
    a statement, the regex splits on those internal numbers, producing
    truncated fragments instead of 2 complete statements.
    """

    def test_simple_case_works(self):
        raw_text = (
            "1. The WOJWSKI Wedges are priced at $46.99 with a 4.0 rating.\n"
            "2. Presenting the WOJWSKI Wedges, available for $46.99 with a 4.0 star rating."
        )
        statements = parse_statements_current(raw_text)
        assert len(statements) == 2
        assert statements[0].endswith("rating.")
        assert statements[1].endswith("rating.")

    def test_numbered_sublist_in_statement_causes_truncation(self):
        raw_text = (
            "1. The **WOJWSKI Wedges for Women Sandals** are priced at $46.99 and have an "
            "average rating of 4.0 stars. Key features include:\n"
            "1. Rubber sole\n"
            "2. 3.15 inch heel height\n"
            "3. Comfortable platform design.\n\n"
            "2. Presenting the **WOJWSKI Wedges for Women Sandals**, available for $46.99 "
            "with a 4.0/5 star rating from 18 reviews. Notable features are the rubber "
            "sole and 3.15 inch heel. Customers praised them for comfort and style."
        )
        statements = parse_statements_current(raw_text)
        first_two = statements[:2]
        assert first_two[0].endswith("include:"), (
            f"Bug demo: first statement is truncated at sub-list: {first_two[0][:80]}"
        )

    def test_dash_separated_features_works(self):
        raw_text = (
            "1. The **WOJWSKI Wedges** are priced at $46.99 with features:\n"
            "- Rubber sole\n"
            "- 3.15 inch heel height\n"
            "- Comfortable platform design.\n\n"
            "2. The **WOJWSKI Wedges** cost $46.99 and have a 4.0 star rating."
        )
        statements = parse_statements_current(raw_text)
        first_two = statements[:2]
        assert len(first_two) == 2
        assert "Rubber sole" in first_two[0]

    def test_decimal_numbers_in_text(self):
        raw_text = (
            "1. The shoes are rated 4.3 out of 5.0 stars based on 10 reviews and cost $50.00.\n"
            "2. With a 4.3 star rating from 10 reviews, the shoes are priced at $50.00."
        )
        statements = parse_statements_current(raw_text)
        assert len(statements) == 2


class TestFixedParsing:
    """
    Tests for the fixed parsing logic. The fix should handle numbered sub-lists
    inside statements without splitting on them.
    """

    def test_simple_case(self):
        raw_text = (
            "1. The WOJWSKI Wedges are priced at $46.99 with a 4.0 rating.\n"
            "2. Presenting the WOJWSKI Wedges, available for $46.99 with a 4.0 star rating."
        )
        statements = parse_statements_fixed(raw_text)
        assert len(statements) == 2
        assert "WOJWSKI Wedges are priced" in statements[0]
        assert "Presenting the WOJWSKI Wedges" in statements[1]
        assert statements[0].endswith("rating.")
        assert statements[1].endswith("rating.")

    def test_numbered_sublist_in_statement(self):
        raw_text = (
            "1. The **WOJWSKI Wedges for Women Sandals** are priced at $46.99 and have an "
            "average rating of 4.0 stars. Key features include:\n"
            "1. Rubber sole\n"
            "2. 3.15 inch heel height\n"
            "3. Comfortable platform design.\n\n"
            "2. Presenting the **WOJWSKI Wedges for Women Sandals**, available for $46.99 "
            "with a 4.0/5 star rating from 18 reviews. Notable features are the rubber "
            "sole and 3.15 inch heel. Customers praised them for comfort and style."
        )
        statements = parse_statements_fixed(raw_text)
        assert len(statements) == 2
        assert "Rubber sole" in statements[0]
        assert "Comfortable platform design." in statements[0]
        assert "Presenting the **WOJWSKI" in statements[1]
        assert statements[1].endswith("style.")

    def test_dash_separated_features(self):
        raw_text = (
            "1. The **WOJWSKI Wedges** are priced at $46.99 with features:\n"
            "- Rubber sole\n"
            "- 3.15 inch heel height\n"
            "- Comfortable platform design.\n\n"
            "2. The **WOJWSKI Wedges** cost $46.99 and have a 4.0 star rating."
        )
        statements = parse_statements_fixed(raw_text)
        assert len(statements) == 2
        assert "Rubber sole" in statements[0]

    def test_decimal_numbers_in_text(self):
        raw_text = (
            "1. The shoes are rated 4.3 out of 5.0 stars based on 10 reviews and cost $50.00.\n"
            "2. With a 4.3 star rating from 10 reviews, the shoes are priced at $50.00."
        )
        statements = parse_statements_fixed(raw_text)
        assert len(statements) == 2
        assert "$50.00." in statements[0]
        assert "$50.00." in statements[1]

    def test_blank_line_between_statements(self):
        raw_text = (
            "1. First statement about the product.\n\n"
            "2. Second statement about the product."
        )
        statements = parse_statements_fixed(raw_text)
        assert len(statements) == 2

    def test_no_leading_number(self):
        raw_text = (
            "The WOJWSKI Wedges are priced at $46.99.\n\n"
            "Presenting the WOJWSKI Wedges for $46.99."
        )
        statements = parse_statements_fixed(raw_text)
        assert len(statements) == 2

    def test_numbered_sublist_in_both_statements(self):
        raw_text = (
            "1. The sandals cost $46.99. Features:\n"
            "1. Rubber sole\n"
            "2. Platform heel\n\n"
            "2. Priced at $46.99, the sandals offer:\n"
            "1. Durable rubber sole\n"
            "2. Stylish platform heel."
        )
        statements = parse_statements_fixed(raw_text)
        assert len(statements) == 2
        assert "Rubber sole" in statements[0]
        assert "Durable rubber sole" in statements[1]


def parse_statements_fixed(raw_text):
    """
    Fixed parsing that handles numbered sub-lists within statements.
    Finds all candidate positions where '2.' starts a line, then picks
    the split point that produces the most balanced two halves.
    """
    text = raw_text.strip()

    candidates = [m.start() for m in re.finditer(r'\n\s*2\.\s', text)]

    if candidates:
        total_len = len(text)
        best_idx = min(candidates, key=lambda idx: abs(idx - total_len / 2))
        stmt1 = re.sub(r'^\s*1\.\s*', '', text[:best_idx].strip()).strip('"')
        stmt2 = re.sub(r'^\s*2\.\s*', '', text[best_idx:].strip()).strip('"')
        return [stmt1, stmt2]

    parts = re.split(r'\n\s*\n', text, maxsplit=1)
    if len(parts) == 2:
        stmt1 = re.sub(r'^\s*1\.\s*', '', parts[0].strip()).strip('"')
        stmt2 = re.sub(r'^\s*2\.\s*', '', parts[1].strip()).strip('"')
        return [stmt1, stmt2]

    return [text]
