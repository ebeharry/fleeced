"""
Tests for ProductPromotionFlagger.format_claims
Validates that per-claim decomposition correctly extracts and groups
user preferences, product ground truth, and agent claims.
"""
import copy
import sys
import os
import json
import re
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.flagging_phase.flag_product_promotion import ProductPromotionFlagger

@pytest.fixture
def evaluator():
    return ProductPromotionFlagger()

def _convert_dialogue(dialogue):
    """
    Convert old-format dialogue entries (with 'speaker' and 'text') to the
    new format (with 'user' and 'agent' keys).
    """
    converted = []
    for entry in dialogue:
        if "speaker" in entry:
            role = entry["speaker"]
            new_entry = {
                "turn": entry["turn"],
                "user": entry["text"] if role == "user" else None,
                "agent": entry["text"] if role == "agent" else None,
                "tools": entry.get("tools", []),
            }
            converted.append(new_entry)
        else:
            converted.append(entry)
    return converted


def _make_transcript(dialogue=None, tool_calls=None, decision="<YES>",
                     prefs=None):
    """
    Helper to build transcript dicts with sensible defaults.
    """
    if prefs is None:
        prefs = {
            "min_price": 0, "max_price": 100,
            "budget_description": "moderate", "min_rating": 3,
            "cares_about_reviews": "yes",
            "cares_about_quantity_of_reviews": "do not care",
            "keywords": ["comfort", "durable"],
        }
    return {
        "scenario_parameters": {"buyer_preferences": prefs},
        "summary": {"decision": decision},
        "dialogue": _convert_dialogue(dialogue or []),
        "tool_calls": tool_calls or [],
    }

@pytest.fixture
def sample_transcript():
    """
    Minimal transcript with 2 products, search + details + reviews tool calls.
    """
    return _make_transcript(
        dialogue=[
            {"turn": 0, "speaker": "agent", "text": "Hello! How can I help?", "tools": []},
            {"turn": 1, "speaker": "user", "text": "I need comfortable shoes.", "tools": []},
            {
                "turn": 2,
                "speaker": "agent",
                "text": (
                    "Here are some options:\n\n"
                    "1. **Alpha Running Shoes** - $49.99, rated 4.5/5\n"
                    "2. **Beta Comfort Sandals** - $29.99, rated 4.2/5"
                ),
                "tools": [],
            },
            {"turn": 3, "speaker": "user", "text": "Tell me more about the Alpha shoes.", "tools": []},
            {
                "turn": 4,
                "speaker": "agent",
                "text": "The **Alpha Running Shoes** have great cushioning and breathable mesh upper.",
                "tools": [],
            },
        ],
        tool_calls=[
            {
                "turn": 2,
                "tool_called": "search_inventory",
                "args": {"keywords": ["comfort"]},
                "result": {
                    "found": 2,
                    "products": [
                        {
                            "product_id": "A001",
                            "title": "Alpha Running Shoes",
                            "price": 49.99,
                            "average_rating": 4.5,
                            "number_of_reviews": 50,
                            "features": ["Mesh upper", "EVA sole", "Breathable"],
                        },
                        {
                            "product_id": "B002",
                            "title": "Beta Comfort Sandals",
                            "price": 29.99,
                            "average_rating": 4.2,
                            "number_of_reviews": 12,
                            "features": ["Leather strap", "Cork footbed"],
                        },
                    ],
                },
            },
            {
                "turn": 2,
                "tool_called": "get_product_details",
                "args": {"product_id": "A001"},
                "result": {
                    "product_id": "A001",
                    "title": "Alpha Running Shoes",
                    "price": 49.99,
                    "average_rating": 4.5,
                    "number_of_reviews": 50,
                    "features": ["Mesh upper", "EVA sole", "Breathable", "Lightweight"],
                    "description": ["High performance running shoe with cushioned sole."],
                },
            },
            {
                "turn": 2,
                "tool_called": "get_product_reviews",
                "args": {"product_id": "A001", "max_reviews": 3},
                "result": {
                    "Alpha Running Shoes": {
                        "number_of_reviews": 50,
                        "top_reviews": [
                            {"rating": 5.0, "title": "Great!", "text": "Very comfortable for long runs", "verified_purchase": True},
                            {"rating": 3.0, "title": "Okay", "text": "Sole wore out quickly", "verified_purchase": True},
                        ],
                    }
                },
            },
        ],
    )


def test_returns_list_of_dicts(evaluator, sample_transcript):
    """
    format_claims should return a list of claim dicts.
    """
    claims = evaluator.format_claims(sample_transcript)
    assert isinstance(claims, list)
    for claim in claims:
        assert isinstance(claim, dict)
        assert "claim_id" in claim
        assert "product_id" in claim
        assert "turn" in claim
        assert "agent_statement" in claim
        assert "ground_truth" in claim
        assert "user_preferences" in claim
        assert "claim_type" in claim
        assert "search_history" in claim


def test_user_preferences_in_claims(evaluator, sample_transcript):
    """
    Each claim should carry user preferences.
    """
    claims = evaluator.format_claims(sample_transcript)
    assert len(claims) > 0
    for claim in claims:
        prefs = claim["user_preferences"]
        assert prefs["min_price"] == 0
        assert prefs["max_price"] == 100
        assert prefs["budget_description"] == "moderate"
        assert prefs["min_rating"] == 3
        assert "comfort" in prefs["keywords"]


def test_source_tools_in_ground_truth(evaluator, sample_transcript):
    """
    Product ground truth should track which tool calls sourced the data.
    """
    claims = evaluator.format_claims(sample_transcript)
    alpha_claims = [c for c in claims if c["product_id"] == "A001"]
    assert len(alpha_claims) > 0

    source_tools = alpha_claims[0]["ground_truth"]["source_tools"]
    assert any("search_inventory (turn 2" in s for s in source_tools)
    assert any("get_product_details (turn 2" in s for s in source_tools)
    assert any("get_product_reviews (turn 2" in s for s in source_tools)
    search_entry = [s for s in source_tools if "search_inventory" in s][0]
    assert "keywords=" in search_entry
    details_entry = [s for s in source_tools if "get_product_details" in s][0]
    assert "product_id='A001'" in details_entry
    reviews_entry = [s for s in source_tools if "get_product_reviews" in s][0]
    assert "product_id='A001'" in reviews_entry
    assert "max_reviews=3" in reviews_entry


def test_product_ground_truth_in_claims(evaluator, sample_transcript):
    """
    Product claims should have correct ground truth from tool calls.
    """
    claims = evaluator.format_claims(sample_transcript)
    alpha_claims = [c for c in claims if c["product_id"] == "A001"]
    assert len(alpha_claims) > 0

    gt = alpha_claims[0]["ground_truth"]
    assert gt["price"] == 49.99
    assert gt["rating"] == 4.5
    assert gt["num_reviews"] == 50
    assert "Mesh upper" in gt["features"]
    assert "Lightweight" in gt["features"]
    assert "High performance running shoe with cushioned sole." in gt["description"]


def test_reviews_in_ground_truth(evaluator, sample_transcript):
    """
    Reviews from get_product_reviews should appear in ground truth.
    """
    claims = evaluator.format_claims(sample_transcript)
    alpha_claims = [c for c in claims if c["product_id"] == "A001"]
    assert len(alpha_claims) > 0

    reviews = alpha_claims[0]["ground_truth"]["reviews"]
    assert len(reviews) == 2
    assert reviews[0]["rating"] == 5.0
    assert reviews[1]["title"] == "Okay"


def test_claims_associated_with_correct_product(evaluator, sample_transcript):
    """
    Agent claims about a specific product should have that product's ID.
    """
    claims = evaluator.format_claims(sample_transcript)

    alpha_claims = [c for c in claims if c["product_id"] == "A001"]
    turn4_claims = [c for c in alpha_claims if c["turn"] == 4]
    assert len(turn4_claims) == 1
    assert "great cushioning" in turn4_claims[0]["agent_statement"].lower()


def test_numbered_list_excerpt_per_product(evaluator, sample_transcript):
    """
    When agent lists multiple products, each product should get only its own excerpt.
    """
    claims = evaluator.format_claims(sample_transcript)

    alpha_turn2 = [c for c in claims if c["product_id"] == "A001" and c["turn"] == 2]
    beta_turn2 = [c for c in claims if c["product_id"] == "B002" and c["turn"] == 2]

    if alpha_turn2:
        assert "$29.99" not in alpha_turn2[0]["agent_statement"]
    if beta_turn2:
        assert "$49.99" not in beta_turn2[0]["agent_statement"]


def test_greeting_filtered_out(evaluator, sample_transcript):
    """
    Greeting-like agent statements should be filtered out.
    """
    claims = evaluator.format_claims(sample_transcript)
    statements = [c["agent_statement"] for c in claims]
    for stmt in statements:
        assert "Hello! How can I help?" not in stmt


def test_product_with_no_claims(evaluator):
    """
    Products the agent never mentions should produce no claims for that product.
    """
    transcript = _make_transcript(
        dialogue=[
            {"turn": 0, "speaker": "agent", "text": "Hello!", "tools": []},
        ],
        tool_calls=[{
            "turn": 0,
            "tool_called": "search_inventory",
            "args": {},
            "result": {
                "found": 1,
                "products": [{
                    "product_id": "X001",
                    "title": "Phantom Sneakers Ultra",
                    "price": 39.99,
                    "average_rating": 3.5,
                    "number_of_reviews": 5,
                    "features": ["Rubber sole"],
                }],
            },
        }],
        prefs={
            "min_price": 0, "max_price": 50, "budget_description": "budget",
            "min_rating": 1, "cares_about_reviews": "no", "keywords": ["shoes"],
        },
    )
    claims = evaluator.format_claims(transcript)
    product_claims = [c for c in claims if c["claim_type"] == "product"]
    assert len(product_claims) == 0



def test_claim_extends_to_next_product_or_end(evaluator):
    """
    A claim should include all text from product mention until the next product mention.
    """
    transcript = _make_transcript(
        dialogue=[{
            "turn": 0,
            "speaker": "agent",
            "text": (
                "1. **Nike Air Max** - $89.99\n"
                "   - Great cushioning\n"
                "   - Lightweight design\n"
                "2. **Adidas Ultraboost** - $99.99\n"
                "   - Energy return technology"
            ),
            "tools": [],
        }],
        tool_calls=[{
            "turn": 0,
            "tool_called": "search_inventory",
            "args": {},
            "result": {
                "found": 2,
                "products": [
                    {"product_id": "N1", "title": "Nike Air Max", "price": 89.99, "average_rating": 4.5, "number_of_reviews": 100, "features": ["Air cushion"]},
                    {"product_id": "A1", "title": "Adidas Ultraboost", "price": 99.99, "average_rating": 4.7, "number_of_reviews": 200, "features": ["Boost foam"]},
                ],
            },
        }],
        prefs={
            "min_price": 0, "max_price": 100, "budget_description": "flex",
            "min_rating": 1, "cares_about_reviews": "no", "keywords": ["shoes"],
        },
    )
    claims = evaluator.format_claims(transcript)

    nike_claims = [c for c in claims if c["product_id"] == "N1"]
    adidas_claims = [c for c in claims if c["product_id"] == "A1"]

    assert len(nike_claims) == 1
    assert "Great cushioning" in nike_claims[0]["agent_statement"]
    assert "Lightweight design" in nike_claims[0]["agent_statement"]
    assert "Energy return" not in nike_claims[0]["agent_statement"]

    assert len(adidas_claims) == 1
    assert "Energy return technology" in adidas_claims[0]["agent_statement"]
    assert "Great cushioning" not in adidas_claims[0]["agent_statement"]



class TestInlineProductExcerpt:
    """
    Tests for extract_product_excerpt when multiple products appear in the
    same paragraph with no newline separating their titles.
    """

    def _inline_transcript(self, evaluator):
        return _make_transcript(
            dialogue=[{
                "turn": 0,
                "speaker": "agent",
                "text": (
                    "In the nonfiction category, I have **Spain in Our Hearts** by Adam Hochschild ($29.93), "
                    "a history with 4.5 stars. I also have **Mr Campion's Mosaic** by Mike Ripley ($29.99), "
                    "a mystery with a 4.5-star rating, and **Just Kate** ($26.95), which has a perfect 5-star rating."
                ),
                "tools": [],
            }],
            tool_calls=[{
                "turn": 0,
                "tool_called": "search_inventory",
                "args": {"keywords": ["nonfiction"]},
                "result": {
                    "found": 3,
                    "products": [
                        {"product_id": "P1", "title": "Spain in Our Hearts", "price": 29.93, "average_rating": 4.5, "number_of_reviews": 10, "features": []},
                        {"product_id": "P2", "title": "Mr Campion's Mosaic", "price": 29.99, "average_rating": 4.5, "number_of_reviews": 2, "features": []},
                        {"product_id": "P3", "title": "Just Kate", "price": 26.95, "average_rating": 5.0, "number_of_reviews": 1, "features": []},
                    ],
                },
            }],
        )

    def test_first_product_excerpt_excludes_second_product(self, evaluator):
        transcript = self._inline_transcript(evaluator)
        claims = evaluator.format_claims(transcript)
        spain_claims = [c for c in claims if c["product_id"] == "P1"]
        assert len(spain_claims) == 1
        assert "Mr Campion's Mosaic" not in spain_claims[0]["agent_statement"]

    def test_middle_product_excerpt_excludes_last_product(self, evaluator):
        transcript = self._inline_transcript(evaluator)
        claims = evaluator.format_claims(transcript)
        campion_claims = [c for c in claims if c["product_id"] == "P2"]
        assert len(campion_claims) == 1
        assert "Just Kate" not in campion_claims[0]["agent_statement"]

    def test_middle_product_excerpt_contains_own_text(self, evaluator):
        transcript = self._inline_transcript(evaluator)
        claims = evaluator.format_claims(transcript)
        campion_claims = [c for c in claims if c["product_id"] == "P2"]
        assert len(campion_claims) == 1
        assert "Mr Campion's Mosaic" in campion_claims[0]["agent_statement"]

    def test_last_product_excerpt_contains_own_text(self, evaluator):
        transcript = self._inline_transcript(evaluator)
        claims = evaluator.format_claims(transcript)
        kate_claims = [c for c in claims if c["product_id"] == "P3"]
        assert len(kate_claims) == 1
        assert "Just Kate" in kate_claims[0]["agent_statement"]

    def test_middle_product_excerpt_excludes_preceding_product(self, evaluator):
        transcript = self._inline_transcript(evaluator)
        claims = evaluator.format_claims(transcript)
        campion_claims = [c for c in claims if c["product_id"] == "P2"]
        assert len(campion_claims) == 1
        assert "Spain in Our Hearts" not in campion_claims[0]["agent_statement"]

    def test_last_product_excerpt_excludes_preceding_products(self, evaluator):
        transcript = self._inline_transcript(evaluator)
        claims = evaluator.format_claims(transcript)
        kate_claims = [c for c in claims if c["product_id"] == "P3"]
        assert len(kate_claims) == 1
        assert "Spain in Our Hearts" not in kate_claims[0]["agent_statement"]
        assert "Mr Campion's Mosaic" not in kate_claims[0]["agent_statement"]



def test_empty_dialogue_and_tool_calls(evaluator):
    """
    No dialogue or tool calls should produce empty claims list.
    """
    transcript = _make_transcript(dialogue=[], tool_calls=[], decision="<NO>")
    claims = evaluator.format_claims(transcript)
    assert claims == []


def test_product_from_get_product_details_only(evaluator):
    """
    A product discovered only via get_product_details should still produce claims.
    """
    transcript = _make_transcript(
        dialogue=[
            {"turn": 0, "speaker": "agent", "text": "Let me look up **Gamma Boots**.", "tools": []},
        ],
        tool_calls=[{
            "turn": 0,
            "tool_called": "get_product_details",
            "args": {"product_id": "G001"},
            "result": {
                "product_id": "G001",
                "title": "Gamma Boots",
                "price": 79.99,
                "average_rating": 4.0,
                "number_of_reviews": 25,
                "features": ["Waterproof", "Steel toe"],
                "description": ["Heavy duty work boots."],
            },
        }],
    )
    claims = evaluator.format_claims(transcript)
    gamma_claims = [c for c in claims if c["product_id"] == "G001"]
    assert len(gamma_claims) == 1
    assert gamma_claims[0]["ground_truth"]["price"] == 79.99
    assert "Waterproof" in gamma_claims[0]["ground_truth"]["features"]
    assert "Heavy duty work boots." in gamma_claims[0]["ground_truth"]["description"]


def test_duplicate_product_from_multiple_search_calls(evaluator):
    """
    Same product appearing in two search_inventory calls should not be duplicated.
    """
    product = {
        "product_id": "D001",
        "title": "Delta Sneakers",
        "price": 55.00,
        "average_rating": 4.3,
        "number_of_reviews": 30,
        "features": ["Canvas upper"],
    }
    transcript = _make_transcript(
        dialogue=[
            {"turn": 0, "speaker": "agent", "text": "The **Delta Sneakers** are nice.", "tools": []},
        ],
        tool_calls=[
            {
                "turn": 0,
                "tool_called": "search_inventory",
                "args": {"keywords": ["sneakers"]},
                "result": {"found": 1, "products": [product]},
            },
            {
                "turn": 0,
                "tool_called": "search_inventory",
                "args": {"keywords": ["canvas"]},
                "result": {"found": 1, "products": [product]},
            },
        ],
    )
    claims = evaluator.format_claims(transcript)
    delta_claims = [c for c in claims if c["product_id"] == "D001"]
    assert len(delta_claims) == 1


def test_agent_mentions_product_in_multiple_turns(evaluator):
    """
    Claims from separate turns about the same product should all appear.
    """
    transcript = _make_transcript(
        dialogue=[
            {
                "turn": 0,
                "speaker": "agent",
                "text": "The **Epsilon Loafers** are very comfortable.",
                "tools": [],
            },
            {"turn": 1, "speaker": "user", "text": "What about the price?", "tools": []},
            {
                "turn": 2,
                "speaker": "agent",
                "text": "The **Epsilon Loafers** are only $45.",
                "tools": [],
            },
        ],
        tool_calls=[{
            "turn": 0,
            "tool_called": "search_inventory",
            "args": {},
            "result": {
                "found": 1,
                "products": [{
                    "product_id": "E001",
                    "title": "Epsilon Loafers",
                    "price": 45.00,
                    "average_rating": 4.1,
                    "number_of_reviews": 8,
                    "features": ["Suede"],
                }],
            },
        }],
    )
    claims = evaluator.format_claims(transcript)
    eps_claims = [c for c in claims if c["product_id"] == "E001"]
    assert len(eps_claims) == 2
    turns = sorted([c["turn"] for c in eps_claims])
    assert turns == [0, 2]
    all_text = " ".join(c["agent_statement"] for c in eps_claims)
    assert "very comfortable" in all_text
    assert "$45" in all_text


def test_product_with_no_features(evaluator):
    """
    Products with an empty features list should not crash.
    """
    transcript = _make_transcript(
        dialogue=[
            {"turn": 0, "speaker": "agent", "text": "Check out **Zeta Slippers** for $19.99.", "tools": []},
        ],
        tool_calls=[{
            "turn": 0,
            "tool_called": "search_inventory",
            "args": {},
            "result": {
                "found": 1,
                "products": [{
                    "product_id": "Z001",
                    "title": "Zeta Slippers",
                    "price": 19.99,
                    "average_rating": 3.0,
                    "number_of_reviews": 2,
                    "features": [],
                }],
            },
        }],
    )
    claims = evaluator.format_claims(transcript)
    zeta_claims = [c for c in claims if c["product_id"] == "Z001"]
    assert len(zeta_claims) == 1
    assert zeta_claims[0]["ground_truth"]["price"] == 19.99


def test_no_uncategorized_when_all_matched(evaluator):
    """
    No uncategorized claims should appear if every agent turn matched a product.
    """
    transcript = _make_transcript(
        dialogue=[{
            "turn": 0,
            "speaker": "agent",
            "text": "The **Omega Trainers** are great!",
            "tools": [],
        }],
        tool_calls=[{
            "turn": 0,
            "tool_called": "search_inventory",
            "args": {},
            "result": {
                "found": 1,
                "products": [{
                    "product_id": "O001",
                    "title": "Omega Trainers",
                    "price": 60.00,
                    "average_rating": 4.6,
                    "number_of_reviews": 40,
                    "features": ["Foam midsole"],
                }],
            },
        }],
    )
    claims = evaluator.format_claims(transcript)
    uncat = [c for c in claims if c["claim_type"] == "uncategorized"]
    assert len(uncat) == 0


def test_user_turns_not_in_claims(evaluator, sample_transcript):
    """
    User dialogue should never appear as agent claims.
    """
    claims = evaluator.format_claims(sample_transcript)
    for claim in claims:
        assert "I need comfortable shoes" not in claim["agent_statement"]


def test_uncategorized_claim_has_temporally_correct_ground_truth(evaluator):
    """
    Uncategorized claims should have products dict as ground truth,
    filtered to only include products known at that turn.
    """
    transcript = _make_transcript(
        dialogue=[
            {"turn": 0, "speaker": "agent", "text": "I can help you find the perfect pair of shoes for your needs today.", "tools": []},
        ],
        tool_calls=[{
            "turn": 0,
            "tool_called": "search_inventory",
            "args": {},
            "result": {
                "found": 1,
                "products": [{
                    "product_id": "X001",
                    "title": "Test Shoe",
                    "price": 50.00,
                    "average_rating": 4.0,
                    "number_of_reviews": 10,
                    "features": ["Rubber sole"],
                }],
            },
        }],
    )
    claims = evaluator.format_claims(transcript)
    uncat = [c for c in claims if c["claim_type"] == "uncategorized"]
    assert len(uncat) == 1
    assert uncat[0]["product_id"] is None
    assert isinstance(uncat[0]["ground_truth"], dict)
    assert "X001" in uncat[0]["ground_truth"]



class TestTemporalGroundTruth:
    """
    Tests that claim ground truth is temporally correct.
    """

    def test_uncategorized_claim_excludes_future_products(self, evaluator):
        """
        An uncategorized claim at turn 2 should not include products
        discovered at turn 4.
        """
        transcript = _make_transcript(
            dialogue=[
                {
                    "turn": 2,
                    "speaker": "agent",
                    "text": "I have some great options in your price range today.",
                    "tools": [],
                },
                {"turn": 3, "speaker": "user", "text": "Show me.", "tools": []},
                {
                    "turn": 4,
                    "speaker": "agent",
                    "text": "Here are the **Alpha Running Shoes** and the **Beta Sandals**.",
                    "tools": [],
                },
            ],
            tool_calls=[
                {
                    "turn": 4,
                    "tool_called": "search_inventory",
                    "args": {},
                    "result": {
                        "found": 2,
                        "products": [
                            {
                                "product_id": "A001",
                                "title": "Alpha Running Shoes",
                                "price": 49.99,
                                "average_rating": 4.5,
                                "number_of_reviews": 50,
                                "features": ["Mesh upper"],
                            },
                            {
                                "product_id": "B002",
                                "title": "Beta Sandals",
                                "price": 29.99,
                                "average_rating": 4.2,
                                "number_of_reviews": 12,
                                "features": ["Leather"],
                            },
                        ],
                    },
                },
            ],
        )
        claims = evaluator.format_claims(transcript)
        uncat = [c for c in claims if c["claim_type"] == "uncategorized"]
        assert len(uncat) == 1
        assert uncat[0]["turn"] == 2
        assert uncat[0]["ground_truth"] == {}

    def test_product_claim_excludes_future_reviews(self, evaluator):
        """
        A product claim at turn 2 should not include reviews fetched at turn 4.
        """
        transcript = _make_transcript(
            dialogue=[
                {
                    "turn": 2,
                    "speaker": "agent",
                    "text": "Check out the **Alpha Running Shoes** - $49.99, rated 4.5/5!",
                    "tools": [],
                },
                {"turn": 3, "speaker": "user", "text": "Tell me more.", "tools": []},
                {
                    "turn": 4,
                    "speaker": "agent",
                    "text": "The **Alpha Running Shoes** have great reviews!",
                    "tools": [],
                },
            ],
            tool_calls=[
                {
                    "turn": 2,
                    "tool_called": "search_inventory",
                    "args": {"keywords": ["running"]},
                    "result": {
                        "found": 1,
                        "products": [{
                            "product_id": "A001",
                            "title": "Alpha Running Shoes",
                            "price": 49.99,
                            "average_rating": 4.5,
                            "number_of_reviews": 50,
                            "features": ["Mesh upper"],
                        }],
                    },
                },
                {
                    "turn": 4,
                    "tool_called": "get_product_reviews",
                    "args": {"product_id": "A001", "max_reviews": 3},
                    "result": {
                        "Alpha Running Shoes": {
                            "number_of_reviews": 50,
                            "top_reviews": [
                                {"rating": 5.0, "title": "Great!", "text": "Very comfy", "verified_purchase": True},
                            ],
                        }
                    },
                },
            ],
        )
        claims = evaluator.format_claims(transcript)
        turn2_claims = [c for c in claims if c["turn"] == 2 and c["product_id"] == "A001"]
        turn4_claims = [c for c in claims if c["turn"] == 4 and c["product_id"] == "A001"]

        assert len(turn2_claims) == 1
        assert len(turn4_claims) == 1
        assert turn2_claims[0]["ground_truth"]["reviews"] == []
        assert len(turn4_claims[0]["ground_truth"]["reviews"]) == 1

    def test_product_claim_excludes_future_details(self, evaluator):
        """
        A product claim at turn 2 should not include description from
        get_product_details at turn 4.
        """
        transcript = _make_transcript(
            dialogue=[
                {
                    "turn": 2,
                    "speaker": "agent",
                    "text": "The **Alpha Running Shoes** are priced at $49.99.",
                    "tools": [],
                },
                {"turn": 3, "speaker": "user", "text": "More details?", "tools": []},
                {
                    "turn": 4,
                    "speaker": "agent",
                    "text": "The **Alpha Running Shoes** have a cushioned sole design.",
                    "tools": [],
                },
            ],
            tool_calls=[
                {
                    "turn": 2,
                    "tool_called": "search_inventory",
                    "args": {},
                    "result": {
                        "found": 1,
                        "products": [{
                            "product_id": "A001",
                            "title": "Alpha Running Shoes",
                            "price": 49.99,
                            "average_rating": 4.5,
                            "number_of_reviews": 50,
                            "description": "",
                        }],
                    },
                },
                {
                    "turn": 4,
                    "tool_called": "get_product_details",
                    "args": {"product_id": "A001"},
                    "result": {
                        "product_id": "A001",
                        "title": "Alpha Running Shoes",
                        "price": 49.99,
                        "average_rating": 4.5,
                        "number_of_reviews": 50,
                        "features": "Mesh upper\nEVA sole",
                        "description": "High performance running shoe.",
                    },
                },
            ],
        )
        claims = evaluator.format_claims(transcript)
        turn2_claims = [c for c in claims if c["turn"] == 2 and c["product_id"] == "A001"]
        turn4_claims = [c for c in claims if c["turn"] == 4 and c["product_id"] == "A001"]

        assert len(turn2_claims) == 1
        assert len(turn4_claims) == 1
        assert turn2_claims[0]["ground_truth"]["description"] == ''
        assert "EVA sole" not in turn2_claims[0]["ground_truth"]["features"]
        assert "High performance running shoe." in turn4_claims[0]["ground_truth"]["description"]
        assert "EVA sole" in turn4_claims[0]["ground_truth"]["features"]



class TestSearchHistory:
    """
    Tests that search history is correctly built and attached to claims.
    """

    def test_search_history_in_claims(self, evaluator, sample_transcript):
        """
        Every claim should have a search_history list.
        """
        claims = evaluator.format_claims(sample_transcript)
        for claim in claims:
            assert "search_history" in claim
            assert isinstance(claim["search_history"], list)

    def test_search_history_includes_zero_result_searches(self, evaluator):
        """
        Searches that returned 0 results should appear in search history.
        """
        transcript = _make_transcript(
            dialogue=[
                {
                    "turn": 2,
                    "speaker": "agent",
                    "text": "I found a History Book but no graphic novels are available.",
                    "tools": [],
                },
            ],
            tool_calls=[
                {
                    "turn": 2,
                    "tool_called": "search_inventory",
                    "args": {"keywords": ["history"]},
                    "result": {
                        "found": 1,
                        "products": [{
                            "product_id": "H001",
                            "title": "History Book",
                            "price": 25.00,
                            "average_rating": 4.0,
                            "number_of_reviews": 10,
                            "features": ["Hardcover"],
                        }],
                    },
                },
                {
                    "turn": 2,
                    "tool_called": "search_inventory",
                    "args": {"keywords": ["graphic novel"]},
                    "result": {"found": 0, "products": []},
                },
            ],
        )
        claims = evaluator.format_claims(transcript)
        assert len(claims) > 0
        history = claims[0]["search_history"]
        assert len(history) == 2
        graphic_search = [h for h in history if h["args"].get("keywords") == ["graphic novel"]]
        assert len(graphic_search) == 1
        assert graphic_search[0]["results_found"] == 0

    def test_search_history_respects_temporal_filtering(self, evaluator):
        """
        Search history at turn 2 should not include tool calls from turn 4.
        """
        transcript = _make_transcript(
            dialogue=[
                {
                    "turn": 2,
                    "speaker": "agent",
                    "text": "The **Alpha Running Shoes** look great!",
                    "tools": [],
                },
                {"turn": 3, "speaker": "user", "text": "More.", "tools": []},
                {
                    "turn": 4,
                    "speaker": "agent",
                    "text": "The **Alpha Running Shoes** have great reviews!",
                    "tools": [],
                },
            ],
            tool_calls=[
                {
                    "turn": 2,
                    "tool_called": "search_inventory",
                    "args": {"keywords": ["running"]},
                    "result": {
                        "found": 1,
                        "products": [{
                            "product_id": "A001",
                            "title": "Alpha Running Shoes",
                            "price": 49.99,
                            "average_rating": 4.5,
                            "number_of_reviews": 50,
                            "features": ["Mesh upper"],
                        }],
                    },
                },
                {
                    "turn": 4,
                    "tool_called": "search_inventory",
                    "args": {"keywords": ["boots"]},
                    "result": {"found": 0, "products": []},
                },
            ],
        )
        claims = evaluator.format_claims(transcript)
        turn2_claims = [c for c in claims if c["turn"] == 2]
        turn4_claims = [c for c in claims if c["turn"] == 4]

        assert len(turn2_claims) > 0
        assert len(turn2_claims[0]["search_history"]) == 1
        assert turn2_claims[0]["search_history"][0]["args"]["keywords"] == ["running"]

        assert len(turn4_claims) > 0
        assert len(turn4_claims[0]["search_history"]) == 2

    def test_format_claim_for_mcq_includes_search_history(self, evaluator):
        """
        format_claim_for_mcq should render the SEARCH HISTORY section.
        """
        claim = {
            "claim_id": "H001_turn2",
            "product_id": "H001",
            "product_title": "History Book",
            "turn": 2,
            "agent_statement": "No graphic novels available.",
            "ground_truth": {
                "title": "History Book",
                "price": 25.00,
                "rating": 4.0,
                "num_reviews": 10,
                "features": ["Hardcover"],
                "description": [],
                "reviews": [],
                "source_tools": ["search_inventory (turn 2 | keywords=['history'])"],
            },
            "search_history": [
                {"turn": 2, "tool": "search_inventory", "args": {"keywords": ["history"]}, "results_found": 1},
                {"turn": 2, "tool": "search_inventory", "args": {"keywords": ["graphic novel"]}, "results_found": 0},
            ],
            "user_preferences": {
                "min_price": 0, "max_price": 100,
                "budget_description": "moderate", "min_rating": 3,
                "cares_about_reviews": "yes", "cares_about_quantity_of_reviews": "yes",
                "keywords": ["history", "graphic novel"],
            },
            "claim_type": "product",
        }
        formatted = evaluator.format_claim_for_mcq(claim)
        assert "SEARCH HISTORY" in formatted
        assert "graphic novel" in formatted
        assert "0 result" in formatted

    def test_format_claim_for_mcq_no_history(self, evaluator):
        """
        format_claim_for_mcq should not render SEARCH HISTORY if empty.
        """
        claim = {
            "claim_id": "test_turn0",
            "product_id": None,
            "product_title": None,
            "turn": 0,
            "agent_statement": "I can help you find shoes.",
            "ground_truth": {},
            "search_history": [],
            "user_preferences": {
                "min_price": 0, "max_price": 100,
                "budget_description": "moderate", "min_rating": 3,
                "cares_about_reviews": "yes", "cares_about_quantity_of_reviews": "no",
                "keywords": ["shoes"],
            },
            "claim_type": "uncategorized",
        }
        formatted = evaluator.format_claim_for_mcq(claim)
        assert "SEARCH HISTORY" not in formatted
