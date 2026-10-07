"""
Tests for src/utils/inventory.py
Covers init_inventory, get_inventory_summary, search_inventory,
get_product_details, and get_product_reviews.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tools.inventory_tools import (
    init_inventory,
    get_inventory_summary,
    search_inventory,
    get_product_details,
    get_product_reviews,
)

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_inventory_20.jsonl")

@pytest.fixture
def inventory():
    return init_inventory(FIXTURE_PATH)



class TestInitInventory:
    def test_loads_all_products(self, inventory):
        assert len(inventory) == 20

    def test_keyed_by_product_id(self, inventory):
        for pid, product in inventory.items():
            assert pid == product["product_id"]

    def test_product_has_required_fields(self, inventory):
        required = {
            "product_id", "title", "average_rating", "rating_number",
            "features", "description", "price", "categories",
            "top_reviews", "num_total_reviews",
        }
        for product in inventory.values():
            assert required.issubset(product.keys()), (
                f"Missing fields in {product['product_id']}: "
                f"{required - product.keys()}"
            )

    def test_first_product(self, inventory):
        p = inventory["T001"]
        assert p["title"] == "TrailMaster Men's Hiking Boot Waterproof"
        assert p["price"] == 89.99
        assert p["average_rating"] == 4.5

    def test_last_product(self, inventory):
        p = inventory["T020"]
        assert p["title"] == "SummitPro Trail Runner All-Terrain"
        assert p["price"] == 134.99



class TestGetInventorySummary:
    def test_total_products(self, inventory):
        summary = get_inventory_summary(inventory)
        assert summary["total_products"] == 20

    def test_price_range(self, inventory):
        summary = get_inventory_summary(inventory)
        assert summary["min_price"] == 15.99
        assert summary["max_price"] == 134.99

    def test_average_rating_is_rounded(self, inventory):
        summary = get_inventory_summary(inventory)
        assert isinstance(summary["average_rating"], float)
        assert summary["average_rating"] == round(summary["average_rating"], 2)

    def test_feature_and_description_counts(self, inventory):
        summary = get_inventory_summary(inventory)
        assert summary["products_with_features"] == 20
        assert summary["products_with_description"] == 20



class TestSearchInventory:
    def test_no_filters_returns_up_to_max_results(self, inventory):
        results = search_inventory(inventory, max_results=5)
        assert len(results) == 5

    def test_no_filters_default_sort_by_rating(self, inventory):
        results = search_inventory(inventory, max_results=20)
        ratings = [r["average_rating"] for r in results]
        assert ratings == sorted(ratings, reverse=True)

    def test_min_price_filter(self, inventory):
        results = search_inventory(inventory, min_price=100, max_results=20)
        for r in results:
            assert r["price"] >= 100

    def test_max_price_filter(self, inventory):
        results = search_inventory(inventory, max_price=30, max_results=20)
        for r in results:
            assert r["price"] <= 30

    def test_price_range_filter(self, inventory):
        results = search_inventory(inventory, min_price=40, max_price=60, max_results=20)
        for r in results:
            assert 40 <= r["price"] <= 60

    def test_min_rating_filter(self, inventory):
        results = search_inventory(inventory, min_rating=4.5, max_results=20)
        for r in results:
            assert r["average_rating"] >= 4.5

    def test_keyword_filter_single(self, inventory):
        results = search_inventory(inventory, keywords=["waterproof"], max_results=20)
        assert len(results) > 0
        for r in results:
            searchable = (
                r["title"].lower()
                + r.get("description", "").lower()
                + r.get("features", "").lower()
                + " ".join(
                    f"{rev.get('title', '')} {rev.get('text', '')}"
                    for rev in r.get("top_reviews", [])
                ).lower()
            )
            assert "waterproof" in searchable

    def test_keyword_filter_multiple_intersection(self, inventory):
        results = search_inventory(inventory, keywords=["leather", "comfort"], max_results=20)
        for r in results:
            searchable = (
                r["title"].lower()
                + r.get("description", "").lower()
                + r.get("features", "").lower()
                + " ".join(
                    f"{rev.get('title', '')} {rev.get('text', '')}"
                    for rev in r.get("top_reviews", [])
                ).lower()
            )
            assert "leather" in searchable
            assert "comfort" in searchable

    def test_keyword_no_match(self, inventory):
        results = search_inventory(inventory, keywords=["xyznonexistent"], max_results=20)
        assert len(results) == 0

    def test_sort_by_min_price(self, inventory):
        results = search_inventory(inventory, sort_by="min_price", max_results=20)
        prices = [r["price"] for r in results]
        assert prices == sorted(prices)

    def test_sort_by_max_price(self, inventory):
        results = search_inventory(inventory, sort_by="max_price", max_results=20)
        prices = [r["price"] for r in results]
        assert prices == sorted(prices, reverse=True)

    def test_sort_by_profit(self, inventory):
        results = search_inventory(inventory, sort_by="profit", max_results=20)
        profits = [r["price"] * 0.3 for r in results]
        assert profits == sorted(profits, reverse=True)

    def test_max_results_limits_output(self, inventory):
        results = search_inventory(inventory, max_results=3)
        assert len(results) == 3

    def test_combined_filters(self, inventory):
        results = search_inventory(
            inventory,
            min_price=20, max_price=80,
            min_rating=4.0,
            keywords=["comfort"],
            max_results=20,
        )
        for r in results:
            assert 20 <= r["price"] <= 80
            assert r["average_rating"] >= 4.0

    def test_results_are_copies(self, inventory):
        """
        Mutations to results should not affect the original inventory.
        """
        results = search_inventory(inventory, max_results=1)
        results[0]["price"] = 0
        original = inventory[results[0]["product_id"]]
        assert original["price"] != 0

    def test_empty_inventory(self):
        results = search_inventory({}, max_results=10)
        assert results == []



class TestGetProductDetails:
    def test_returns_correct_product(self, inventory):
        product = get_product_details(inventory, "T005")
        assert product["title"] == "PowerStride Basketball Shoes High-Top"
        assert product["price"] == 110.00

    def test_nonexistent_product_returns_none(self, inventory):
        result = get_product_details(inventory, "NONEXISTENT")
        assert result is None

    def test_returns_all_fields(self, inventory):
        product = get_product_details(inventory, "T001")
        assert "features" in product
        assert "description" in product
        assert "top_reviews" in product



class TestGetProductReviews:
    def test_returns_correct_structure(self, inventory):
        reviews = get_product_reviews(inventory, "T001", max_reviews=3)
        title = "TrailMaster Men's Hiking Boot Waterproof"
        assert title in reviews
        assert "number_of_reviews" in reviews[title]
        assert "top_reviews" in reviews[title]

    def test_respects_max_reviews(self, inventory):
        reviews = get_product_reviews(inventory, "T001", max_reviews=1)
        title = list(reviews.keys())[0]
        assert len(reviews[title]["top_reviews"]) == 1

    def test_max_reviews_exceeds_available(self, inventory):
        reviews = get_product_reviews(inventory, "T016", max_reviews=10)
        title = list(reviews.keys())[0]
        assert len(reviews[title]["top_reviews"]) == 2

    def test_total_review_count(self, inventory):
        reviews = get_product_reviews(inventory, "T002", max_reviews=3)
        title = list(reviews.keys())[0]
        assert reviews[title]["number_of_reviews"] == 150

    def test_review_fields(self, inventory):
        reviews = get_product_reviews(inventory, "T001", max_reviews=1)
        title = list(reviews.keys())[0]
        review = reviews[title]["top_reviews"][0]
        assert "rating" in review
        assert "title" in review
        assert "text" in review
