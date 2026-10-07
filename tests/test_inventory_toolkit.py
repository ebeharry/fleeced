"""
Tests for src/tools/inventory_tools.py (InventoryToolkit)
Covers the execute() method for all 4 tools and get_tools() schema.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tools.inventory_tools import init_inventory, InventoryToolkit

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "test_inventory_20.jsonl")

@pytest.fixture
def toolkit():
    inventory = init_inventory(FIXTURE_PATH)
    return InventoryToolkit(inventory)


class TestGetTools:
    def test_returns_four_tools(self, toolkit):
        tools = toolkit.get_tools()
        assert len(tools) == 4

    def test_tool_names(self, toolkit):
        tools = toolkit.get_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {
            "search_inventory",
            "get_product_details",
            "get_product_reviews",
            "get_inventory_summary",
        }

    def test_tools_have_litellm_format(self, toolkit):
        for tool in toolkit.get_tools():
            assert tool["type"] == "function"
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]



class TestExecuteSearchInventory:
    def test_basic_search(self, toolkit):
        result = toolkit.execute("search_inventory", {})
        assert "found" in result
        assert "products" in result
        assert result["found"] <= 10

    def test_simplified_fields(self, toolkit):
        result = toolkit.execute("search_inventory", {})
        for product in result["products"]:
            assert set(product.keys()) == {
                "product_id", "title", "price",
                "average_rating", "number_of_reviews", "description",
            }

    def test_description_is_string(self, toolkit):
        result = toolkit.execute("search_inventory", {"max_results": 20})
        for product in result["products"]:
            assert isinstance(product["description"], str)

    def test_price_filters_passed_through(self, toolkit):
        result = toolkit.execute("search_inventory", {
            "min_price": 100, "max_price": 140, "max_results": 20,
        })
        for product in result["products"]:
            assert 100 <= product["price"] <= 140

    def test_keyword_filter(self, toolkit):
        result = toolkit.execute("search_inventory", {
            "keywords": ["waterproof"], "max_results": 20,
        })
        assert result["found"] > 0

    def test_sort_by_min_price(self, toolkit):
        result = toolkit.execute("search_inventory", {
            "sort_by": "min_price", "max_results": 20,
        })
        prices = [p["price"] for p in result["products"]]
        assert prices == sorted(prices)

    def test_found_count_matches_products_length(self, toolkit):
        result = toolkit.execute("search_inventory", {"max_results": 5})
        assert result["found"] == len(result["products"])

    def test_number_of_reviews_maps_from_num_total_reviews(self, toolkit):
        """
        Toolkit maps 'num_total_reviews' to 'number_of_reviews'.
        """
        result = toolkit.execute("search_inventory", {"max_results": 1})
        product = result["products"][0]
        pid = product["product_id"]
        original = toolkit.inventory[pid]
        assert product["number_of_reviews"] == original["num_total_reviews"]



class TestExecuteGetProductDetails:
    def test_returns_product(self, toolkit):
        result = toolkit.execute("get_product_details", {"product_id": "T001"})
        assert result["product_id"] == "T001"
        assert result["title"] == "TrailMaster Men's Hiking Boot Waterproof"

    def test_excludes_reviews(self, toolkit):
        result = toolkit.execute("get_product_details", {"product_id": "T001"})
        assert "top_reviews" not in result

    def test_includes_full_features(self, toolkit):
        result = toolkit.execute("get_product_details", {"product_id": "T001"})
        assert result["features"] == toolkit.inventory["T001"]["features"]

    def test_includes_description(self, toolkit):
        result = toolkit.execute("get_product_details", {"product_id": "T001"})
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0

    def test_nonexistent_product_returns_error(self, toolkit):
        result = toolkit.execute("get_product_details", {"product_id": "FAKE"})
        assert "error" in result

    def test_field_mapping(self, toolkit):
        result = toolkit.execute("get_product_details", {"product_id": "T003"})
        expected_keys = {
            "product_id", "title", "price", "average_rating",
            "number_of_reviews", "rating_number", "features", "description",
        }
        assert expected_keys == set(result.keys())



class TestExecuteGetProductReviews:
    def test_returns_reviews(self, toolkit):
        result = toolkit.execute("get_product_reviews", {
            "product_id": "T001", "max_reviews": 3,
        })
        title = list(result.keys())[0]
        assert "number_of_reviews" in result[title]
        assert "top_reviews" in result[title]

    def test_simplified_review_fields(self, toolkit):
        result = toolkit.execute("get_product_reviews", {
            "product_id": "T001", "max_reviews": 3,
        })
        title = list(result.keys())[0]
        for review in result[title]["top_reviews"]:
            assert set(review.keys()) == {
                "rating", "title", "text", "verified_purchase",
            }
            assert "helpful_vote" not in review

    def test_default_max_reviews(self, toolkit):
        result = toolkit.execute("get_product_reviews", {"product_id": "T001"})
        title = list(result.keys())[0]
        assert len(result[title]["top_reviews"]) <= 3

    def test_respects_max_reviews(self, toolkit):
        result = toolkit.execute("get_product_reviews", {
            "product_id": "T001", "max_reviews": 1,
        })
        title = list(result.keys())[0]
        assert len(result[title]["top_reviews"]) == 1

    def test_keyed_by_product_title(self, toolkit):
        result = toolkit.execute("get_product_reviews", {
            "product_id": "T002", "max_reviews": 2,
        })
        assert "CloudStep Women's Running Shoes Lightweight" in result

    def test_number_of_reviews_is_total_count(self, toolkit):
        result = toolkit.execute("get_product_reviews", {
            "product_id": "T002", "max_reviews": 1,
        })
        title = list(result.keys())[0]
        assert result[title]["number_of_reviews"] == 150



class TestExecuteGetInventorySummary:
    def test_returns_summary(self, toolkit):
        result = toolkit.execute("get_inventory_summary", {})
        assert result["total_products"] == 20

    def test_summary_has_price_stats(self, toolkit):
        result = toolkit.execute("get_inventory_summary", {})
        assert "min_price" in result
        assert "max_price" in result
        assert "average_price" in result

    def test_summary_has_rating_stats(self, toolkit):
        result = toolkit.execute("get_inventory_summary", {})
        assert "average_rating" in result



class TestExecuteUnknownTool:
    def test_unknown_tool_returns_error(self, toolkit):
        result = toolkit.execute("nonexistent_tool", {})
        assert "error" in result
        assert "nonexistent_tool" in result["error"]



class TestExecuteStringCoercionExtended:
    def test_search_inventory_string_max_results_respects_limit(self, toolkit):
        result = toolkit.execute("search_inventory", {"max_results": "5"})
        assert result["found"] <= 5
        assert len(result["products"]) <= 5

    def test_search_inventory_json_string_keywords_same_as_list(self, toolkit):
        as_list = toolkit.execute("search_inventory", {
            "keywords": ["waterproof"], "max_results": 20,
        })
        as_string = toolkit.execute("search_inventory", {
            "keywords": '["waterproof"]', "max_results": 20,
        })
        assert as_list["found"] == as_string["found"]
        assert [p["product_id"] for p in as_list["products"]] == [
            p["product_id"] for p in as_string["products"]
        ]

    def test_get_product_reviews_string_max_reviews_respects_limit(self, toolkit):
        result = toolkit.execute("get_product_reviews", {
            "product_id": "T001", "max_reviews": "1",
        })
        title = list(result.keys())[0]
        assert len(result[title]["top_reviews"]) == 1


class TestExecuteRobustness:
    def test_search_inventory_string_price_params_match_numeric(self, toolkit):
        numeric = toolkit.execute("search_inventory", {
            "min_price": 100, "max_price": 140, "max_results": 20,
        })
        string = toolkit.execute("search_inventory", {
            "min_price": "100", "max_price": "140", "max_results": 20,
        })
        assert numeric["found"] == string["found"]
        assert [p["product_id"] for p in numeric["products"]] == [
            p["product_id"] for p in string["products"]
        ]

    def test_search_inventory_string_rating_param_filters_correctly(self, toolkit):
        result = toolkit.execute("search_inventory", {
            "min_rating": "4", "max_results": 20,
        })
        assert result["found"] > 0
        for product in result["products"]:
            assert product["average_rating"] >= 4.0

    def test_get_product_reviews_invalid_id_returns_error_dict(self, toolkit):
        result = toolkit.execute("get_product_reviews", {
            "product_id": "CloudWalkers Comfort Sneakers",
        })
        assert "error" in result
        assert isinstance(result, dict)



class TestGetProductReviewsEmptyReviews:
    def test_empty_reviews_returns_list_not_string(self, toolkit):
        original = toolkit.inventory["T001"]["top_reviews"]
        toolkit.inventory["T001"]["top_reviews"] = []
        try:
            result = toolkit.execute("get_product_reviews", {
                "product_id": "T001", "max_reviews": 3,
            })
            title = list(result.keys())[0]
            assert isinstance(result[title]["top_reviews"], list), (
                "top_reviews must be a list even when no reviews are available"
            )
            assert result[title]["top_reviews"] == []
        finally:
            toolkit.inventory["T001"]["top_reviews"] = original

    def test_empty_reviews_iterable(self, toolkit):
        original = toolkit.inventory["T001"]["top_reviews"]
        toolkit.inventory["T001"]["top_reviews"] = []
        try:
            result = toolkit.execute("get_product_reviews", {
                "product_id": "T001", "max_reviews": 3,
            })
            title = list(result.keys())[0]
            reviews = result[title]["top_reviews"]
            items = list(reviews)
            assert items == []
        finally:
            toolkit.inventory["T001"]["top_reviews"] = original
