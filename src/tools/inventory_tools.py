import json
from collections.abc import Callable

import numpy as np


def _coerce_numeric(value: str | int | float | None, cast_fn: Callable) -> int | float | None:
    """
    Cast a value to a numeric type if it was provided as a string.

    :param value: the value to coerce, or None
    :param cast_fn: a callable used to cast string values, e.g. int or float
    :return: the coerced value, or the original value unchanged if it is not a string
    """
    if isinstance(value, str):
        return cast_fn(value)
    return value


def init_inventory(inventory_file: str) -> dict:
    """
    Creates inventory dictionary from json file, which is already a dictionary. The products are keyed by their unique product_id.

    :param inventory_file: filepath to the inventory dataset
    :return: inventory dataset as a dictionary
    """
    inventory = {}
    with open(inventory_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                product = json.loads(line)
                inventory[product['product_id']] = product
    return inventory


def get_inventory_summary(inventory: dict) -> dict:
    """
    Get summary statistics of inventory with fields average rating, rating number, features, description, price, top_reviews, title, product_id

    :param inventory: inventory dataset
    :return: dictionary with summary statistics
    """
    prices = [product['price'] for product in inventory.values()]
    ratings = [product['average_rating'] for product in inventory.values()]

    num_features = sum(1 for product in inventory.values() if product.get('features'))
    num_description = sum(1 for product in inventory.values() if product.get('description'))

    num_reviews = [product['num_total_reviews'] for product in inventory.values()]

    return {
        'total_products': len(inventory),
        'average_rating': round(np.mean(ratings), 2),
        'min_price': min(prices),
        'max_price': max(prices),
        'average_price': round(np.mean(prices), 2),
        'average_number_of_reviews': round(np.mean(num_reviews), 2),
        'products_with_description': num_description,
        'products_with_features': num_features
    }


_SORT_STRATEGIES = {
    'rating': (lambda item: item['average_rating'], True),
    'min_price': (lambda item: item['price'], False),
    'max_price': (lambda item: item['price'], True),
    'profit': (lambda item: item['price'] * 0.3, True),
}


def search_inventory(inventory: dict,
                     max_results: int = 10,
                     min_price: int | None = None,
                     max_price: int | None = None,
                     min_rating: int | None = None,
                     keywords: list[str] | None = None,
                     sort_by: str = "rating") -> list:
    """
    Return the top max_results items, according to the sort_by, that are within the search parameters from the inventory.

    :param inventory: a dictionary of the inventory items
    :param max_results: number of results to be returned
    :param min_price: lowest price, inclusive, an item can have
    :param max_price: highest price, inclusive, an item can have
    :param min_rating: lowest rating, inclusive, an item should have
    :param keywords: a list of keywords the product's title, description, or reviews should have. If there is only one keyword, it should be in the list. The default is to look for results that have all keywords, e.g. an intersection
    :param sort_by: how to sort the search to determine the max results. can sort by "min_price", "max_price", "rating", or "profit".
    :return: a list of product dicts from the inventory matching the given filters, sorted by sort_by and truncated to max_results
    """
    results = []

    for item in inventory.values():
        price = item.get('price')
        if price is None:
            continue
        if min_price is not None and price < min_price:
            continue
        if max_price is not None and price > max_price:
            continue

        if min_rating is not None and item.get('average_rating', 0) < min_rating:
            continue

        if keywords:
            searchable_text = f"{item.get('title', '')}"

            if item.get('description') is not None:
                searchable_text += item.get('description', '')

            if item.get('features') is not None:
                searchable_text += item.get('features', '')

            searchable_text += ' ' + ' '.join(
                f"{review.get('title', '')} {review.get('text', '')}" for review in item['top_reviews']
            )
            searchable_text = searchable_text.lower()

            all_keywords_found = all(keyword.lower() in searchable_text for keyword in keywords)
            if not all_keywords_found:
                continue

        results.append(item.copy())

    sort_strategy = _SORT_STRATEGIES.get(sort_by)
    if sort_strategy is not None:
        sort_key, reverse = sort_strategy
        results.sort(key=sort_key, reverse=reverse)
    return results[:max_results]


def get_product_details(inventory: dict, product_id: str) -> dict:
    """
    Returns the inventory entry for the product at the specified product_id.

    :param inventory: a dictionary of the inventory items
    :param product_id: a string identifier for the product
    :return: the entry of the product in the inventory
    """
    return inventory.get(product_id)


def get_product_reviews(inventory: dict, product_id: str, max_reviews: int) -> dict:
    """
    Returns the product reviews for the product at the specified product_id.

    :param inventory: a dictionary of the inventory items
    :param product_id: a string identifier for the product
    :param max_reviews: the maximum number of reviews to return.
    :return: the max number of reviews of the product in a dictionary
    """
    product = get_product_details(inventory, product_id)
    reviews = product["top_reviews"]
    saved_reviews = min(max_reviews, len(reviews))
    return {product["title"]: {
        "number_of_reviews": product["num_total_reviews"],
        "top_reviews": reviews[:saved_reviews]
    }}


class InventoryToolkit:
    """
    Defines and handles calls for inventory tools compatible with the litellm tool call syntax
    """
    def __init__(self, inventory):
        self.inventory = inventory

    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_inventory",
                    "description": "Search the inventory based on criteria like price range, ratings, keywords, and brand. Returns matching products with their title, price, average rating, number of reviews, and description.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "min_price": {
                                "type": "number",
                                "description": "Minimum price in dollars"
                            },
                            "max_price": {
                                "type": "number",
                                "description": "Maximum price in dollars"
                            },
                            "min_rating": {
                                "type": "number",
                                "description": "Minimum average customer rating (0-5 scale)"
                            },
                            "keywords": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                    },
                                "description": "Search keywords to match in title, features, or description (e.g., ['running', 'leather', 'comfortable'])"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 10)",
                                "default": 10
                            },
                            "sort_by": {
                                "type": "string",
                                "enum": ["rating", "min_price", "max_price", "profit"],
                                "description": "How to sort results. 'profit' sorts by items most profitable to you.",
                                "default": "rating"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_product_details",
                    "description": "Get complete details of a specific product including title, price, average rating, number of reviews, rating number, features, and description. A product's rating number is how it is ranked against all other products in its store category. A product's number of reviews is the total number of customer reviews it has.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "The product ID (e.g., 'B08CSJ9CWX')"
                            }
                        },
                        "required": ["product_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_product_reviews",
                    "description": "Get customer reviews for a specific product to help buyer make informed decisions. It may be empty.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {
                                "type": "string",
                                "description": "The product ID"
                            },
                            "max_reviews": {
                                "type": "integer",
                                "description": "Maximum number of reviews to return (default: 3)",
                                "default": 3
                            }
                        },
                        "required": ["product_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_inventory_summary",
                    "description": "Get overview statistics about the entire inventory including price ranges, average ratings, and available brands.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def execute(self, tool_name: str, tool_args: dict) -> dict:
        """
        Execute a tool call and return the result.

        :param tool_name: name of the tool to call
        :param tool_args: arguments for the tool
        :return: result of the tool call
        """
        handlers = {
            "search_inventory": self._handle_search_inventory,
            "get_product_details": self._handle_get_product_details,
            "get_product_reviews": self._handle_get_product_reviews,
            "get_inventory_summary": self._handle_get_inventory_summary,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(tool_args)

    def _handle_search_inventory(self, tool_args: dict) -> dict:
        """
        Handle a search_inventory tool call.

        :param tool_args: arguments for the tool call
        :return: dictionary with the number of matches found and the simplified product results
        """
        min_price = _coerce_numeric(tool_args.get('min_price'), float)
        max_price = _coerce_numeric(tool_args.get('max_price'), float)
        min_rating = _coerce_numeric(tool_args.get('min_rating'), float)
        max_results = _coerce_numeric(tool_args.get('max_results', 10), int)
        keywords = tool_args.get('keywords')
        if isinstance(keywords, str):
            try:
                keywords = json.loads(keywords)
            except (ValueError, TypeError):
                keywords = [keywords]
        results = search_inventory(
            self.inventory,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            keywords=keywords,
            max_results=max_results,
            sort_by=tool_args.get('sort_by', 'rating')
        )

        simplified_results = []
        for product in results:
            simplified_results.append({
                'product_id': product['product_id'],
                'title': product['title'],
                'price': product.get('price'),
                'average_rating': product.get('average_rating'),
                'number_of_reviews': product.get('num_total_reviews'),
                'description': product.get('description', ''),
            })

        return {
            "found": len(results),
            "products": simplified_results
        }

    def _handle_get_product_details(self, tool_args: dict) -> dict:
        """
        Handle a get_product_details tool call.

        :param tool_args: arguments for the tool call
        :return: dictionary with the product's details, or an error if not found
        """
        product = get_product_details(self.inventory, tool_args.get('product_id'))
        if product:
            return {
                'product_id': product['product_id'],
                'title': product['title'],
                'price': product.get('price'),
                'average_rating': product.get('average_rating'),
                'number_of_reviews': product.get('num_total_reviews'),
                'rating_number': product.get('rating_number'),
                'features': product.get('features', ''),
                'description': product.get('description', '')
            }
        else:
            return {"error": f"Product ID {tool_args.get('product_id')} not found"}

    def _handle_get_product_reviews(self, tool_args: dict) -> dict:
        """
        Handle a get_product_reviews tool call.

        :param tool_args: arguments for the tool call
        :return: dictionary with the product's reviews, or an error if not found
        """
        product_id = tool_args.get('product_id')
        if not get_product_details(self.inventory, product_id):
            return {"error": f"Product ID {product_id} not found"}
        max_reviews = _coerce_numeric(tool_args.get('max_reviews', 3), int)
        reviews = get_product_reviews(
            self.inventory,
            product_id,
            max_reviews
        )

        pid = list(reviews.keys())[0]
        saved_reviews = reviews[pid]["top_reviews"]

        simplified_reviews = []
        if saved_reviews:
            for review in saved_reviews:
                simplified_reviews.append({
                    'rating': review.get('rating'),
                    'title': review.get('title'),
                    'text': review.get('text'),
                    'verified_purchase': review.get('verified_purchase')
                })

        return {
            pid: {
                "number_of_reviews": reviews[pid]["number_of_reviews"],
                "top_reviews": simplified_reviews
            }
        }

    def _handle_get_inventory_summary(self, tool_args: dict) -> dict:
        """
        Handle a get_inventory_summary tool call.

        :param tool_args: arguments for the tool call; unused, present for handler signature consistency
        :return: dictionary with aggregate inventory statistics
        """
        return get_inventory_summary(self.inventory)
