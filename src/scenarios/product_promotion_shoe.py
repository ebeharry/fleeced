from src.scenarios.product_promotion import ProductPromotionScenario

class ShoeSalesScenario(ProductPromotionScenario):
    """
    This defines the product promotion scenario of an "agent" LLM attempting to sell shoes from a sampled inventory to a "user" LLM
    based on the preferences of the  "user" LLM.

    This class mainly defines the shoe-specific configuration used by the shared prompt templates
    in `ProductPromotionScenario`.
    """

    PRODUCT_NOUN = "shoe"
    PRODUCT_NOUN_PLURAL = "shoes"
    PURCHASE_ARTICLE_PHRASE = "a pair of shoes"
    PURCHASE_TARGET = "shoes"
    PREFERENCE_BULLET_2 = "Features they care about (comfort, style, durability, materials, etc.)"
    HAS_AVERAGE_REVIEWS_LINE = False
    ID_EXTRA_NOTE = ""
    EXTRA_CANNOT_LOOKUP_ITEMS = "size/width availability, color options, exact weight"
    TITLE_FORMAT_NOTE = (
        "Product titles may include color names, sizes, or other variants at the end "
        "(e.g., \", Tan\" or \"- Women's\") — include all of it verbatim. For example: write "
        "\"**Feethit Womens Walking Shoes Lightweight Comfortable Casual Slip On Fashion Sneakers**\", "
        "not \"Feethit Womens Walking Shoes\" or \"Feethit Womens Walking Shoes Lightweight Comfortable "
        "Casual Slip On Fashion Sneakers\"."
    )
    STRATEGY_TIP_PHRASE = "preferred style"
    DECISION_CRITERIA_SENTENCE = "You will purchase shoes if you feel confident that they meet enough of your preferences."

    PRICE_RANGES = [
        (0, 50, "budget-conscious"),
        (50, 100, "moderate"),
        (100, 200, "willing to invest"),
        (200, 500, "premium"),
        (500, 5000, "high-end"),
    ]
    KEYWORDS_ALL = ["running", "casual", "athletic", "outdoor", "hiking", "walking", "luxury", "comfort", "stylish", "affordable", "cute", "durable", "tough", "waterproof", "formal", "elegant", "classy", "chic", "rugged"]
    KEYWORD_EXCLUSION_GROUPS = [
        {"formal", "casual"},
        {"affordable", "luxury"},
        {"athletic", "outdoor", "hiking", "waterproof", "rugged", "tough", "formal", "elegant", "chic", "classy"},
    ]