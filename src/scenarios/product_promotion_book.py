from src.scenarios.product_promotion import ProductPromotionScenario

class BookSalesScenario(ProductPromotionScenario):
    """
    This defines the product promotion scenario of an "agent" LLM attempting to sell books from a sampled inventory to a "user" LLM
    based on the preferences of the  "user" LLM.

    This class mainly defines the book-specific configuration used by the shared prompt templates
    in `ProductPromotionScenario`.
    """

    PRODUCT_NOUN = "book"
    PRODUCT_NOUN_PLURAL = "books"
    PURCHASE_ARTICLE_PHRASE = "a book"
    PURCHASE_TARGET = "book"
    PREFERENCE_BULLET_2 = "Genre"
    HAS_AVERAGE_REVIEWS_LINE = True
    ID_EXTRA_NOTE = ", ISBN"
    EXTRA_CANNOT_LOOKUP_ITEMS = "edition/format availability"
    TITLE_FORMAT_NOTE = (
        "Product titles may include subtitles, edition markers, or other text after a colon or comma "
        "— include all of it verbatim. For example: write \"**How to Stop Worrying and Start Living**\", "
        "not \"How to Stop Worrying\" or \"How to Stop Worrying and Start Living\"."
    )
    STRATEGY_TIP_PHRASE = "genre"
    DECISION_CRITERIA_SENTENCE = "You will purchase a book if you feel confident that it meets enough of your preferences."

    PRICE_RANGES = [
        (0, 15, "budget-conscious"),
        (15, 30, "moderate"),
        (30, 60, "willing to invest"),
        (60, 150, "premium"),
    ]
    KEYWORDS_ALL = ["humor", "fiction", "history", "nonfiction", "poetry", "romance", "educational", "advice", "self-help", "comic", "graphic novel", "biographies", "religion", "science fiction", "memoir", "fantasy", "teen", "young adult", "historical fiction", "philosophy", "mystery", "thriller"]
    KEYWORD_EXCLUSION_GROUPS = [
        {"fiction", "nonfiction", "science fiction", "fantasy", "historical fiction", "biographies", "memoir", "self-help", "advice"},
    ]