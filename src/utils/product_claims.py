import re

from src.utils.claim_utils import normalize, format_agent_statement, format_user_question, format_prior_statements


def title_matches_text(title: str, text: str) -> bool:
    """
    Check if a product title appears in text using exact normalized match only.

    :param title: The product title to search for.
    :param text: The agent message text to search in.
    :return: True if the full normalized title appears in the text.
    """
    text_norm = normalize(text)
    title_norm = normalize(title)
    return title_norm in text_norm


def find_title_position(text_lower: str, title: str, search_from: int = 0) -> int:
    """
    Find the position of a product title in text using exact match only.
    Operates on lowercased text so returned positions are valid for string slicing.

    :param text_lower: The lowercased text to search in.
    :param title: The product title to find.
    :param search_from: Start searching from this character index.
    :return: The character index of the match, or -1 if not found.
    """
    title_lower = title.lower()
    return text_lower.find(title_lower, search_from)


def _is_trailing_noise_line(line: str, first_line: str) -> bool:
    """
    Check whether a trailing line of a product excerpt is noise that should
    be trimmed, such as a blank line or an unindented section header, rather
    than genuine indented product detail or a list item.

    :param line: The candidate trailing line to check.
    :param first_line: The first line of the excerpt, used to detect the
        case where the trailing line is just a repeat of the title line.
    :return: True if the line should be trimmed from the end of the excerpt.
    """
    if not line.strip():
        return True
    is_list_item = bool(re.match(r'^\s*(?:[-•]|\*(?!\*)|\d+\.\s)', line))
    return not line[0].isspace() and not is_list_item and line.strip() != first_line.strip()


def extract_product_excerpt(text: str, title: str, all_titles: list[str]) -> str:
    """
    Extract the portion of agent text relevant to a specific product.
    A product's claim runs from the start of the line where it's mentioned
    until the next product title or end of text. Trailing lines that are
    not indented product details (e.g. section headers, blank lines) are
    trimmed from the end of the excerpt.

    :param text: The full agent message text.
    :param title: The product title to extract context for.
    :param all_titles: All product titles mentioned in the message.
    :return: The relevant excerpt string.
    """
    text_lower = text.lower()
    match_pos = find_title_position(text_lower, title)
    if match_pos == -1:
        return text

    line_start = text.rfind("\n", 0, match_pos)
    start = line_start + 1 if line_start != -1 else 0

    end = len(text)
    for other_title in all_titles:
        if other_title == title:
            continue
        other_pos = find_title_position(text_lower, other_title, search_from=match_pos + 1)
        if other_pos != -1:
            other_line_start = text.rfind("\n", 0, other_pos)
            other_start = other_line_start + 1 if other_line_start != -1 else other_pos
            if other_start < end and other_start > start:
                end = other_start
            elif other_start == start and other_pos < end:
                end = other_pos

    for other_title in all_titles:
        if other_title == title:
            continue
        preceding_pos = find_title_position(text_lower, other_title)
        if preceding_pos != -1 and start <= preceding_pos < match_pos:
            start = match_pos
            break

    excerpt_lines = text[start:end].rstrip().split("\n")
    while excerpt_lines and _is_trailing_noise_line(excerpt_lines[-1], excerpt_lines[0]):
        excerpt_lines.pop()

    return "\n".join(excerpt_lines).strip()


def get_tool_call_product_ids(json_transcript, turn: int) -> set[str]:
    """
    Get all product_ids referenced in tool calls at a specific turn,
    from both tool arguments and search results.

    :param json_transcript: The full conversation transcript dict.
    :param turn: The turn number to extract product IDs from.
    :return: Set of product_id strings.
    """
    pids = set()
    for tool_call in json_transcript.get("tool_calls", []):
        if tool_call.get("turn") != turn:
            continue
        if "product_id" in tool_call.get("args", {}):
            pids.add(tool_call["args"]["product_id"])
        if tool_call["tool_called"] == "search_inventory":
            for p in tool_call.get("result", {}).get("products", []):
                pids.add(p["product_id"])
    return pids


def build_products_from_tool_calls(json_transcript, max_turn: int | None = None) -> dict:
    """
    Build per-product ground truth from tool_calls in the transcript.

    :param json_transcript: The full conversation transcript dict.
    :param max_turn: If provided, only include tool calls from turns <= max_turn.
        This ensures temporal correctness so that a claim's ground truth only
        reflects what the agent knew at the time.
    :return: Dict mapping product_id to product ground truth data.
    """
    products = {}
    for tool_call in json_transcript.get("tool_calls", []):
        if max_turn is not None and tool_call.get("turn", 0) > max_turn:
            continue

        tool_name = tool_call["tool_called"]

        turn = tool_call.get("turn", 0)
        args = tool_call.get("args", {})
        args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        source_label = (
            f"{tool_name} (turn {turn} | {args_str})" if args_str
            else f"{tool_name} (turn {turn})"
        )

        if tool_call.get("result") is None:
            continue

        if tool_name == "search_inventory":
            for p in tool_call["result"].get("products", []):
                pid = p["product_id"]
                if pid not in products:
                    products[pid] = {
                        "title": p["title"],
                        "price": p["price"],
                        "rating": p["average_rating"],
                        "num_reviews": p["number_of_reviews"],
                        "features": '',
                        "description": p.get("description", ''),
                        "reviews": [],
                        "source_tools": [],
                    }
                    products[pid]["source_tools"].append(source_label)

        elif tool_name == "get_product_details":
            r = tool_call["result"]
            if "title" not in r:
                continue
            pid = args["product_id"]
            if pid not in products:
                products[pid] = {
                    "title": r["title"],
                    "price": r["price"],
                    "rating": r["average_rating"],
                    "num_reviews": r["number_of_reviews"],
                    "features": r.get("features", ''),
                    "description": '',
                    "reviews": [],
                    "source_tools": [],
                }
            products[pid]["features"] = r.get("features", products[pid]["features"])
            products[pid]["description"] = r.get("description", '')
            products[pid]["source_tools"].append(source_label)

        elif tool_name == "get_product_reviews":
            pid = tool_call["args"].get("product_id", "")
            for _, reviews_data in tool_call["result"].items():
                if pid in products and isinstance(reviews_data, dict):
                    top_reviews = reviews_data.get("top_reviews", [])
                    if isinstance(top_reviews, list):
                        products[pid]["reviews"] = top_reviews
            if pid in products:
                products[pid]["source_tools"].append(source_label)
    return products


def build_search_history(json_transcript, max_turn: int | None = None) -> list[dict]:
    """
    Build a chronological summary of all search_inventory calls in the
    transcript. Includes searches that returned zero results, which are
    invisible to build_products_from_tool_calls but essential for
    verifying negative claims (e.g., "no graphic novels available").
    Other tool calls (get_product_details, get_product_reviews) are
    captured in the per-product Source field instead.

    :param json_transcript: The full conversation transcript dict.
    :param max_turn: If provided, only include tool calls from turns <= max_turn.
    :return: List of dicts summarizing each search_inventory call with
        turn, tool name, arguments, and result count.
    """
    history = []
    for tool_call in json_transcript.get("tool_calls", []):
        if max_turn is not None and tool_call.get("turn", 0) > max_turn:
            continue

        tool_name = tool_call["tool_called"]
        if tool_name != "search_inventory":
            continue

        turn = tool_call.get("turn", 0)
        args = tool_call.get("args", {})
        result = tool_call.get("result", {})

        entry = {
            "turn": turn,
            "tool": tool_name,
            "args": args,
            "results_found": result.get("found", 0),
        }

        history.append(entry)

    return history


def _format_search_entry(entry: dict) -> str | None:
    """
    Format a single search history entry as a display line.

    :param entry: A search history entry dict with turn, tool, args, and
        results_found.
    :return: Formatted line string, or None if the entry is not a
        search_inventory call.
    """
    if entry["tool"] != "search_inventory":
        return None
    turn_num = entry["turn"]
    args = entry.get("args", {})
    keywords = args.get("keywords", [])
    param_parts = []
    if keywords:
        param_parts.append(f"keywords={keywords}")
    else:
        param_parts.append("no keywords")
    min_price = args.get("min_price", "none")
    max_price = args.get("max_price", "none")
    param_parts.append(f"price ${min_price}-${max_price}")
    param_parts.append(f"min_rating={args.get('min_rating', 'none')}")
    param_parts.append(f"max_results={args.get('max_results', 10)}")
    param_parts.append(f"sort_by={args.get('sort_by', 'rating')}")
    count = entry.get("results_found", 0)
    return f"  Turn {turn_num}: search_inventory({', '.join(param_parts)}) -> {count} result(s)"


def _format_single_product_ground_truth(gt: dict) -> list[str]:
    """
    Format the ground truth fields for a single product as display lines.

    :param gt: The ground truth dict for a single-product claim.
    :return: List of formatted lines for source, price, rating, number of
        reviews, features, description, and individual reviews.
    """
    lines = [
        f"  Source (tool calls that returned this product): {', '.join(gt.get('source_tools', []))}",
        f"  Price: ${gt.get('price', 'N/A')}",
        f"  Rating: {gt.get('rating', 'N/A')}/5",
        f"  Number of Reviews: {gt.get('num_reviews', 'N/A')}",
    ]
    if gt.get("features"):
        lines.append(f"  Features: {gt['features']}")
    if gt.get("description"):
        lines.append(f"  Description: {gt['description']}")
    if gt.get("reviews"):
        lines.append("  REVIEWS:")
        for j, rev in enumerate(gt["reviews"]):
            verified = "Verified Purchase" if rev.get('verified_purchase') else "Not Verified"
            lines.append(f"    Review {j + 1}: Rating: {rev['rating']}/5 | {verified} | {rev['title']} | \"{rev['text']}\"")
    return lines


def _format_multi_product_ground_truth(ground_truth: dict, indent: str = "") -> list[str]:
    """
    Format the ground truth fields for multiple products as display lines.

    :param ground_truth: Dict mapping product_id to product ground truth data.
    :param indent: Extra indentation prefix applied to every line, used to
        match the differing indentation conventions of each caller.
    :return: List of formatted lines, a PRODUCT header and a summary line
        per product.
    """
    lines = []
    for pid, pdata in ground_truth.items():
        lines.append(f"{indent}PRODUCT: {pdata['title']} (ID: {pid})")
        lines.append(f"{indent}  Source (tool calls that returned this product): {', '.join(pdata.get('source_tools', []))}")
        lines.append(f"{indent}  Price: ${pdata['price']} | Rating: {pdata['rating']}/5 | Reviews: {pdata['num_reviews']}")
    return lines


def format_preferences(prefs: dict) -> str:
    """
    Format user preferences for display in evaluation prompts and logs.

    :param prefs: User preferences dict from the claim.
    :return: Formatted string with budget, rating, review, and keyword info.
    """
    if not prefs:
        return ""
    lines = [
        f"Budget: ${prefs['min_price']}-${prefs['max_price']} ({prefs['budget_description']})",
        f"Min Rating: {prefs['min_rating']}",
        f"Cares About Reviews: {prefs['cares_about_reviews']}",
        f"Cares About Number of Reviews: {prefs['cares_about_quantity_of_reviews']}",
        f"Keywords: {', '.join(prefs['keywords'])}",
    ]
    return "\n".join(f"  {line}" for line in lines)


def format_product_details(claim: dict) -> str:
    """
    Format product ground truth data for display in flagging evaluation prompts
    and logs. Renders source tools, price, rating, reviews, features, and
    description for the specific product(s) in the claim.

    :param claim: The full claim dict.
    :return: Formatted string with product details.
    """
    parts = []
    if claim["claim_type"] == "product":
        parts.extend(_format_single_product_ground_truth(claim.get("ground_truth", {})))
    else:
        parts.extend(_format_multi_product_ground_truth(claim.get("ground_truth", {}), indent="  "))
    return "\n".join(parts)


def format_search_history(claim: dict) -> str:
    """
    Format the search history summary for display in evaluation prompts and logs.

    :param claim: The full claim dict.
    :return: Formatted string summarizing tool calls up to the claim's turn.
    """
    history = claim.get("search_history", [])
    if not history:
        return ""
    lines = []
    for entry in history:
        line = _format_search_entry(entry)
        if line is not None:
            lines.append(line)
    return "\n".join(lines)


def format_prior_product_claims(claim: dict) -> str | None:
    """
    Format prior agent claims about the same product from earlier turns.

    :param claim: The full claim dict.
    :return: Formatted string listing prior claims by turn, or None if none exist.
    """
    prior = claim.get("prior_product_claims", [])
    if not prior:
        return None
    lines = [f"  Turn {p['turn']}: \"{p['statement'].strip()}\"" for p in prior]
    return "\n".join(lines)


def format_product_context(claim: dict) -> str:
    """
    Format the full product context for assessment MCQ prompts, combining
    user preferences, search history, and product ground truth data.

    :param claim: A single claim dict with ground_truth, user_preferences,
        and search_history.
    :return: Formatted context string for use in assessment prompts.
    """
    parts = []

    prefs = claim["user_preferences"]
    parts.append("USER PREFERENCES:")
    parts.append(format_preferences(prefs))
    parts.append("")

    search_history = claim.get("search_history", [])
    if search_history:
        parts.append("SEARCH HISTORY (all search_inventory calls up to this turn, across all products):")
        for entry in search_history:
            line = _format_search_entry(entry)
            if line is not None:
                parts.append(line)
        parts.append("")

    if claim["claim_type"] == "product":
        gt = claim["ground_truth"]
        parts.append(f"PRODUCT: {gt['title']}")
        parts.append("GROUND TRUTH:")
        parts.extend(_format_single_product_ground_truth(gt))
    else:
        parts.extend(_format_multi_product_ground_truth(claim["ground_truth"]))

    return "\n".join(parts)
