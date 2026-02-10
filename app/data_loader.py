"""
Data loader — reads the Allrecipes dataset and ingredient list once at startup.
Exposes two DataFrames: `recipes_df` and `ingredients_df`.
"""

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RECIPES_FILE = DATA_DIR / "allrecipes.com_database_12042020000000.json"
INGREDIENTS_FILE = DATA_DIR / "ingredient-list.json"

# ── Measurement / quantity words to strip ───────────────────────────────
MEASUREMENT_WORDS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "teaspoon",
    "teaspoons", "tsp", "ounce", "ounces", "oz", "pound", "pounds", "lb",
    "lbs", "gram", "grams", "g", "kilogram", "kg", "ml", "milliliter",
    "liter", "liters", "l", "gallon", "gallons", "quart", "quarts", "pint",
    "pints", "pinch", "pinches", "dash", "dashes", "slice", "slices",
    "piece", "pieces", "can", "cans", "package", "packages", "packet",
    "packets", "bag", "bags", "jar", "jars", "bottle", "bottles",
    "bunch", "bunches", "head", "heads", "stalk", "stalks", "sprig",
    "sprigs", "clove", "cloves", "stick", "sticks", "container",
    "large", "medium", "small", "fluid",
}

# Preparation words that appear after the ingredient name
PREP_WORDS = {
    "chopped", "diced", "minced", "sliced", "grated", "shredded",
    "crushed", "ground", "melted", "softened", "divided", "beaten",
    "peeled", "seeded", "halved", "quartered", "cubed", "julienned",
    "thawed", "drained", "rinsed", "sifted", "packed", "trimmed",
    "toasted", "cooked", "uncooked", "chilled", "frozen", "fresh",
    "dried", "canned", "optional", "or more to taste", "to taste",
    "room temperature", "at room temperature",
}

# Regex to strip leading quantity (numbers, fractions, ranges)
QTY_RE = re.compile(
    r"^[\d\s/.\-½¼¾⅓⅔⅛⅜⅝⅞]+",
)

# Regex to strip parenthetical content
PAREN_RE = re.compile(r"\(.*?\)")


def _clean_ingredient(raw: str) -> str:
    """
    Extract a normalised ingredient name from a raw string like
    '4 cups all-purpose flour, sifted'.
    """
    s = raw.lower().strip()

    # Remove parenthetical info  e.g. "(about 2 cups)"
    s = PAREN_RE.sub("", s)

    # Strip leading quantities
    s = QTY_RE.sub("", s).strip()

    # Tokenize, remove measurement words
    tokens = re.split(r"[\s,]+", s)
    tokens = [t for t in tokens if t and t not in MEASUREMENT_WORDS]

    # Rejoin and strip prep suffixes
    s = " ".join(tokens)
    for pw in PREP_WORDS:
        s = s.replace(pw, "")

    # Final cleanup
    s = re.sub(r"[,\-]+$", "", s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _load_known_ingredients() -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Returns:
      - mapping: lowered searchValue → canonical term, longest-first
      - word_index: word → list of searchValues containing that word
    """
    with open(INGREDIENTS_FILE, "r") as f:
        raw = json.load(f)
    mapping: dict[str, str] = {}
    for item in raw:
        sv = item["searchValue"].lower().strip()
        mapping[sv] = item["term"].lower().strip()
    mapping = dict(sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True))

    # Build reverse word index for fast candidate lookup
    word_index: dict[str, list[str]] = defaultdict(list)
    for sv in mapping:
        for word in sv.split():
            word_index[word].append(sv)

    return mapping, word_index


def _normalise_with_known(
    cleaned: str,
    known: dict[str, str],
    word_index: dict[str, list[str]],
) -> str:
    """
    Fast ingredient normalisation using word-level reverse index.
    Instead of checking all 9k ingredients, only check candidates that share
    at least one word with the cleaned string.
    """
    words = set(cleaned.split())
    # Gather candidate searchValues that share at least one word
    candidates: set[str] = set()
    for w in words:
        if w in word_index:
            candidates.update(word_index[w])

    # Check candidates longest-first (mapping is pre-sorted by length desc)
    for sv in sorted(candidates, key=len, reverse=True):
        if sv in cleaned:
            return known[sv]
    return cleaned


def load_data() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Returns
    -------
    recipes_df : DataFrame with columns [id, name, title, ingredients_raw, ingredients]
        `ingredients` is a frozenset of normalised ingredient names per recipe.
    known_ingredients : dict mapping searchValue → canonical term
    """
    logger.info("Loading known ingredients …")
    known, word_index = _load_known_ingredients()

    logger.info("Loading recipe database …")
    with open(RECIPES_FILE, "r") as f:
        raw_recipes = json.load(f)

    rows = []
    for r in raw_recipes:
        raw_ings = r.get("ingredients", [])
        cleaned = []
        for ri in raw_ings:
            c = _clean_ingredient(ri)
            c = _normalise_with_known(c, known, word_index)
            if c:
                cleaned.append(c)
        rows.append(
            {
                "id": r["id"],
                "name": r.get("name", ""),
                "title": r.get("title", ""),
                "description": r.get("description", ""),
                "ingredients_raw": raw_ings,
                "ingredients": frozenset(cleaned),
            }
        )

    recipes_df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(recipes_df)} recipes with {len(known)} known ingredients.")
    return recipes_df, known
