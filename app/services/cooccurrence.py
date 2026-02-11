"""
Task 1 — Ingredient Co-occurrence Service

Builds an in-memory co-occurrence index at startup and answers queries in O(k log k)
where k = number of unique co-occurring ingredients for the queried term.
"""

from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd


class CooccurrenceService:
    def __init__(self, recipes_df: pd.DataFrame):
        self._index: dict[str, Counter] = defaultdict(Counter)
        self._build_index(recipes_df)

    # ── Index Construction ──────────────────────────────────────────────

    def _build_index(self, df: pd.DataFrame) -> None:
        """
        For every recipe, iterate over all ingredient pairs and increment
        both directions of the co-occurrence counter.

        Time complexity: O(R * I²) where R = recipes, I = avg ingredients/recipe.
        Space: O(U²) where U = unique ingredients (sparse — only observed pairs stored).
        """
        for ingredient_set in df["ingredients"]:
            if len(ingredient_set) < 2:
                continue
            for a, b in combinations(ingredient_set, 2):
                self._index[a][b] += 1
                self._index[b][a] += 1

    # ── Query ───────────────────────────────────────────────────────────

    def query(self, ingredient: str, top_n: int = 10) -> list[dict]:
        """
        Return the top-N ingredients most frequently co-occurring with `ingredient`.
        """
        ingredient = ingredient.lower().strip()
        counter = self._index.get(ingredient)
        if counter is None:
            return []
        return [
            {"ingredient": ing, "count": cnt}
            for ing, cnt in counter.most_common(top_n)
        ]
