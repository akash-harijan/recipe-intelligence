"""
Task 2 — Recipe Duplicate Detection Service

Uses TF-IDF on a combined text representation (title + ingredient names) and
cosine similarity to find the most similar existing recipes to an input.

Why TF-IDF over embeddings?
  - Zero external dependencies (no API keys, no GPU)
  - Deterministic, interpretable similarity scores
  - Fast: vectorizing a single query against a pre-built matrix is O(V) where V = vocab size
  - Good enough for near-duplicate detection on structured text
"""

import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DuplicateService:
    def __init__(self, recipes_df: pd.DataFrame):

        self._df = recipes_df.copy()
        self._vectorizer: TfidfVectorizer = None
        self._tfidf_matrix = None
        self._build_index()

    # ── Index Construction ──────────────────────────────────────────────

    def _recipe_text(self, row: pd.Series) -> str:
        """
        Create a combined text representation for TF-IDF.
        Title is repeated to give it more weight (name is the strongest
        signal for near-duplicates).
        """
        title = row["title"] or ""
        ingredients = " ".join(sorted(row["ingredients"])) if row["ingredients"] else ""
        # Repeat title 3× to boost its weight relative to ingredients
        return f"{title} {title} {title} {ingredients}".lower()

    def _build_index(self) -> None:
        logger.info("Building TF-IDF matrix for duplicate detection …")
        corpus = self._df.apply(self._recipe_text, axis=1).tolist()

        self._vectorizer = TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),   # unigrams + bigrams for phrase matching
            sublinear_tf=True,    # dampens high term-frequency dominance
            strip_accents="unicode",
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        logger.info(
            f"TF-IDF matrix shape: {self._tfidf_matrix.shape} "
            f"(recipes × features)"
        )

    # ── Query ───────────────────────────────────────────────────────────

    def find_duplicates(
        self,
        name: str,
        ingredients: list[str],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Vectorize the input recipe and find the top-N most similar recipes
        from the existing corpus.
        """
        ingredient_text = " ".join(sorted(ingredients))
        query_text = f"{name} {name} {name} {ingredient_text}".lower()

        query_vec = self._vectorizer.transform([query_text])
        
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Get top N+1 in case the exact recipe is in the DB (we'd skip it)
        top_indices = np.argsort(scores)[::-1][: top_n + 5]

        results = []
        for idx in top_indices:
            if len(results) >= top_n:
                break
            score = float(scores[idx])
            if score < 0.01:  # ignore near-zero similarities
                break
            title = self._df.iloc[idx]["title"]
            results.append(
                {"name": title, "similarity": round(score, 2)}
            )
        return results
