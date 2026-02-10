"""
Recipe Intelligence API — Allrecipes PoC
=========================================
FastAPI application that exposes:
  - GET  /api/ingredient-cooccurrence   (Task 1 — required)
  - POST /api/recipe-duplicates         (Task 2 — optional)

Startup loads ~68k recipes into memory and pre-computes indexes.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app.data_loader import load_data
from app.models import (
    CooccurrenceResponse,
    DuplicateResponse,
    RecipeDuplicateRequest,
)
from app.services.cooccurrence import CooccurrenceService
from app.services.duplicates import DuplicateService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Global service singletons (initialised on startup) ──────────────────
cooccurrence_svc: CooccurrenceService | None = None
duplicate_svc: DuplicateService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build indexes once before the first request."""
    global cooccurrence_svc, duplicate_svc

    t0 = time.perf_counter()
    recipes_df, _known = load_data()

    logger.info("Building co-occurrence index …")
    cooccurrence_svc = CooccurrenceService(recipes_df)

    logger.info("Building duplicate-detection index …")
    duplicate_svc = DuplicateService(recipes_df)

    elapsed = time.perf_counter() - t0
    logger.info(f"Startup complete in {elapsed:.1f}s")

    yield  # app is running

    logger.info("Shutting down …")


app = FastAPI(
    title="Recipe Intelligence API",
    description=(
        "PoC for Allrecipes data engineering case — ingredient co-occurrence "
        "and recipe duplicate detection."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Task 1: Ingredient Co-occurrence ───────────────────────────────────

@app.get(
    "/api/ingredient-cooccurrence",
    response_model=CooccurrenceResponse,
    summary="Top ingredients co-occurring with a given ingredient",
    responses={
        404: {"description": "Ingredient not found in the dataset"},
    },
)
async def ingredient_cooccurrence(
    ingredient: str = Query(
        ...,
        description="The ingredient to look up (e.g. 'cinnamon')",
        min_length=1,
    ),
    top_n: int = Query(10, ge=1, le=50, description="Number of results"),
):
    results = cooccurrence_svc.query(ingredient, top_n=top_n)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"Ingredient '{ingredient}' not found or has no co-occurrences.",
        )
    return CooccurrenceResponse(ingredient=ingredient, cooccurrence=results)


# ── Task 2: Recipe Duplicates ──────────────────────────────────────────

@app.post(
    "/api/recipe-duplicates",
    response_model=DuplicateResponse,
    summary="Find similar recipes in the database",
)
async def recipe_duplicates(body: RecipeDuplicateRequest):
    ingredient_names = [ing.name for ing in body.recipe.ingredients]
    results = duplicate_svc.find_duplicates(
        name=body.recipe.name,
        ingredients=ingredient_names,
        top_n=5,
    )
    return DuplicateResponse(duplicates=results)
