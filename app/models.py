from pydantic import BaseModel, Field


# ── Task 1: Co-occurrence ──────────────────────────────────────────────

class CooccurrenceItem(BaseModel):
    ingredient: str
    count: int


class CooccurrenceResponse(BaseModel):
    ingredient: str
    cooccurrence: list[CooccurrenceItem]


# ── Task 2: Duplicates ─────────────────────────────────────────────────

class IngredientInput(BaseModel):
    name: str
    quantity: str


class RecipeInput(BaseModel):
    name: str
    ingredients: list[IngredientInput]


class RecipeDuplicateRequest(BaseModel):
    recipe: RecipeInput


class DuplicateItem(BaseModel):
    name: str
    similarity: float = Field(..., ge=0, le=1)


class DuplicateResponse(BaseModel):
    duplicates: list[DuplicateItem]
