# Recipe Intelligence API — Technical Presentation

## Allrecipes Data Engineering Case Study

---

## 1. Architecture Overview

```
┌──────────────┐      ┌──────────────────────────────────────────┐
│   Client     │      │          FastAPI Application             │
│  (cURL/UI)   │─────▶│                                          │
│              │      │  ┌────────────────┐  ┌────────────────┐  │
│              │◀─────│  │ Co-occurrence   │  │  Duplicate     │  │
│              │      │  │ Service         │  │  Service       │  │
│              │      │  │ (dict[Counter]) │  │  (TF-IDF +     │  │
│              │      │  │                 │  │   cosine sim)  │  │
│              │      │  └───────┬────────┘  └───────┬────────┘  │
│              │      │          │                    │           │
│              │      │  ┌───────▼────────────────────▼────────┐  │
│              │      │  │         Data Loader                 │  │
│              │      │  │  (ingredient parsing + normalisation)│  │
│              │      │  └───────────────┬─────────────────────┘  │
│              │      └─────────────────┼──────────────────────────┘
│              │                        │
│              │               ┌────────▼─────────┐
│              │               │  Allrecipes JSON  │
│              │               │  (68,721 recipes) │
│              │               │  + ingredient     │
│              │               │    list (9,178)   │
│              │               └──────────────────┘
```

**Key design principle:** Pre-compute everything at startup, serve queries in milliseconds.

---

## 2. Technology Choices & Rationale

### Framework: FastAPI (over Flask, Django)

| Criteria | FastAPI | Flask | Django |
|---|---|---|---|
| **Auto API docs** | Built-in (Swagger + ReDoc) | Manual/extension | DRF needed |
| **Type safety** | Native Pydantic | Manual | Serializers |
| **Async support** | Native | Bolt-on | Bolt-on |
| **Startup hooks** | `lifespan` context manager | `before_first_request` (deprecated) | `AppConfig.ready()` |
| **Performance** | ~3x Flask throughput | Baseline | Heavy overhead |

**Why it matters for this case:**
- Pydantic models give us **input validation for free** — the POST body is automatically validated
- Built-in `/docs` page means the evaluator can test endpoints immediately without Postman
- `lifespan` context manager cleanly handles our expensive startup (data loading + index building)
- Type hints throughout make the codebase self-documenting

### Data Processing: Pandas (over Polars, raw Python)

| Criteria | Pandas | Polars | Raw Python |
|---|---|---|---|
| **Ecosystem** | Universal | Growing | N/A |
| **Readability** | High | Medium | Low for tabular |
| **Integration** | scikit-learn native | Needs conversion | Manual |
| **Dataset size (68k)** | Perfect fit | Overkill | Verbose |

**Why Pandas for this PoC:**
- 68k records fits comfortably in memory (~200MB)
- scikit-learn's TF-IDF expects Pandas/NumPy — zero conversion overhead
- If this were 10M+ recipes, Polars or Spark would be the move

### Duplicate Detection: TF-IDF + Cosine Similarity (over embeddings, Levenshtein, Jaccard)

| Approach | Pros | Cons | Fit for PoC |
|---|---|---|---|
| **TF-IDF + Cosine** | Fast, deterministic, no API keys, interpretable | No semantic understanding | Excellent |
| **Sentence Embeddings** | Semantic similarity | Requires GPU/API, non-deterministic | Overkill |
| **Levenshtein** | Simple | Character-level only, O(n*m) per pair | Poor |
| **Jaccard (set overlap)** | Simple | Ignores word importance/frequency | Decent |

**Why TF-IDF wins here:**
- Recipe names and ingredient lists are **structured, keyword-rich text** — TF-IDF excels at this
- Bigram features (`ngram_range=(1,2)`) catch phrases like "baking powder" as a unit
- `sublinear_tf=True` dampens high-frequency ingredients (salt, sugar) that would otherwise dominate
- Title repeated 3x to boost name-matching weight (a recipe named "Cinnamon Bread" should rank higher than one that just contains cinnamon)
- Query time: **<100ms** against 68k recipes

### Co-occurrence: In-memory Counter Dict (over graph DB, SQL, sparse matrix)

| Approach | Query Time | Memory | Complexity |
|---|---|---|---|
| **dict[Counter]** | O(k log k) | ~50MB | Low |
| **Neo4j graph** | Network RTT | Separate service | High |
| **SQLite** | Disk I/O | Low | Medium |
| **SciPy sparse matrix** | O(k) | ~30MB | Medium |

**Why dict[Counter]:**
- Simplest possible structure that gives O(1) lookup + O(k log k) top-N
- 68k recipes × ~8 ingredients/recipe → manageable in memory
- No external dependencies to deploy or manage
- `Counter.most_common(N)` is literally the API we need

---

## 3. Ingredient Normalisation Strategy

This is the hardest part of the problem — raw strings like `"4 cups all-purpose flour, sifted"` need to become `"all-purpose flour"`.

### Pipeline:

```
"4 cups all-purpose flour, sifted"
     │
     ▼  Strip parenthetical content
"4 cups all-purpose flour, sifted"
     │
     ▼  Remove leading quantities (regex: digits, fractions, unicode fractions)
"all-purpose flour, sifted"
     │
     ▼  Remove measurement words (cups, tablespoons, etc.)
"all-purpose flour sifted"
     │
     ▼  Remove preparation words (sifted, chopped, diced, etc.)
"all-purpose flour"
     │
     ▼  Match against known ingredients (9,178 from ingredient-list.json)
"all-purpose flour" ✓ (canonical match)
```

### Optimisation: Word-level Reverse Index

**Problem:** Naive matching = 550k strings × 9k ingredients = 5 billion substring checks → 66s startup

**Solution:** Build a reverse index: `word → [ingredients containing that word]`

For input `"all-purpose flour"`, only check ingredients containing "all-purpose" or "flour" (~50 candidates instead of 9,178).

**Result:** Startup dropped from **66s → 11s** (6x speedup).

---

## 4. API Design Decisions

### GET for co-occurrence, POST for duplicates
- Co-occurrence is idempotent + cacheable → GET with query params
- Duplicate detection takes a complex payload → POST with JSON body
- Follows REST conventions

### Error handling
- 404 when ingredient not found (not a 400 — the request is valid, the data doesn't exist)
- Pydantic validates request body structure automatically
- Similarity scores clamped to [0, 1] via Pydantic `Field(ge=0, le=1)`

### Configurable top_n
- Default 10 for co-occurrence (as spec'd), but user can request up to 50
- Hardcoded 5 for duplicates (as spec'd)

---

## 5. What I'd Do Differently in Production

| PoC | Production |
|---|---|
| In-memory data loading | PostgreSQL + Redis cache |
| Startup builds indexes | Background job / pre-built indexes |
| Single process | Kubernetes pods with shared cache (Redis) |
| TF-IDF for similarity | Hybrid: TF-IDF pre-filter + sentence embeddings re-rank |
| No auth | API key / OAuth2 |
| No rate limiting | Token bucket per client |
| `ingredient-list.json` lookup | NER model (spaCy) for ingredient extraction |
| Hardcoded data path | S3/GCS with versioned datasets |
| No monitoring | Prometheus metrics + Grafana dashboards |

### Scaling the co-occurrence engine:
- Precompute and store in Redis sorted sets → `ZREVRANGE` gives top-N in O(log N + k)
- Incremental updates: when a new recipe is added, only update pairs involving its ingredients

### Scaling duplicate detection:
- Use approximate nearest neighbors (FAISS / Annoy) instead of brute-force cosine similarity
- For 10M+ recipes, TF-IDF → FAISS index reduces query from O(N) to O(log N)
- Add a two-stage pipeline: TF-IDF pre-filter (top 100) → embedding re-rank (top 5)

---

## 6. Running the PoC

```bash
# Install
pip install -r requirements.txt

# Ensure data/ contains the unzipped dataset
# (allrecipes.com_database_12042020000000.json + ingredient-list.json)

# Run
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test Task 1
curl "http://localhost:8000/api/ingredient-cooccurrence?ingredient=cinnamon"

# Test Task 2
curl -X POST "http://localhost:8000/api/recipe-duplicates" \
  -H "Content-Type: application/json" \
  -d '{
    "recipe": {
      "name": "Cinnamon Bun Bread",
      "ingredients": [
        {"name": "all-purpose flour", "quantity": "3 cups"},
        {"name": "baking powder", "quantity": "1 tablespoon"}
      ]
    }
  }'

# Interactive docs
open http://localhost:8000/docs
```

---

## 7. Results Validation

### Task 1: `cinnamon` co-occurrence
| Rank | Ingredient | Count | Makes Sense? |
|---|---|---|---|
| 1 | white sugar | 3,421 | Baking staple with cinnamon |
| 2 | salt | 3,391 | Universal in baking |
| 3 | all-purpose flour | 2,777 | Baking base |
| 4 | butter | 2,530 | Cinnamon rolls, cookies |
| 5 | vanilla extract | 2,290 | Classic pairing |
| 6 | eggs | 2,179 | Baking |
| 7 | brown sugar | 1,964 | Cinnamon-brown sugar is iconic |
| 8 | nutmeg | 1,893 | The classic spice companion |
| 9 | baking soda | 1,657 | Leavening |
| 10 | baking powder | 1,548 | Leavening |

Cinnamon + nutmeg, cinnamon + brown sugar, cinnamon + vanilla — these are textbook flavor pairings. The data validates the approach.

### Task 1: `garlic` co-occurrence
| Rank | Ingredient | Count |
|---|---|---|
| 1 | black pepper | 7,186 |
| 2 | onion | 5,848 |
| 3 | olive oil | 5,838 |
| 4 | salt | 5,414 |
| 5 | water | 3,386 |

Garlic + olive oil + onion + black pepper = the foundation of Mediterranean/savory cooking. Validated.

### Task 2: "Cinnamon Bun Bread" duplicates
| Recipe | Similarity |
|---|---|
| Cinnamon Bun Icing | 0.47 |
| Cinnamon Carrot Bread | 0.46 |
| Cinnamon Bread I | 0.43 |
| Cinnamon Chip Bread | 0.38 |

Name-similar recipes with overlapping ingredient profiles rank highest. Working as intended.

---

## 8. Key Metrics

| Metric | Value |
|---|---|
| Dataset size | 68,721 recipes |
| Known ingredients | 9,178 |
| Startup time | ~11 seconds |
| Co-occurrence query | <50ms |
| Duplicate query | <100ms |
| Memory footprint | ~500MB |
| Lines of code | ~300 |
| External services | 0 |
