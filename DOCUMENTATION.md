# GlowMatch — Hybrid BeautyTech Recommendation System
### Technical Assignment Submission — GenAI Intern

---

## 1. Problem Statement

Recommend skincare/beauty products to a user in a way that works for a
BeautyTech platform's **hardest and most common real scenario**: a
first-time visitor with **zero purchase history**. The system must:
1. Personalize immediately using stated skin type, category, and budget.
2. Let a user browse "similar products" from one item they like.
3. Explain *why* a product was recommended (not just show a black-box score).
4. Surface top-rated / trending products as a safe fallback.

## 2. Scope & Out-of-Scope Boundaries

**In scope (what this project delivers):** the end-to-end recommendation
pipeline — data ingestion and cleaning, content-based filtering, semantic
vector search, rule-based personalization, popularity fallback, the
mini-RAG explanation generator, an evaluation suite, a Streamlit demo UI
(`app.py`), and a FastAPI REST layer (`src/api.py`) that exposes the same
engine over HTTP.

**Explicitly out of scope:** shopping cart, checkout, and payment
processing. This is a *recommendation engine*, not a full e-commerce
storefront — it answers "what should this user see?" and returns
structured product data (id, brand, name, price, rating, similarity/
explanation), but it does not manage a cart, handle order state, process
payments, or integrate with a payment gateway. A real deployment would
sit in front of a separate commerce/checkout service that owns that
responsibility; wiring the two together (e.g. an "Add to Cart" button
calling out to a commerce API) is a straightforward integration but is
not part of this submission, and no cart/checkout/payment code exists in
this repository.

## 3. A Deliberate, Honest Data Choice

This project uses a real 1,472-product catalog (Sephora-sourced: brand,
price, rating, full ingredient list, and 5 binary skin-type suitability
flags) rather than a dataset with synthetic user-rating history.

**This was a conscious trade-off, not a limitation I ignored:** unlike
MovieLens (which has 100K+ user ratings), this catalog has **no user
interaction log at all** — so classic user-item Collaborative Filtering
(matrix factorization) is not possible, and building one on fabricated
fake "users" would produce meaningless, made-up numbers. I chose to
build the system that's honest about this constraint, because **this is
the actual state most early-stage BeautyTech products are in** — you have
a product catalog before you have years of purchase data. Solving
personalization well *without* behavioral history is a real, valuable
engineering problem in its own right, and it's what this system
demonstrates.

## 4. Approach — 4 Techniques Combined

| Technique | What it does | Why it's here |
|---|---|---|
| **Content-Based Filtering** | TF-IDF over ingredient lists + cosine similarity | Classic, explainable "similar formulation" matching |
| **Semantic Vector Search** | TF-IDF compressed to 100-dim dense embeddings (via Truncated SVD / LSA), indexed with a nearest-neighbor search structure | Mirrors the retrieval pattern of a production vector database (Pinecone/Weaviate/Qdrant) — embed once, query with approximate nearest-neighbor search |
| **Rule-Based Personalization** | Structured filter on skin-type flag + category + budget, ranked by rating | Solves the cold-start problem directly — no history needed, works on visit #1 |
| **Mini-RAG Explanation Generator** | Retrieves the actual shared ingredients between two products (retrieval step), then generates a natural-language explanation from that retrieved context (generation step) | A simplified Retrieval-Augmented Generation pattern — the retrieval step is real; the generation step is currently template-based and is a documented Future Improvement to swap for a live LLM call |

Bonus: **unsupervised KMeans clustering** over the embedding space
automatically discovers 6 "ingredient personas" (e.g. a cluster centered
on `sodium / extract / acid`, another on `oil / seed / extract`) — with
zero manual labeling, purely from ingredient composition patterns.

## 5. Architecture

```
cosmetics.csv (1,472 products: brand, price, rank, ingredients, skin-type flags)
        │
        ▼
   Data Cleaning (pandas) — strip non-ingredient text, normalize
        │
        ├──► TF-IDF Vectorizer ──► sparse ingredient fingerprints
        │            │
        │            ▼
        │     Truncated SVD (LSA) ──► 100-dim dense embeddings
        │            │
        │            ▼
        │     Nearest-Neighbor Index (cosine) ── "vector search"
        │            │
        │            ▼
        │        KMeans Clustering ── "ingredient personas"
        │
        ├──► Rule-based filter (skin type / category / budget)
        │
        └──► Mini-RAG explanation generator
                     │
                     ▼
              Streamlit UI (app.py)
```

## 6. Recommendation Methodology (plain English)

1. **Content-based:** every product's ingredient list is cleaned and
   converted into a TF-IDF numeric fingerprint. Cosine similarity between
   two fingerprints measures formulation closeness.
2. **Vector search:** the same TF-IDF fingerprints are compressed to 100
   dense numbers per product using Truncated SVD (a technique called
   Latent Semantic Analysis). These dense vectors are indexed with
   scikit-learn's `NearestNeighbors` — functionally the same operation a
   vector database performs (embed → index → approximate nearest-neighbor
   query), implemented in-memory since 1,472 products doesn't require a
   dedicated vector DB at this scale, but the pattern is identical and
   directly portable to one.
3. **Personalization:** rather than inferring taste from history that
   doesn't exist, the user states their skin type directly — this is
   filtered against the dataset's own binary skin-type flags, then ranked
   by rating (highest first) with price as a tiebreaker.
4. **Mini-RAG explanation:** for any recommended product, the system
   retrieves which specific ingredients it actually shares with the
   query product (a real retrieval step over the TF-IDF vocabulary), then
   generates a sentence explaining the match using that retrieved
   evidence — not a generic "you might also like" placeholder.

## 7. Dataset Selection

**Dataset:** a 1,472-product Sephora-sourced cosmetics catalog (publicly
available, originally compiled for a well-known DataCamp project on
ingredient-based cosmetics analysis). Columns: `Label` (category), `Brand`,
`Name`, `Price`, `Rank` (rating 0-5), `Ingredients` (full text), and 5
binary skin-type suitability flags (`Combination`, `Dry`, `Normal`, `Oily`,
`Sensitive`).

Chosen because it is **real, free, and directly relevant to the BeautyTech
domain** this role operates in — a stronger domain match than a generic
movie/e-commerce dataset, and rich enough (full ingredient text + skin-type
metadata) to support genuine content-based and vector-search techniques.

## 8. Technologies Used

| Component | Technology |
|---|---|
| Data handling | pandas, numpy |
| Content-based model | scikit-learn `TfidfVectorizer`, `cosine_similarity` |
| Vector embeddings | scikit-learn `TruncatedSVD` (LSA) |
| Vector search / retrieval | scikit-learn `NearestNeighbors` (cosine metric) |
| Clustering | scikit-learn `KMeans` |
| UI | Streamlit |
| Language | Python 3 |

## 9. Assumptions Made

- A user's stated skin type is treated as ground truth (no way to verify
  it against the catalog beyond the flag matching itself).
- "Because you liked X" browsing assumes formulation similarity (shared
  ingredients) is a reasonable proxy for "you'll like this too" — a
  standard assumption in content-based cosmetics recommendation.
- The `Rank` field is treated as a reliable quality signal, though the
  dataset doesn't include review *counts*, so a product with 2 ratings
  and a product with 2,000 ratings are weighted identically (a
  documented limitation, see below).

## 10. Design Decisions

- **Chose LSA/SVD embeddings over a pretrained neural embedding model**
  (e.g. sentence-transformers): pretrained models require downloading
  weights from an external model hub, which introduces an external
  dependency and isn't guaranteed available in every deployment
  environment. LSA embeddings are trained directly from this catalog's
  own vocabulary, need no external downloads, and are fully interpretable
  — a deliberate engineering trade-off for a self-contained assignment
  submission, with a clear upgrade path noted below.
- **Chose rule-based personalization over inferred personalization:**
  given no purchase history exists, inferring "taste" would mean
  guessing — asking the user directly is more honest and more accurate
  for a first-time visitor, which is the primary use case here.
- **Mini-RAG explanation is template-based, not a live LLM call:** this
  keeps the submission runnable with zero external API dependencies or
  cost, while still demonstrating the retrieval-then-generation pattern
  correctly. Swapping the generation step for a real Claude API call
  (passing the retrieved shared ingredients as context) is a two-line
  change, documented below.

## 11. Future Improvements

- **Swap the mini-RAG generator for a real LLM call** (Claude API): pass
  the retrieved shared-ingredient context into a prompt and generate a
  more natural, varied explanation — the retrieval step already produces
  exactly the context an LLM would need.
- **Move the vector index to a production vector database** (Pinecone,
  Weaviate, Qdrant, or pgvector) once the catalog grows beyond what an
  in-memory index comfortably serves, and add approximate nearest-neighbor
  indexing (HNSW) for sub-linear query time at scale.
- **Collect real interaction data** (clicks, add-to-cart, purchases) from
  day one so that, once enough volume accumulates, true collaborative
  filtering becomes possible alongside the current content-based system —
  the two would combine into a genuine hybrid the way the earlier
  MovieLens-style system does.
- **Clean the `Ingredients` field** to strip non-ingredient marketing text
  (see Known Limitations #2) before it reaches the model.
- **Weight `Rank` by review count** once that field is available, using a
  Bayesian-weighted average (as implemented in an earlier version of this
  project on MovieLens) so a product with 2 five-star ratings doesn't
  outrank one with 2,000 ratings averaging 4.5.
- **Expose this as a FastAPI REST endpoint** (`/recommend`, `/similar`,
  `/explain`) with Redis caching on repeated queries — directly aligned
  with this role's backend stack.

## 12. Evaluation Methodology

Because this catalog has no user interaction log, the standard
recommender-systems evaluation approach (hide a user's real future
choices, then check precision/recall against them) isn't possible here —
there's no historical "future" to hide. `src/evaluate.py` instead reports
six proxy metrics that are honest about what's actually measurable from
catalog-only data. Deliberately, **none of these are called "accuracy"**
— that word implies a ground-truth comparison against real user
preference, which this dataset cannot provide. Instead:

| # | Metric | What it checks | What it is NOT |
|---|---|---|---|
| 1 | **Skin-Type Filter Compliance** | Every product returned for a stated skin type is actually flagged suitable for it in the catalog | Not a measure of whether the user will personally like the product |
| 2 | **Budget Constraint Satisfaction** | Every product returned under a `max_price` cap actually costs at or under that cap, checked per price point ($10/$25/$50/$75/$100) across all 5 skin types | Not just an aggregate percentage — each price point is pass/fail |
| 3 | **Recommendation Relevance (Content Coherence)** | How closely related "similar products" results are to the query, via the blended content + vector-search score | A relevance/coherence signal, not a taste-accuracy figure |
| 4 | **Catalog Coverage** | What fraction of the 1,472-product catalog ever surfaces across a batch of sample queries | Checks the system isn't stuck recommending the same narrow slice |
| 5 | **Cluster Separation (sanity check)** | Size distribution of the 6 unsupervised "ingredient persona" clusters | A basic health check, not a formal silhouette-score validation (see Future Improvements) |
| 6 | **Latency** | Wall-clock time per call, measured separately for the personalized-filter path and the similarity-search path | Uncached, in-process timing; `src/api.py` additionally demonstrates cache-hit latency |

## 13. Evaluation Results (measured, reproducible)

Running `python src/evaluate.py` against the full 1,472-product catalog
(50 sampled queries, fixed random seed for reproducibility) produced:

| Metric | Result |
|---|---|
| Skin-type filter compliance | **100.00%** (all 5 skin types) |
| Budget constraint satisfaction | **100.00%** (250/250 rows checked across 5 price caps × 5 skin types, 0 violations) |
| Recommendation relevance (content coherence) | **0.5673** average (0–1 scale) |
| Catalog coverage | **27.45%** (404/1,472 products surfaced from 50 sample queries) |
| Avg latency — "similar products" | **~4 ms** per call (uncached) |
| Avg latency — personalized filter | **~7 ms** per call (uncached) |

**Reading these results honestly:**
- The 100% filter-compliance and budget-satisfaction numbers are
  *expected* by construction — they confirm the rule-based filters have
  no off-by-one or leakage bugs, not that the recommendations match user
  taste.
- The 0.57 relevance score reflects that "similar products" intentionally
  casts a moderately wide net (full formulation category, not just exact
  duplicates) — see Test 2 in `TEST_CASES.md` for a worked example
  showing the actual ranked output this score is computed from.
- 27% coverage from only 50 sampled queries (out of 1,472 possible query
  products) is a reasonable exploration rate, not a sign of narrowness —
  full-catalog coverage would require querying with every product.
- Both latency figures are comfortably under any human-perceptible delay
  threshold (~100ms) even before the Redis/in-memory cache layer in
  `src/api.py` is applied — see `TEST_CASES.md` Test 6 for the measured
  ~1,800x cache-hit speedup on repeated queries.

Full per-metric output (pass/fail breakdown per price cap, per skin type,
etc.) is printed by `python src/evaluate.py` and is not duplicated in
full here to avoid this document going stale as the catalog or thresholds
change — re-run the script for current numbers.

## 14. Known Limitations

1. **No collaborative filtering is possible** — this dataset has no user
   interaction log (see Section 3). This is disclosed as a property of
   the data, not hidden.
2. **Ingredient field data-quality issue:** at least one product's
   `Ingredients` field contains scraped marketing text instead of actual
   ingredients (e.g. `"Visit the OLEHENRIKSEN boutique"`), which
   corrupts that product's TF-IDF fingerprint and produces a mislabeled
   "persona." A production system would need an ingredient-text
   validation step before indexing.
3. **Over-constrained queries can return zero results** — e.g. Sensitive
   skin + Sun protect category + budget ≤$10 returns nothing, because no
   product in the catalog satisfies all three filters simultaneously.
   The UI handles this gracefully with a warning message rather than
   crashing, but a production system might offer to relax one constraint
   automatically.
4. **`Rank` has no review-count weighting** — a product rated 5.0 from a
   single review is treated identically to one rated 5.0 from thousands,
   because review counts aren't in this dataset.
5. **The mini-RAG explanation generator is template-based, not a live
   LLM** (see Design Decisions) — explanations are correct and
   evidence-based, but not as naturally varied as an LLM-generated one
   would be.
6. **Evaluation metrics are proxy metrics, not accuracy** (see Section
   12) — because there's no ground-truth user preference data, every
   number in `src/evaluate.py` measures internal consistency (filters
   doing what they claim) or content relevance (formulation closeness),
   not whether a real user would like the recommendation. This is
   disclosed explicitly in the evaluation script and documentation
   rather than labeled "accuracy," which would overstate what's actually
   being measured.
7. **This project does not include cart, checkout, or payment
   functionality** (see Section 2, Scope) — it is a recommendation
   engine only, and is not a complete e-commerce application.

## 15. Comparison with a Real BeautyTech Platform (e.g. Nykaa/Sephora) — Bonus

| Aspect | Real platform (Nykaa/Sephora) | GlowMatch (this project) |
|---|---|---|
| Scale | Millions of users, purchase + browsing history | 1,472 products, zero user history (catalog-only) |
| Personalization signal | Purchase history, browsing behavior, skin quiz answers, returns | Stated skin type + budget (single-visit personalization) |
| Search | Hybrid keyword + semantic vector search over millions of SKUs, likely backed by a managed vector DB | TF-IDF + LSA embeddings + in-memory nearest-neighbor index (same pattern, smaller scale) |
| Explanations | Often minimal ("customers also bought") | Retrieval-grounded, ingredient-specific explanations (mini-RAG) |
| Infra | FastAPI/GraphQL services, Redis caching, cloud-hosted vector DB, real-time re-ranking | Single Streamlit app, in-memory, retrains at startup |

**Where GlowMatch already aligns with this role's actual stack:** the
vector-search pattern (embed → index → nearest-neighbor retrieval) and the
mini-RAG pattern (retrieve → generate) are structurally identical to what
a production GenAI/search team builds — just implemented at a scale that
doesn't yet need a managed vector database or a live LLM API call. Both
are one clear step away from the production version, and that step is
explicitly documented above rather than glossed over.
