"""
GlowMatch REST API
=====================
Exposes the exact same recommender engine (src/recommender.py) as a
REST API using FastAPI -- so any client (a website, a mobile app, Postman,
another microservice) can get recommendations over HTTP instead of only
through the Streamlit UI.

WHY THIS FILE EXISTS (in plain English):
Streamlit is great for a human clicking buttons in a browser. It is NOT
how real production systems talk to each other -- a mobile app, another
backend service, or a frontend website needs a REST API it can call
programmatically and get back structured JSON. This file is that layer.

KEY CONCEPTS DEMONSTRATED:
1. FastAPI route decorators (@app.get) define REST endpoints.
2. `async def` -- endpoints are declared asynchronous so the server can
   handle many concurrent requests without one slow request blocking
   everyone else (real production APIs almost always do this).
3. Redis caching -- repeated identical queries are served from a fast
   in-memory cache instead of recomputing every time. If a real Redis
   server isn't running (e.g. on a laptop with nothing installed), this
   file automatically and gracefully falls back to a plain Python dict
   as an in-memory cache, so the API still works and still demonstrates
   the caching PATTERN correctly -- it just isn't distributed/persistent
   without real Redis running. This fallback is intentional and logged
   clearly, not hidden.
4. Pydantic response models -- FastAPI auto-generates request validation
   and interactive API docs (visit /docs) from these type hints, for free.

HOW TO RUN:
    pip install -r requirements.txt
    uvicorn src.api:app --reload --port 8000

Then open http://localhost:8000/docs for an interactive Postman-like UI,
or test with:
    curl "http://localhost:8000/recommend?skin_type=Dry&max_price=50"
"""

import sys, os, json, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from recommender import GlowMatchRecommender, SKIN_TYPES

# ----------------------------------------------------------------------
# CACHE LAYER: try real Redis first, fall back to an in-memory dict
# ----------------------------------------------------------------------
CACHE_TTL_SECONDS = 300  # how long a cached answer stays valid

class _InMemoryCache:
    """Fallback cache used when no real Redis server is reachable.
    Same get/set interface as redis-py so the rest of the code doesn't
    need to know or care which backend is actually active."""
    def __init__(self):
        self.store = {}

    def get(self, key):
        entry = self.store.get(key)
        if not entry:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del self.store[key]
            return None
        return value

    def set(self, key, value, ex=CACHE_TTL_SECONDS):
        self.store[key] = (value, time.time() + ex)


def _connect_cache():
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=1)
        client.ping()  # actually test the connection, don't just assume it worked
        print("[cache] Connected to real Redis at localhost:6379")
        return client, "redis"
    except Exception as e:
        print(f"[cache] No Redis server reachable ({e}). Falling back to in-memory cache.")
        return _InMemoryCache(), "in-memory"

cache, cache_backend = _connect_cache()


def cache_get(key):
    val = cache.get(key)
    if val is None:
        return None
    return json.loads(val) if isinstance(val, (str, bytes)) else val


def cache_set(key, value):
    payload = json.dumps(value)
    if cache_backend == "redis":
        cache.set(key, payload, ex=CACHE_TTL_SECONDS)
    else:
        cache.set(key, payload)


# ----------------------------------------------------------------------
# LOAD THE MODEL ONCE AT STARTUP (not per-request -- would be far too slow)
# ----------------------------------------------------------------------
print("Loading GlowMatch recommender...")
engine = GlowMatchRecommender()
print(f"Ready. {len(engine.products)} products loaded. Cache backend: {cache_backend}")

app = FastAPI(
    title="GlowMatch API",
    description="Hybrid BeautyTech recommendation REST API — content-based, "
                 "vector search, and rule-based personalization over a "
                 "1,472-product cosmetics catalog.",
    version="1.0.0",
)


# ----------------------------------------------------------------------
# RESPONSE SCHEMAS (Pydantic) -- defines the exact JSON shape returned.
# This IS the Low-Level Design (LLD) for the API contract, expressed as code.
# ----------------------------------------------------------------------
class Product(BaseModel):
    product_id: int
    Brand: str
    Name: str
    Label: str
    Price: float
    Rank: float
    match_score: Optional[float] = None

class RecommendResponse(BaseModel):
    count: int
    cache_hit: bool
    cache_backend: str
    latency_ms: float
    results: List[Product]

class ExplainResponse(BaseModel):
    query_product_id: int
    recommended_product_id: int
    explanation: str


# ----------------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "GlowMatch API",
        "status": "running",
        "cache_backend": cache_backend,
        "products_loaded": len(engine.products),
        "docs": "/docs",
    }


@app.get("/recommend", response_model=RecommendResponse)
async def recommend(
    skin_type: str = Query(..., description=f"One of: {SKIN_TYPES}"),
    category: Optional[str] = Query(None, description="Optional product category filter"),
    max_price: Optional[float] = Query(None, description="Optional max budget"),
    n: int = Query(10, le=50),
):
    """Personalized, cold-start-friendly recommendations by skin profile."""
    if skin_type not in SKIN_TYPES:
        raise HTTPException(status_code=400, detail=f"skin_type must be one of {SKIN_TYPES}")

    cache_key = f"recommend:{skin_type}:{category}:{max_price}:{n}"
    t0 = time.time()

    cached = cache_get(cache_key)
    if cached is not None:
        return RecommendResponse(
            count=len(cached), cache_hit=True, cache_backend=cache_backend,
            latency_ms=round((time.time() - t0) * 1000, 3), results=cached,
        )

    df = engine.recommend_for_skin_profile(skin_type, category=category, max_price=max_price, n=n)
    results = df.to_dict(orient="records")
    cache_set(cache_key, results)

    return RecommendResponse(
        count=len(results), cache_hit=False, cache_backend=cache_backend,
        latency_ms=round((time.time() - t0) * 1000, 3), results=results,
    )


@app.get("/similar", response_model=RecommendResponse)
async def similar(product_id: int, n: int = Query(10, le=50)):
    """'Because you liked X' -- content-based + vector search blend."""
    if product_id not in engine.products["product_id"].values:
        raise HTTPException(status_code=404, detail=f"product_id {product_id} not found")

    cache_key = f"similar:{product_id}:{n}"
    t0 = time.time()

    cached = cache_get(cache_key)
    if cached is not None:
        return RecommendResponse(
            count=len(cached), cache_hit=True, cache_backend=cache_backend,
            latency_ms=round((time.time() - t0) * 1000, 3), results=cached,
        )

    df = engine.recommend_similar_products(product_id, n=n)
    results = df.to_dict(orient="records")
    cache_set(cache_key, results)

    return RecommendResponse(
        count=len(results), cache_hit=False, cache_backend=cache_backend,
        latency_ms=round((time.time() - t0) * 1000, 3), results=results,
    )


@app.get("/trending", response_model=RecommendResponse)
async def trending(category: Optional[str] = None, n: int = Query(10, le=50)):
    """Top-rated products, optionally filtered by category. Popularity fallback."""
    cache_key = f"trending:{category}:{n}"
    t0 = time.time()

    cached = cache_get(cache_key)
    if cached is not None:
        return RecommendResponse(
            count=len(cached), cache_hit=True, cache_backend=cache_backend,
            latency_ms=round((time.time() - t0) * 1000, 3), results=cached,
        )

    df = engine.trending(n=n, category=category)
    results = df.to_dict(orient="records")
    cache_set(cache_key, results)

    return RecommendResponse(
        count=len(results), cache_hit=False, cache_backend=cache_backend,
        latency_ms=round((time.time() - t0) * 1000, 3), results=results,
    )


@app.get("/explain", response_model=ExplainResponse)
async def explain(query_product_id: int, recommended_product_id: int):
    """Mini-RAG: retrieval-grounded explanation for why a product was recommended."""
    for pid in (query_product_id, recommended_product_id):
        if pid not in engine.products["product_id"].values:
            raise HTTPException(status_code=404, detail=f"product_id {pid} not found")

    explanation = engine.explain_recommendation(query_product_id, recommended_product_id)
    return ExplainResponse(
        query_product_id=query_product_id,
        recommended_product_id=recommended_product_id,
        explanation=explanation,
    )
