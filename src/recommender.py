"""
GlowMatch — Hybrid Beauty/Skincare Recommendation Engine
===========================================================
Built for a BeautyTech context (Sephora-sourced product catalog: 1,472
products, 116 brands, 6 categories, full ingredient lists, 5 skin-type
suitability flags).

IMPORTANT DATA REALITY (stated up front, not hidden):
This catalog has NO per-user rating history (no userId/purchase log) —
unlike MovieLens. So classic user-item Collaborative Filtering (matrix
factorization) is not possible here, and pretending otherwise would be
dishonest. Instead, this system uses techniques that are appropriate for
a **catalog-only, cold-start-heavy** domain — which is realistic, because
a new beauty D2C brand launching a recommender genuinely faces exactly
this problem (few users, rich product metadata) before it accumulates
years of purchase history.

ARCHITECTURE (4 techniques combined):
1. Content-Based Filtering       — TF-IDF over ingredient lists + cosine similarity
2. Semantic Vector Search        — dense LSA embeddings (TF-IDF -> Truncated SVD)
                                     indexed with approximate nearest-neighbor search,
                                     the same retrieval pattern a production vector
                                     database (e.g. Pinecone/Weaviate/Qdrant) implements
3. Rule-Based Personalization     — skin-type / category / budget filtering, the kind
                                     of structured query a recommendation API layer
                                     would expose
4. Retrieval-Augmented Explanation (mini-RAG) — retrieves the actual shared
                                     ingredients between the query and each
                                     recommendation (retrieval step), then generates
                                     a natural-language "why this was recommended"
                                     explanation (generation step). This is a
                                     template-based generator standing in for an LLM
                                     call — swapping the generation step for a real
                                     Claude API call is a documented Future Improvement.
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SKIN_TYPES = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]

# Columns returned to callers (API/UI) for every recommendation-style
# result. Centralized here (instead of repeated inline literals) so the
# response shape only needs to change in one place.
DISPLAY_COLS = ["product_id", "Label", "Brand", "Name", "Price", "Rank"]


class GlowMatchRecommender:
    def __init__(self, data_dir=DATA_DIR, n_embed_dims=100, n_personas=6):
        self.data_dir = data_dir
        self.n_embed_dims = n_embed_dims
        self.n_personas = n_personas
        self._load_data()
        self._build_content_model()
        self._build_vector_index()
        self._build_personas()

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def _load_data(self):
        self.products = pd.read_csv(os.path.join(self.data_dir, "cosmetics.csv"))
        self.products = self.products.reset_index(drop=True)
        self.products["product_id"] = self.products.index

    # ------------------------------------------------------------------
    # 1) CONTENT-BASED MODEL (ingredients -> TF-IDF -> cosine similarity)
    # ------------------------------------------------------------------
    def _build_content_model(self):
        def clean_ingredients(text):
            text = str(text).lower()
            text = re.sub(r"[^a-z,\s]", " ", text)
            return text

        self.products["ingredients_clean"] = self.products["Ingredients"].apply(clean_ingredients)
        self.tfidf = TfidfVectorizer(stop_words="english", max_features=3000)
        self.tfidf_matrix = self.tfidf.fit_transform(self.products["ingredients_clean"])
        # Cached once (fit is expensive to re-run; the vocabulary doesn't
        # change after fit_transform), reused by persona naming and by
        # every explain_recommendation() call instead of recomputing it
        # from the vectorizer each time.
        self._feature_names = np.array(self.tfidf.get_feature_names_out())

    # ------------------------------------------------------------------
    # 2) SEMANTIC VECTOR SEARCH (dense embeddings + nearest-neighbor index)
    #    Stands in for a production vector database: same retrieval
    #    pattern (embed -> index -> approximate nearest neighbor search),
    #    implemented in-memory for this project's scale (1,472 products).
    # ------------------------------------------------------------------
    def _build_vector_index(self):
        k = min(self.n_embed_dims, min(self.tfidf_matrix.shape) - 1)
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        self.embeddings = self.svd.fit_transform(self.tfidf_matrix)  # dense (n_products x k)

        self.vector_index = NearestNeighbors(n_neighbors=20, metric="cosine")
        self.vector_index.fit(self.embeddings)

    def vector_search(self, product_id, n=10):
        """Semantic nearest-neighbor retrieval in embedding space."""
        vec = self.embeddings[product_id].reshape(1, -1)
        distances, indices = self.vector_index.kneighbors(vec, n_neighbors=n + 1)
        results = [(idx, 1 - dist) for idx, dist in zip(indices[0], distances[0]) if idx != product_id][:n]
        return results  # list of (product_id, similarity_score)

    # ------------------------------------------------------------------
    # UNSUPERVISED CLUSTERING -> "ingredient personas"
    # (a lightweight substitute for audience/cohort signals, given no
    # user interaction data exists in this catalog-only dataset)
    # ------------------------------------------------------------------
    def _build_personas(self):
        self.kmeans = KMeans(n_clusters=self.n_personas, random_state=42, n_init=10)
        self.products["persona"] = self.kmeans.fit_predict(self.embeddings)

        # Auto-name each persona using its most distinctive ingredient words
        self.persona_names = {}
        for p in range(self.n_personas):
            mask = self.products["persona"] == p
            if mask.sum() == 0:
                self.persona_names[p] = f"Persona {p}"
                continue
            centroid_tfidf = self.tfidf_matrix[mask.values].mean(axis=0).A1
            top_words = self._feature_names[np.argsort(-centroid_tfidf)[:3]]
            self.persona_names[p] = " / ".join(top_words).title()

    # ------------------------------------------------------------------
    # DIVERSIFICATION HELPER
    # BUG FOUND & FIXED (v1): sorting purely by (Rank desc, Price asc) let
    # tied 5.0-rated products fall back to arbitrary row order, silently
    # crowding out categories -- even though "All categories" was selected.
    #
    # BUG FOUND & FIXED (v2): a plain equal round-robin fix guaranteed every
    # category *appeared*, but gave Moisturizer (this catalog's LARGEST
    # category, ~20% of all products) the exact same 1-slot share as
    # categories a sixth its size -- technically present, but not a
    # genuine "top-rated across the catalog" result either.
    #
    # FINAL FIX: Weighted Round-Robin (the same scheduling algorithm used
    # by network packet schedulers to fairly interleave traffic from
    # sources of different sizes). Each category accumulates "credit"
    # proportional to how many eligible products it has; whichever
    # category has the most accumulated credit gets the next slot, then
    # its credit is spent. Net effect: every eligible category is
    # guaranteed to appear (no starvation), AND larger categories get
    # proportionally more of the results -- Moisturizer legitimately
    # shows up more often than Sun protect when it has ~5x more eligible
    # products, without ever silently dropping to zero.
    # ------------------------------------------------------------------
    def _diversify_by_category(self, df, n):
        if len(df) == 0:
            return df
        groups = {
            label: g.sort_values(["Rank", "Price"], ascending=[False, True]).reset_index(drop=True)
            for label, g in df.groupby("Label")
        }
        categories = list(groups.keys())
        counts = {c: len(groups[c]) for c in categories}
        total_weight = sum(counts.values())

        credits = {c: 0.0 for c in categories}
        pointers = {c: 0 for c in categories}
        picked = []

        while len(picked) < n:
            eligible = [c for c in categories if pointers[c] < counts[c]]
            if not eligible:
                break
            for c in categories:
                credits[c] += counts[c]
            chosen = max(eligible, key=lambda c: credits[c])
            picked.append(groups[chosen].iloc[pointers[chosen]])
            pointers[chosen] += 1
            credits[chosen] -= total_weight

        return pd.DataFrame(picked).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3) RULE-BASED PERSONALIZATION FILTER
    # ------------------------------------------------------------------
    def recommend_for_skin_profile(self, skin_type, category=None, max_price=None, n=10):
        """
        The core 'Recommend for a User' entry point. Since there's no
        purchase history to personalize from, personalization here means:
        the user *tells* the system their skin type / category / budget
        (a structured query) -- this is intentional, and is exactly how a
        first-time visitor to a beauty site is personalized before any
        behavioral data exists (the universal cold-start case).
        """
        assert skin_type in SKIN_TYPES, f"skin_type must be one of {SKIN_TYPES}"
        df = self.products[self.products[skin_type] == 1].copy()

        if category:
            df = df[df["Label"] == category]
        if max_price:
            df = df[df["Price"] <= max_price]

        if len(df) == 0:
            return df

        diversified = self._diversify_by_category(df, n)
        return diversified[DISPLAY_COLS]

    # ------------------------------------------------------------------
    # "Because you liked X" -- content + vector search blended
    # ------------------------------------------------------------------
    def recommend_similar_products(self, product_id, n=10):
        content_scores = cosine_similarity(self.tfidf_matrix[product_id], self.tfidf_matrix).flatten()
        content_scores[product_id] = -1
        content_top = np.argsort(-content_scores)[:50]

        vector_results = dict(self.vector_search(product_id, n=50))

        blended = {}
        for idx in content_top:
            blended[idx] = blended.get(idx, 0) + 0.5 * content_scores[idx]
        for idx, sim in vector_results.items():
            blended[idx] = blended.get(idx, 0) + 0.5 * sim

        ranked = sorted(blended.items(), key=lambda x: -x[1])[:n]
        ids = [i for i, _ in ranked]
        scores = [round(s, 3) for _, s in ranked]

        out = self.products.iloc[ids][DISPLAY_COLS].copy()
        out["match_score"] = scores
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4) MINI-RAG EXPLANATION GENERATOR
    # (retrieval: shared ingredients; generation: template-based text —
    #  documented as a stand-in for a real LLM call)
    # ------------------------------------------------------------------
    def explain_recommendation(self, query_product_id, rec_product_id):
        query_ingredients = set(self._top_ingredients(query_product_id))
        rec_ingredients = set(self._top_ingredients(rec_product_id))
        shared = query_ingredients & rec_ingredients

        rec = self.products.iloc[rec_product_id]
        if shared:
            ing_text = ", ".join(list(shared)[:3])
            return (f"Recommended because it shares key ingredients ({ing_text}) with your selection, "
                    f"is rated {rec['Rank']}/5, and falls in the {rec['Label']} category.")
        else:
            return (f"Recommended because it's a highly-rated ({rec['Rank']}/5) {rec['Label']} "
                    f"with a similar overall formulation profile.")

    def _top_ingredients(self, product_id, top_n=8):
        row = self.tfidf_matrix[product_id].toarray().flatten()
        top_idx = np.argsort(-row)[:top_n]
        return [self._feature_names[i] for i in top_idx if row[i] > 0]

    # ------------------------------------------------------------------
    # Popularity fallback (top-rated overall / per category)
    # ------------------------------------------------------------------
    def trending(self, n=10, category=None):
        df = self.products.copy()
        if category:
            df = df[df["Label"] == category]
            df = df.sort_values("Rank", ascending=False)
            return df.head(n)[DISPLAY_COLS]
        # "All categories" -- diversify so trending isn't accidentally
        # dominated by whichever category has the most 5-star ties (same
        # bug class as recommend_for_skin_profile, fixed the same way).
        diversified = self._diversify_by_category(df, n)
        return diversified[DISPLAY_COLS]


if __name__ == "__main__":
    print("Building GlowMatch recommender...")
    rec = GlowMatchRecommender()

    print(f"\nLoaded {len(rec.products)} products, {rec.products['Brand'].nunique()} brands")
    print("\nDiscovered ingredient personas:")
    for p, name in rec.persona_names.items():
        count = (rec.products["persona"] == p).sum()
        print(f"  Persona {p} ({count} products): {name}")

    print("\n=== Personalized: Dry skin, Moisturizer, under $60 ===")
    print(rec.recommend_for_skin_profile("Dry", category="Moisturizer", max_price=60, n=5))

    sample_id = rec.products[rec.products["Name"].str.contains("Facial Treatment Essence", case=False)].index[0]
    print(f"\n=== Similar to product_id {sample_id} (SK-II Facial Treatment Essence) ===")
    similar = rec.recommend_similar_products(sample_id, n=5)
    print(similar)

    print("\n=== RAG-style explanation for top match ===")
    print(rec.explain_recommendation(sample_id, similar.iloc[0]["product_id"]))
