"""
GlowMatch Evaluation
======================
IMPORTANT: This dataset has no user interaction log, so the MovieLens-style
"hide the user's future ratings and check precision/recall" method is not
possible here (there is no future to hide). Instead, this script reports
PROXY METRICS that are honest about what's actually measurable from
catalog-only data.

A note on terminology: none of the metrics below are called "accuracy."
"Accuracy" implies a comparison against a ground-truth user preference,
which does not exist in this dataset. What IS measurable -- and reported
here -- is whether the system's own constraints and outputs are internally
consistent (compliance / satisfaction checks) and whether recommended
items are meaningfully related to the query (relevance / coherence). These
are weaker, more honest claims than "accuracy," and are labeled that way
throughout.

Metrics reported:
1. Skin-Type Filter Compliance  -- of the products returned for a stated
   skin type, what fraction are actually flagged suitable for that skin
   type in the catalog? This is a correctness check on the filter logic,
   not a measure of whether the user will personally like the product.
2. Budget Constraint Satisfaction -- of the products returned under a
   max_price cap, what fraction actually cost at or under that cap? A
   production recommender that violates its own stated budget filter is a
   functional bug, so this is checked explicitly and reported as a hard
   pass/fail per price cap, not only an aggregate percentage.
3. Recommendation Relevance (Content Coherence) -- for "similar products"
   results, how closely related are they to the query product (average
   blended content + vector-search similarity of the returned list)?
   Higher means the returned items are more clearly related in
   formulation; this is a relevance signal, not an accuracy figure.
4. Catalog Coverage -- the percentage of the catalog that ever appears
   across a batch of "similar products" queries, i.e. a check that the
   system isn't just repeatedly recommending the same narrow slice of
   products regardless of query.
5. Cluster Separation (sanity check) -- how the "ingredient persona"
   clusters are distributed in size, as a basic sanity check on the
   unsupervised clustering step (not a formal validation like a
   silhouette score, which is noted as a Future Improvement).
6. Latency -- wall-clock response time per recommendation call, measured
   separately for the personalized filter path and the similarity-search
   path, since they have different computational costs.
"""
import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from recommender import GlowMatchRecommender, SKIN_TYPES

N_SAMPLE_QUERIES = 50
RANDOM_SEED = 42


def evaluate_skin_type_filter_compliance(rec):
    """
    Metric 1 -- Skin-Type Filter Compliance.
    For each skin type, checks that every returned product is actually
    flagged suitable for that skin type in the source catalog. This is
    expected to be 100% by construction (the filter is a direct boolean
    mask on that flag) -- reported as a regression check on the filter
    logic itself, not a measure of recommendation "accuracy."
    """
    print("-" * 55)
    print("1. SKIN-TYPE FILTER COMPLIANCE")
    print("-" * 55)
    compliance_rates = []
    for skin_type in SKIN_TYPES:
        results = rec.recommend_for_skin_profile(skin_type, n=10)
        if len(results) == 0:
            print(f"  [{skin_type:12s}] no results returned -- skipped")
            continue
        actually_flagged = rec.products.loc[results["product_id"], skin_type]
        compliance = actually_flagged.mean()
        compliance_rates.append(compliance)
        status = "PASS" if compliance == 1.0 else "FAIL"
        print(f"  [{skin_type:12s}] {compliance:.2%} correctly flagged  [{status}]")

    overall = np.mean(compliance_rates) if compliance_rates else float("nan")
    print(f"\n  Overall filter compliance: {overall:.2%}")
    print("  (This confirms the personalization filter itself is correct --")
    print("   it is a construction check, not a measure of user taste.)")
    return overall


def evaluate_budget_constraint_satisfaction(rec, price_caps=(10, 25, 50, 75, 100)):
    """
    Metric 2 -- Budget Constraint Satisfaction.
    Confirms that when a user sets max_price, every returned product
    actually costs at or under that amount. Any violation is a functional
    bug (returning something outside the user's stated budget), so this
    is reported as an explicit pass/fail per price cap, not only an
    aggregate percentage.
    """
    print("\n" + "-" * 55)
    print("2. BUDGET CONSTRAINT SATISFACTION")
    print("-" * 55)
    all_satisfied = True
    total_checked = 0
    total_violations = 0

    for cap in price_caps:
        violations_for_cap = 0
        checked_for_cap = 0
        for skin_type in SKIN_TYPES:
            results = rec.recommend_for_skin_profile(skin_type, max_price=cap, n=10)
            if len(results) == 0:
                continue
            over_budget = results[results["Price"] > cap]
            checked_for_cap += len(results)
            violations_for_cap += len(over_budget)

        total_checked += checked_for_cap
        total_violations += violations_for_cap
        status = "PASS" if violations_for_cap == 0 else "FAIL"
        if violations_for_cap:
            all_satisfied = False
        print(f"  max_price=${cap:<4} -> {checked_for_cap} products checked, "
              f"{violations_for_cap} over budget  [{status}]")

    satisfaction_rate = (1 - total_violations / total_checked) if total_checked else float("nan")
    print(f"\n  Budget constraint satisfaction rate: {satisfaction_rate:.2%} "
          f"({total_checked - total_violations}/{total_checked} within budget)")
    print(f"  Overall: {'PASS -- no budget violations found' if all_satisfied else 'FAIL -- see violations above'}")
    return satisfaction_rate


def evaluate_recommendation_relevance(rec, sample_ids):
    """
    Metric 3 -- Recommendation Relevance (Content Coherence).
    For "similar products" results, this measures how closely related the
    returned items are to the query (average blended similarity score).
    This reflects formulation-level relevance, not personal taste
    accuracy, and is reported as a bounded 0-1 score.
    """
    print("\n" + "-" * 55)
    print("3. RECOMMENDATION RELEVANCE (CONTENT COHERENCE)")
    print("-" * 55)
    relevance_scores = []
    for pid in sample_ids:
        similar = rec.recommend_similar_products(pid, n=10)
        relevance_scores.append(similar["match_score"].mean())

    avg_relevance = np.mean(relevance_scores)
    print(f"  Avg relevance score across {len(sample_ids)} sample queries: {avg_relevance:.4f}")
    print("  (0 = unrelated formulation, 1 = near-identical formulation.")
    print("   This is a content-similarity relevance signal, not accuracy.)")
    return avg_relevance


def evaluate_catalog_coverage(rec, sample_ids):
    """
    Metric 4 -- Catalog Coverage.
    Checks what fraction of the catalog is ever surfaced across a batch
    of "similar products" queries, i.e. whether the system leans on a
    narrow, repetitive slice of the catalog or genuinely explores it.
    """
    print("\n" + "-" * 55)
    print("4. CATALOG COVERAGE")
    print("-" * 55)
    all_recommended = set()
    for pid in sample_ids:
        similar = rec.recommend_similar_products(pid, n=10)
        all_recommended.update(similar["product_id"].tolist())

    coverage = len(all_recommended) / len(rec.products)
    print(f"  {len(all_recommended)}/{len(rec.products)} products surfaced "
          f"from {len(sample_ids)} sample queries: {coverage:.2%} coverage")
    return coverage


def evaluate_persona_cluster_sizes(rec):
    """
    Metric 5 -- Cluster Separation (sanity check).
    Reports cluster sizes for the unsupervised "ingredient persona"
    groups. A healthy clustering should not collapse into one dominant
    cluster or produce empty clusters. A formal silhouette-score
    validation is listed as a Future Improvement in DOCUMENTATION.md.
    """
    print("\n" + "-" * 55)
    print("5. INGREDIENT PERSONA CLUSTER SIZES (sanity check)")
    print("-" * 55)
    sizes = rec.products["persona"].value_counts().sort_index()
    print(sizes.to_string())
    return sizes


def evaluate_latency(rec, sample_ids):
    """
    Metric 6 -- Latency.
    Measures wall-clock time per call for the two main recommendation
    paths. Reported in milliseconds since both paths are expected to
    respond well under typical human-perceptible delay (~100ms).
    """
    print("\n" + "-" * 55)
    print("6. LATENCY")
    print("-" * 55)

    similar_latencies = []
    for pid in sample_ids:
        t0 = time.time()
        rec.recommend_similar_products(pid, n=10)
        similar_latencies.append(time.time() - t0)

    filter_latencies = []
    for skin_type in SKIN_TYPES:
        t0 = time.time()
        rec.recommend_for_skin_profile(skin_type, n=10)
        filter_latencies.append(time.time() - t0)

    avg_similar_ms = np.mean(similar_latencies) * 1000
    avg_filter_ms = np.mean(filter_latencies) * 1000
    print(f"  Avg latency -- 'similar products' call:       {avg_similar_ms:.2f} ms")
    print(f"  Avg latency -- personalized-filter call:      {avg_filter_ms:.2f} ms")
    print("  (Uncached, in-process latency. src/api.py additionally reports")
    print("   cache-hit latency, which is 1-2 orders of magnitude faster --")
    print("   see TEST_CASES.md, Test 6.)")
    return {"similar_ms": avg_similar_ms, "filter_ms": avg_filter_ms}


def main():
    print("=" * 55)
    print("GLOWMATCH EVALUATION RESULTS")
    print("=" * 55)

    rec = GlowMatchRecommender()
    sample_ids = np.random.RandomState(RANDOM_SEED).choice(
        len(rec.products), size=N_SAMPLE_QUERIES, replace=False
    )

    results = {
        "skin_type_filter_compliance": evaluate_skin_type_filter_compliance(rec),
        "budget_constraint_satisfaction": evaluate_budget_constraint_satisfaction(rec),
        "recommendation_relevance": evaluate_recommendation_relevance(rec, sample_ids),
        "catalog_coverage": evaluate_catalog_coverage(rec, sample_ids),
    }
    evaluate_persona_cluster_sizes(rec)
    results["latency"] = evaluate_latency(rec, sample_ids)

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Skin-type filter compliance:      {results['skin_type_filter_compliance']:.2%}")
    print(f"  Budget constraint satisfaction:   {results['budget_constraint_satisfaction']:.2%}")
    print(f"  Recommendation relevance:         {results['recommendation_relevance']:.4f}")
    print(f"  Catalog coverage:                 {results['catalog_coverage']:.2%}")
    print(f"  Avg latency (similar / filter):   "
          f"{results['latency']['similar_ms']:.2f} ms / {results['latency']['filter_ms']:.2f} ms")
    print("=" * 55)
    return results


if __name__ == "__main__":
    main()
