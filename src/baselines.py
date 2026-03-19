"""Baseline matching methods: TF-IDF, fuzzy, and BM25."""

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def run_tfidf(
    targets: list[dict],
    test_cases: list[dict],
    granularity: str,
) -> list[dict]:
    """Rank targets using TF-IDF character n-gram similarity."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    target_texts = [t["text"] for t in targets]
    query_texts = [c["input_title"] for c in test_cases]

    target_matrix = vectorizer.fit_transform(target_texts)
    query_matrix = vectorizer.transform(query_texts)
    sim_matrix = cosine_similarity(query_matrix, target_matrix)

    results = []
    for i, case in enumerate(test_cases):
        scores = np.clip(sim_matrix[i], 0.0, 1.0)
        top_indices = np.argsort(scores)[::-1][:10]
        ranked = [
            {"target_id": targets[j]["id"], "score": float(scores[j])}
            for j in top_indices
        ]

        for r in ranked:
            if not (0.0 <= r["score"] <= 1.0):
                raise ValueError(f"TF-IDF score {r['score']} out of [0, 1] range")
        for k in range(len(ranked) - 1):
            if ranked[k]["score"] < ranked[k + 1]["score"]:
                raise ValueError("TF-IDF results not sorted descending by score")

        results.append({
            "test_case_id": case["id"],
            "method": "tfidf",
            "granularity": granularity,
            "ranked_results": ranked,
        })
    return results


def run_fuzzy(
    targets: list[dict],
    test_cases: list[dict],
    granularity: str,
) -> list[dict]:
    """Rank targets using Jaro-Winkler fuzzy string similarity."""
    results = []
    for case in test_cases:
        query = case["input_title"]
        scores = np.array([
            JaroWinkler.similarity(query, t["text"])
            for t in targets
        ])
        top_indices = np.argsort(scores)[::-1][:10]
        ranked = [
            {"target_id": targets[j]["id"], "score": float(scores[j])}
            for j in top_indices
        ]

        for r in ranked:
            if not (0.0 <= r["score"] <= 1.0):
                raise ValueError(f"Fuzzy score {r['score']} out of [0, 1] range")
        for k in range(len(ranked) - 1):
            if ranked[k]["score"] < ranked[k + 1]["score"]:
                raise ValueError("Fuzzy results not sorted descending by score")

        results.append({
            "test_case_id": case["id"],
            "method": "fuzzy",
            "granularity": granularity,
            "ranked_results": ranked,
        })
    return results


def run_bm25(
    targets: list[dict],
    test_cases: list[dict],
    granularity: str,
) -> list[dict]:
    """Rank targets using BM25 with whitespace tokenization."""
    tokenized_targets = [t["text"].lower().split() for t in targets]
    bm25 = BM25Okapi(tokenized_targets)

    results = []
    for case in test_cases:
        tokenized_query = case["input_title"].lower().split()
        scores = bm25.get_scores(tokenized_query)
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score  # Normalize to [0, 1]
        top_indices = np.argsort(scores)[::-1][:10]
        ranked = [
            {"target_id": targets[j]["id"], "score": float(scores[j])}
            for j in top_indices
        ]

        for r in ranked:
            if not (0.0 <= r["score"] <= 1.0):
                raise ValueError(f"BM25 score {r['score']} out of [0, 1] range")
        for k in range(len(ranked) - 1):
            if ranked[k]["score"] < ranked[k + 1]["score"]:
                raise ValueError("BM25 results not sorted descending by score")

        results.append({
            "test_case_id": case["id"],
            "method": "bm25",
            "granularity": granularity,
            "ranked_results": ranked,
        })
    return results


def run_all_baselines(
    target_sets: dict,
    test_cases: list[dict],
) -> list[dict]:
    """Run all baseline methods on role and category granularities."""
    results = []
    for granularity in ["role", "category"]:
        targets = target_sets[granularity]
        results.extend(run_tfidf(targets, test_cases, granularity))
        results.extend(run_fuzzy(targets, test_cases, granularity))
        results.extend(run_bm25(targets, test_cases, granularity))
    return results
