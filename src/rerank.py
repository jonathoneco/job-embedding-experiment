"""Cross-encoder reranking for two-stage retrieval."""

import numpy as np
from sentence_transformers import CrossEncoder


def load_reranker(config: dict) -> CrossEncoder:
    """Load cross-encoder model from config."""
    return CrossEncoder(
        config["reranking"]["model"],
        trust_remote_code=False,
    )


def rerank(
    reranker: CrossEncoder,
    query: str,
    candidates: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """Re-score candidates with cross-encoder and return top-N.

    Args:
        reranker: CrossEncoder instance.
        query: Query string.
        candidates: [{target_id, score, text}, ...] — top-K from bi-encoder.
            Caller must enrich with "text" field from target dicts.
        top_n: Number of results to return.

    Returns:
        [{target_id, score}, ...] sorted descending by cross-encoder score.
        Scores are in [0.0, 1.0] range (sigmoid output).
    """
    if not candidates:
        raise ValueError("Empty candidates list")

    pairs = [(query, c["text"]) for c in candidates]
    logits = reranker.predict(pairs)
    # Apply sigmoid to convert logits to [0, 1] probabilities
    scores = 1 / (1 + np.exp(-np.array(logits)))

    scored = [
        {"target_id": c["target_id"], "score": float(s)}
        for c, s in zip(candidates, scores)
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


def rerank_batch(
    reranker: CrossEncoder,
    queries: list[str],
    candidates_per_query: list[list[dict]],
    top_n: int = 10,
) -> list[list[dict]]:
    """Rerank candidates for multiple queries.

    Returns list of reranked results per query.
    """
    results = []
    for query, candidates in zip(queries, candidates_per_query):
        if not candidates:
            results.append([])
        else:
            results.append(rerank(reranker, query, candidates, top_n))
    return results
