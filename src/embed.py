"""Embedding engine for job title matching."""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from src.preprocess import expand_abbreviations


def load_model(model_config: dict) -> SentenceTransformer:
    """Load a SentenceTransformer model from config.

    Revision hashes are pinned in config.yaml for reproducibility and
    supply-chain safety. Local fine-tuned models use revision: null.
    """
    kwargs = {"trust_remote_code": False}
    if model_config.get("revision"):
        kwargs["revision"] = model_config["revision"]
    return SentenceTransformer(model_config["id"], **kwargs)


def encode_targets(
    model: SentenceTransformer,
    targets: list[dict],
    batch_size: int,
    cache_dir: str,
    model_label: str,
) -> np.ndarray:
    """Encode target texts with L2 normalization and caching."""
    if os.sep in model_label or '/' in model_label:
        raise ValueError(f"Invalid model_label: {model_label}")

    granularity = targets[0]["granularity"]
    cache_path = os.path.join(cache_dir, model_label, f"targets_{granularity}.npy")
    expected_dim = model.get_sentence_embedding_dimension()
    expected_shape = (len(targets), expected_dim)

    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if cached.shape == expected_shape:
            return cached

    texts = [t["text"] for t in targets]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding vector detected — input text may be empty")
    embeddings = embeddings / norms

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, got {embeddings.shape}"
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)

    return embeddings


def encode_queries(
    model: SentenceTransformer,
    queries: list[str],
    batch_size: int,
    prompt: str | None = None,
) -> np.ndarray:
    """Encode query strings with L2 normalization. No caching."""
    embeddings = model.encode(
        queries, batch_size=batch_size, show_progress_bar=False, prompt=prompt,
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding vector detected — input text may be empty")
    embeddings = embeddings / norms

    expected_dim = model.get_sentence_embedding_dimension()
    if embeddings.shape != (len(queries), expected_dim):
        raise ValueError(
            f"Expected shape ({len(queries)}, {expected_dim}), got {embeddings.shape}"
        )

    return embeddings


def rank_targets(
    query_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    targets: list[dict],
    top_k: int = 10,
) -> list[list[dict]]:
    """Rank targets by cosine similarity (dot product of L2-normalized vectors)."""
    similarity_matrix = query_embeddings @ target_embeddings.T  # (n_queries, n_targets)

    results = []
    for i in range(len(query_embeddings)):
        scores = np.clip(similarity_matrix[i], -1.0, 1.0)
        top_indices = np.argsort(scores)[::-1][:top_k]
        ranked = [
            {"target_id": targets[j]["id"], "score": float(scores[j])}
            for j in top_indices
        ]

        # Validate scores are in valid cosine similarity range for normalized vectors
        for r in ranked:
            if not (-1.0 <= r["score"] <= 1.0):
                raise ValueError(
                    f"Score {r['score']} out of [-1, 1] range"
                )
        # Validate descending order
        for k in range(len(ranked) - 1):
            if ranked[k]["score"] < ranked[k + 1]["score"]:
                raise ValueError(
                    "Results not sorted descending by score"
                )

        results.append(ranked)
    return results


def run_embedding_model(
    model_config: dict,
    target_sets: dict,
    test_cases: list[dict],
    config: dict,
) -> list[dict]:
    """Run a single embedding model across all granularities."""
    model = load_model(model_config)
    query_texts = [case["input_title"] for case in test_cases]
    query_texts = [expand_abbreviations(q) for q in query_texts]
    batch_size = config["embedding"]["batch_size"]
    cache_dir = config["embedding"]["cache_dir"]
    model_label = model_config["label"]
    instruction = model_config.get("instruction")

    reranking_config = config.get("reranking", {})
    reranking_enabled = reranking_config.get("enabled", False)
    reranker = None
    rerank_batch = None
    if reranking_enabled:
        from src.rerank import load_reranker
        from src.rerank import rerank_batch as _rerank_batch
        reranker = load_reranker(config)
        rerank_batch = _rerank_batch

    all_results = []
    for granularity in target_sets:
        targets = target_sets.get(granularity)
        if targets is None:
            continue

        # Targets intentionally get no instruction prefix — BGE-v1.5 applies
        # instruction only to queries, not passages (per model documentation).
        target_emb = encode_targets(model, targets, batch_size, cache_dir, model_label)
        query_emb = encode_queries(model, query_texts, batch_size, prompt=instruction)
        top_k = reranking_config.get("initial_k", 10) if reranking_enabled else 10
        rankings = rank_targets(query_emb, target_emb, targets, top_k=top_k)

        for case_idx, case in enumerate(test_cases):
            all_results.append({
                "test_case_id": case["id"],
                "method": model_label,
                "granularity": granularity,
                "ranked_results": rankings[case_idx],
            })

        if reranking_enabled and reranker is not None:
            target_lookup = {t["id"]: t["text"] for t in targets}
            candidates_per_query = []
            for ranking in rankings:
                enriched = [
                    {**r, "text": target_lookup[r["target_id"]]}
                    for r in ranking
                ]
                candidates_per_query.append(enriched)

            reranked = rerank_batch(
                reranker, query_texts, candidates_per_query,
                top_n=reranking_config.get("top_n", 10),
            )

            for case_idx, case in enumerate(test_cases):
                all_results.append({
                    "test_case_id": case["id"],
                    "method": f"{model_label}+rerank",
                    "granularity": granularity,
                    "ranked_results": reranked[case_idx],
                })

    fusion_config = config.get("fusion", {})
    if fusion_config.get("enabled", False):
        from src.fusion import fuse_all
        fused = fuse_all(
            all_results,
            fusion_config["configs"],
            test_cases,
            k=fusion_config.get("k", 60),
        )
        all_results.extend(fused)

    return all_results
