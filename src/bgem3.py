"""BGE-M3 multi-modal embedding for job title matching.

Self-contained module -- does NOT use encode_targets/encode_queries from embed.py.
Uses FlagEmbedding library (separate from sentence-transformers).
"""

import numpy as np

from src.preprocess import expand_abbreviations


def load_bgem3(config: dict):
    """Load BGE-M3 model with all 3 modalities enabled.

    Lazy import to avoid FlagEmbedding dependency when disabled.
    Note: BGEM3FlagModel does not support trust_remote_code or revision
    pinning — FlagEmbedding library limitation. Pin the library version instead.
    """
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(config["bgem3"]["model"], use_fp16=True)


def encode_bgem3(model, texts: list[str], batch_size: int = 64) -> dict:
    """Encode texts with BGE-M3, returning all 3 representations.

    Returns:
        {
            "dense": np.ndarray (n, 1024) -- L2-normalized dense embeddings,
            "sparse": list[dict] -- [{token_id: weight}, ...] sparse vectors,
            "colbert": list[np.ndarray] -- token-level embeddings per text,
        }
    """
    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )
    dense = np.array(output["dense"])
    # L2-normalize dense embeddings
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm dense embedding detected")
    dense = dense / norms

    return {
        "dense": dense,
        "sparse": output["lexical_weights"],
        "colbert": output["colbert_vecs"],
    }


def rank_bgem3_dense(
    query_repr: dict,
    target_repr: dict,
    targets: list[dict],
    top_k: int = 10,
) -> list[list[dict]]:
    """Rank using dense embeddings (dot product of L2-normalized vectors)."""
    sim = query_repr["dense"] @ target_repr["dense"].T
    results = []
    for i in range(len(query_repr["dense"])):
        scores = np.clip(sim[i], -1.0, 1.0)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results.append([
            {"target_id": targets[j]["id"], "score": float(scores[j])}
            for j in top_idx
        ])
    return results


def rank_bgem3_sparse(
    query_repr: dict,
    target_repr: dict,
    targets: list[dict],
    top_k: int = 10,
) -> list[list[dict]]:
    """Rank using sparse (lexical) representations via token overlap."""
    results = []
    for qi in range(len(query_repr["sparse"])):
        q_sparse = query_repr["sparse"][qi]
        scores = []
        for ti in range(len(target_repr["sparse"])):
            t_sparse = target_repr["sparse"][ti]
            # Sparse dot product: sum of products for overlapping tokens
            score = sum(
                q_sparse[tok] * t_sparse[tok]
                for tok in q_sparse
                if tok in t_sparse
            )
            scores.append(score)
        scores_arr = np.array(scores)
        top_idx = np.argsort(scores_arr)[::-1][:top_k]
        results.append([
            {"target_id": targets[j]["id"], "score": float(scores_arr[j])}
            for j in top_idx
        ])
    return results


def rank_bgem3_colbert(
    query_repr: dict,
    target_repr: dict,
    targets: list[dict],
    top_k: int = 10,
) -> list[list[dict]]:
    """Rank using ColBERT late interaction (MaxSim).

    For each query token, find max similarity to any target token,
    then average across query tokens.
    """
    results = []
    for qi in range(len(query_repr["colbert"])):
        q_vecs = query_repr["colbert"][qi]  # (n_query_tokens, dim)
        scores = []
        for ti in range(len(target_repr["colbert"])):
            t_vecs = target_repr["colbert"][ti]  # (n_target_tokens, dim)
            # MaxSim: for each query token, max similarity to any target token
            sim_matrix = q_vecs @ t_vecs.T  # (n_q_tokens, n_t_tokens)
            max_sim = sim_matrix.max(axis=1)  # max per query token
            score = float(max_sim.mean())  # average across query tokens
            scores.append(score)
        scores_arr = np.array(scores)
        top_idx = np.argsort(scores_arr)[::-1][:top_k]
        results.append([
            {"target_id": targets[j]["id"], "score": float(scores_arr[j])}
            for j in top_idx
        ])
    return results


def run_bgem3(
    config: dict,
    target_sets: dict,
    test_cases: list[dict],
) -> list[dict]:
    """Run BGE-M3 with all 3 modalities on configured granularities.

    Returns uniform ranking results (S0.1 format) for each modality.
    """
    model = load_bgem3(config)
    granularities = config["bgem3"].get("granularities", ["role", "category"])
    query_texts = [expand_abbreviations(case["input_title"]) for case in test_cases]

    all_results = []
    for granularity in granularities:
        targets = target_sets.get(granularity)
        if targets is None:
            continue

        target_texts = [t["text"] for t in targets]

        query_repr = encode_bgem3(model, query_texts)
        target_repr = encode_bgem3(model, target_texts)

        modalities = [
            ("bgem3-dense", rank_bgem3_dense),
            ("bgem3-sparse", rank_bgem3_sparse),
            ("bgem3-colbert", rank_bgem3_colbert),
        ]

        for method_name, rank_fn in modalities:
            rankings = rank_fn(query_repr, target_repr, targets)
            for case_idx, case in enumerate(test_cases):
                all_results.append({
                    "test_case_id": case["id"],
                    "method": method_name,
                    "granularity": granularity,
                    "ranked_results": rankings[case_idx],
                })

    return all_results
