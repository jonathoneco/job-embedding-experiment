# Stream C: Phase 2 — Reranking + Fusion + Orchestration

**Phase:** 2
**Work Items:** W-05 (JEM-rac), W-06 (JEM-e1x), W-07 (JEM-mjz)
**Dependencies:** Phase 1 complete (Stream A)
**Parallel with:** Stream B

---

## Overview

Create the cross-encoder reranker and RRF fusion modules, then wire Phase 2 orchestration changes into `embed.py` and `config.yaml`.

## File Ownership

| Action | File | Work Item |
|--------|------|-----------|
| Create | `src/rerank.py` | W-05 |
| Create | `src/fusion.py` | W-06 |
| Modify | `config.yaml` | W-05 (reranking section), W-06 (fusion section) |
| Modify | `src/embed.py` | W-07 (granularity loop, reranking flow, fusion call) |

**No overlap with Stream B** (which owns augment.py, targets.py, evaluate.py).

## W-05: Cross-Encoder Reranker (JEM-rac) — spec 04

**Spec:** `.work/job-title-embedding-futures/specs/04-cross-encoder-reranker.md`

**What:** Create `src/rerank.py` with cross-encoder reranking. Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Steps:**
1. `load_reranker(config)` — load `CrossEncoder(model_name)` from config
2. `rerank(reranker, query, candidates, top_n=10)` — form (query, candidate["text"]) pairs, call `reranker.predict()`, apply sigmoid to logits, sort descending, return top-N as `[{target_id, score}]`. Raise `ValueError` if candidates empty.
3. `rerank_batch(reranker, queries, candidates_per_query, top_n=10)` — iterate calling `rerank()` per query. Handle empty candidate lists.
4. Add `reranking` section to `config.yaml`: `enabled: false`, `model`, `initial_k: 20`, `top_n: 10`

**NOTE:** `rerank()` candidates must be enriched with `text` field by the caller (spec 09 step 4 handles this). The `candidates` parameter expects `[{target_id, score, text}, ...]`.

**Acceptance Criteria:**
- `load_reranker()` returns `CrossEncoder` instance
- `rerank()` applies sigmoid, sorts descending, returns top-N
- Empty candidates raises `ValueError`
- `rerank_batch()` processes multiple queries, handles empty lists
- Config section present with `enabled: false` default

**Tests:** Mock CrossEncoder for sorting/top-N; batch processing; empty candidates ValueError.

## W-06: RRF Score Fusion (JEM-e1x) — spec 05

**Spec:** `.work/job-title-embedding-futures/specs/05-rrf-score-fusion.md`

**What:** Create `src/fusion.py` with Reciprocal Rank Fusion (RRF).

**Steps:**
1. `fuse_rankings(rankings_by_method, k=60, top_n=10)` — core RRF: for each query, collect unique target_ids across methods, compute `sum(1/(k + rank))` per target, sort descending, return top_n. Rank is 1-based. Missing targets contribute 0. Raise `ValueError` if method query counts mismatch.
2. `fuse_all(all_rankings, fusion_configs, test_cases, k=60)` — group rankings by (method, granularity), for each fusion config extract and fuse. Skip if method missing (log warning). Return S0.1 format results.
3. Add `fusion` section to `config.yaml`: `enabled: false`, `k: 60`, configs list with `fusion-bge-large-bm25`.

**Acceptance Criteria:**
- RRF computed correctly with 1-based ranks
- Handles 2+ methods, overlapping/disjoint candidates
- `ValueError` on mismatched query counts
- `fuse_all()` groups correctly, skips missing methods with warning
- Config section present with at least one fusion config

**Tests:** Two methods overlapping/disjoint; single method passthrough; mismatched query counts; fuse_all grouping.

## W-07: Phase 2 Orchestration (JEM-mjz) — spec 09 steps 3-6

**Spec:** `.work/job-title-embedding-futures/specs/09-orchestration-updates.md`
**Depends on:** W-05 (JEM-rac), W-06 (JEM-e1x)

**What:** Wire reranking, fusion, and granularity loop changes into `embed.py`.

**Steps:**
1. **Granularity loop** (spec 09 step 3): Add `"role_augmented"` to the hardcoded loop in `run_embedding_model()`. Use `.get()` with `None` check — skip missing granularities.
2. **Reranking integration** (spec 09 step 4): After `rank_targets()`, when `config["reranking"]["enabled"]`, increase `top_k` to `initial_k`, enrich candidates with `text` field from target dicts, call `rerank_batch()`, create separate result entries with method `{model_label}+rerank`. Keep original results too.
3. **Baseline orchestration** (spec 09 step 5): Ensure `run_all_baselines()` results are in `all_rankings` before fusion.
4. **Fusion call** (spec 09 step 6): When `config["fusion"]["enabled"]`, call `fuse_all()` after all individual rankings collected. Append fused results.

**Acceptance Criteria:**
- `role_augmented` in granularity loop with `.get()` guard
- Reranking config-gated, produces `+rerank` method entries
- Candidates enriched with `text` field for reranker
- Baseline results available as fusion inputs
- Fusion runs after all individual methods complete
- All config-gated with `.get()` defaults — no errors when sections absent

**Tests:** Full experiment with features disabled (backward compat); reranking enabled produces +rerank results; fusion produces fusion results.

## Completion

Close issues sequentially as completed. W-05 and W-06 can proceed independently; W-07 runs after both are done.
