# Stream E: Phase 4 — BGE-M3 Integration (Conditional)

**Phase:** 4
**Work Items:** W-10 (JEM-40z)
**Dependencies:** Phase 3 complete (Stream D)
**Gate condition:** Only pursue if cumulative MRR < 0.75 after Phase 3 evaluation

---

## Overview

Integrate BGE-M3 with all 3 modalities (dense + sparse + ColBERT) via the FlagEmbedding library. Each modality produces its own ranking; fused via RRF (reuses `src/fusion.py` from Phase 2).

**If Phase 3 achieves MRR >= 0.75, this stream is skipped.**

## File Ownership

| Action | File | Work Item |
|--------|------|-----------|
| Create | `src/bgem3.py` | W-10 |
| Modify | `config.yaml` | W-10 (bgem3 section + fusion config) |
| Modify | requirements/dependencies | W-10 (FlagEmbedding) |

## W-10: BGE-M3 Integration (JEM-40z) — spec 08 + spec 09 step 8

**Spec:** `.work/job-title-embedding-futures/specs/08-bgem3-integration.md`
**Also:** `.work/job-title-embedding-futures/specs/09-orchestration-updates.md` step 8

**What:** Full BGE-M3 integration with 3-modality encoding and RRF fusion.

**Steps:**
1. `load_bgem3(config)` — `BGEM3FlagModel(model_name, use_fp16=True)` with all modalities
2. `encode_bgem3(model, texts, batch_size=64)` — returns `{dense: ndarray, sparse: list[dict], colbert: list[ndarray]}`. Dense embeddings L2-normalized.
3. Ranking functions:
   - `rank_bgem3_dense()` — dot product (same as rank_targets)
   - `rank_bgem3_sparse()` — sparse dot product (token overlap)
   - `rank_bgem3_colbert()` — MaxSim (max similarity per query token, averaged)
4. `run_bgem3(config, target_sets, test_cases)` — encode targets+queries per configured granularity, rank with all 3 modalities. Methods: `bgem3-dense`, `bgem3-sparse`, `bgem3-colbert`.
5. Config: `bgem3: {enabled: false, model: "BAAI/bge-m3", granularities: ["role", "category"]}`
6. Add FlagEmbedding dependency (lazy import — no error if not installed when disabled)
7. Orchestration (spec 09 step 8): When enabled, call `run_bgem3()`, add results to `all_rankings`, add `fusion-bgem3-all` fusion config, re-run fusion.

**Acceptance Criteria:**
- All 3 modalities encoded and ranked
- Self-contained module (does NOT use encode_targets/encode_queries from embed.py)
- Methods: `bgem3-dense`, `bgem3-sparse`, `bgem3-colbert`
- Results feed into RRF as `fusion-bgem3-all`
- Lazy import — no FlagEmbedding import when disabled
- Config-gated with `enabled: false` default

**Tests:** Encoding returns all 3 types; each ranking produces correct format; full pipeline on small subset; RRF fusion of 3 modalities.

## Completion

Run full evaluation after integration. Compare BGE-M3 fused results against best Phase 3 result.
