# Architecture: Job Title Embedding Improvements

## Problem Statement

Current best embedding match (bge-large @ role) achieves MRR 0.732 / Top-3 0.795 against thresholds of MRR 0.75 / Top-3 0.85. The gap is concentrated in hard cases (MRR 0.638) where category confusion (70% of failures) and abbreviation blindness (20-30%) dominate.

## Goals

1. Reach production thresholds: MRR ≥ 0.75, Top-3 ≥ 0.85
2. Maintain backward-compatible evaluation — existing 21 configs remain runnable
3. New techniques appear as additional configs/methods, not replacements
4. All improvements measurable independently via the existing metrics pipeline

## Non-Goals

- Replacing the existing embedding pipeline (additive, not destructive)
- Production serving infrastructure (this is an experiment)
- Cross-language support (deferred to futures)
- Confidence-based routing (deferred to futures)

---

## Architecture Overview

The improvements organize into 4 layers that augment the existing pipeline:

```
                          ┌─────────────────────┐
                          │   Query Input        │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ Query Preprocessing  │  ← NEW: abbreviation expansion (T2)
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ Bi-Encoder Ranking   │  ← MODIFIED: instruction prefix (T1)
                          │ (existing pipeline)  │     + fine-tuned model (T6/T7)
                          └──────────┬──────────┘     + BGE-M3 model (T8)
                                     │
                          ┌──────────▼──────────┐
                          │ Cross-Encoder        │  ← NEW: reranking stage (T4)
                          │ Reranking            │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ Score Fusion (RRF)   │  ← NEW: rank fusion (T5)
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │ Evaluation           │  ← UNCHANGED
                          └─────────────────────┘
```

Supporting infrastructure:

```
  ┌─────────────────────┐    ┌─────────────────────┐
  │ Target Augmentation  │    │ Training Data Gen    │
  │ (T3)                │    │ (T6/T7)              │
  └──────────┬──────────┘    └──────────┬──────────┘
             │                          │
             ▼                          ▼
  ┌─────────────────────┐    ┌─────────────────────┐
  │ data/taxonomy/       │    │ Fine-Tuning Pipeline │
  │ augmented_targets    │    │ (T6: contrastive,   │
  └─────────────────────┘    │  T7: TSDAE stage 1) │
                              └─────────────────────┘
```

---

## Component Map

### C1: Query Preprocessor
**Scope:** Small (new module, ~80 lines)
**Technique:** T2 — Abbreviation Dictionary Expansion
**What:** New `src/preprocess.py` module with `expand_abbreviations(query: str) -> str`. Maintains a reverse dictionary (abbrev → full form) derived from the existing abbreviation rules in `src/generate_rules.py`. Applied before encoding.
**Integration:** Called at the top of `run_embedding_model()` before `encode_queries()`.
**Design decisions:**
- Separate module, not extension of `generate_rules.py` (the existing dict maps full→abbrev for test generation; this needs the reverse)
- Static dictionary loaded once, not per-query computation
- Case-insensitive matching with word-boundary awareness
- Returns original query if no abbreviations found (fail-through, not fail-open — the original query is valid input)

### C2: Instruction Prefix
**Scope:** Trivial (~10 lines of config + 2 lines of code)
**Technique:** T1 — Instruction Prefixing
**What:** Add `prompt` kwarg to `encode_queries()` call in `embed.py`. BGE models expect instruction prefixes like `"Represent this sentence for searching relevant passages: "`. Config-driven per model. Query-side only — targets remain un-prefixed.
**Integration:** Modify `encode_queries()` in `embed.py` to pass `prompt=` from config.
**Design decisions:**
- Instruction strings defined in `config.yaml` per model (only BGE models use them)
- Query-side instruction only; targets remain un-prefixed (research finding: short job title targets perform better without prefix)
- Models without instructions get `prompt=None` (no-op)

### C3: Target Augmentation
**Scope:** Moderate (new module, ~150 lines + data generation)
**Technique:** T3 — Target-Side Augmentation
**What:** New `src/augment.py` module that generates alias variants for each taxonomy role. Creates a new granularity level `role_augmented` with ~3.5-7K targets (vs 692 for role).
**Integration:** Extends `build_target_sets()` in `src/targets.py` to include the new granularity. Each augmented target maps back to its parent role ID for evaluation.
**Design decisions:**
- Add as 6th granularity level (`role_augmented`), not a replacement for `role`. This preserves backward compatibility and allows direct comparison.
- Augmented targets stored in `data/taxonomy/augmented_targets.json`
- Generation is offline (run once, cache result) — uses existing LLM generation patterns from `src/generate_llm.py`
- Each augmented target carries a `role` field (matching role-level schema) for evaluation compatibility with `_is_correct()`. The `role` field maps to the parent role's ID. Spec must define the exact augmented target dict schema.
- `run_embedding_model()` has a hardcoded granularity list (embed.py:~131) that must be extended — covered in C9 orchestration.

### C4: Cross-Encoder Reranker
**Scope:** Moderate (new module, ~120 lines)
**Technique:** T4 — Cross-Encoder Re-Ranking
**What:** New `src/rerank.py` module. Takes bi-encoder top-K candidates, re-scores with `cross-encoder/ms-marco-MiniLM-L-6-v2` (or BGE reranker), returns reranked top-N.
**Integration:** New stage between `rank_targets()` and evaluation. Called as a separate method in the experiment orchestration.
**Design decisions:**
- `rank_targets()` gets a config-driven `initial_k` parameter (default 10 → 20 when reranking is enabled)
- Reranker is a separate method, not embedded in `rank_targets()` — keeps bi-encoder ranking pure
- Appears as a new "method" in results (e.g., `bge-large+rerank`) so existing evaluation pipeline handles it
- Model loaded once, reused across queries (same pattern as bi-encoder model loading)

### C5: RRF Score Fusion
**Scope:** Small (new module, ~80 lines)
**Technique:** T5 — Hybrid Dense+Sparse Score Fusion
**What:** New `src/fusion.py` module implementing Reciprocal Rank Fusion. Combines rankings from multiple methods (e.g., embedding + BM25 + cross-encoder) into a single fused ranking.
**Integration:** Post-processing step that takes multiple ranking results for the same test cases and produces a fused ranking. Appears as a new method in results.
**Design decisions:**
- RRF is rank-based, not score-based — solves the [-1,1] vs [0,1] score incompatibility
- Standard RRF formula: `score(d) = Σ 1/(k + rank_i(d))` with k=60 (standard constant)
- Fusion configs defined in `config.yaml` specifying which methods to fuse
- Each fusion config produces a new method entry in results
- Cross-encoder + RRF are additive, not redundant: cross-encoder improves individual rankings that RRF then combines
- Primary fusion use case: dense (bge-large) + sparse (TF-IDF/BM25 from existing `baselines.py`). Existing baselines already produce rankings in the uniform format — C9 orchestration ensures they're available as fusion inputs

### C6: Training Data Generator
**Scope:** Moderate (new script, ~200 lines)
**Technique:** T6/T7 prerequisite
**What:** New `src/generate_training_data.py` that generates (noisy_title, canonical_role) pairs for contrastive training plus a job title corpus for TSDAE.
**Integration:** Standalone script, run before training. Outputs to `data/training/`.
**Design decisions:**
- Dedicated script, not extension of `generate_rules.py` — different output format (training pairs vs test cases), different intent (training vs evaluation)
- Training data generated from taxonomy roles only — NOT from test cases (prevents data leakage)
- ~5K pairs: 692 roles × 7-8 variants each, using the same transform rules as test generation but with different seeds/samples
- Hard negative mining: for each anchor, include roles from different categories with similar surface forms
- Output formats: `data/training/pairs.jsonl` (contrastive) and `data/training/corpus.txt` (TSDAE)

### C7: Fine-Tuning Pipeline
**Scope:** Moderate (new script, ~250 lines)
**Technique:** T6 — Contrastive Fine-Tuning, T7 — TSDAE
**What:** New `src/fine_tune.py` that fine-tunes bge-large-en-v1.5 using sentence-transformers training API. Two-stage: optional TSDAE pre-training (Stage 1) then contrastive fine-tuning (Stage 2).
**Integration:** Standalone training script. Produces a model directory that gets added to `config.yaml` as a new model.
**Design decisions:**
- Fine-tuned model added as 4th model in config (not replacement) — allows direct A/B comparison with base bge-large
- Stage 1 (TSDAE): DenoisingAutoEncoderLoss, 10 epochs, ~60-90 min on RTX 4080
- Stage 2 (Contrastive): MultipleNegativesRankingLoss, 3 epochs, ~15-20 min on RTX 4080
- Hyperparameters: lr=2e-5 with warmup (prevents catastrophic forgetting)
- Dev set used for hyperparameter tuning; test set for final evaluation only
- Model saved to `models/bge-large-finetuned/`

### C8: BGE-M3 Integration
**Scope:** Moderate (new module, ~200 lines + new dependency)
**Technique:** T8 — BGE-M3 Full Fusion
**What:** Integration of BGE-M3 model with all 3 modalities (dense + sparse + ColBERT). Requires FlagEmbedding library.
**Integration:** New encoding path in experiment pipeline. Produces 3 separate rankings that get fused via RRF (reuses C5).
**Design decisions:**
- Only pursue if Phase 3 (fine-tuning) doesn't hit thresholds — this is the fallback path
- Must use all 3 modalities (dense + sparse + ColBERT) to justify FlagEmbedding dependency
- Dense-only BGE-M3 is NOT worth the switch (research finding)
- New dependency: `FlagEmbedding` (not `sentence-transformers` — different API)
- Each modality produces its own ranking; fused via RRF from C5

### C9: Experiment Orchestration Updates
**Scope:** Small (~50 lines of config changes)
**Technique:** Cross-cutting
**What:** Updates to `config.yaml` and the main experiment runner to support new methods, granularities, and pipeline stages.
**Integration:** Modifies existing experiment flow to optionally invoke preprocessing, reranking, and fusion stages.
**Design decisions:**
- New pipeline stages are opt-in via config — existing 21 configs remain unchanged
- New configs added for: instruction-prefixed models, reranked results, fused results, fine-tuned model, augmented targets
- No A/B comparison infrastructure needed — the experiment already runs all configs and compares via metrics. New techniques just add more configs.
- Hardcoded granularity loop in `run_embedding_model()` (embed.py:~131) must be extended to include `role_augmented`
- Baseline rankings from `baselines.py` must be orchestrated as fusion inputs for C5

---

## Resolved Open Questions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Evaluate quick wins independently or together? | Together in Phase 1, but as separate configs | Each technique creates its own method/config entry in results, so they're independently measurable even when implemented together |
| 2 | A/B comparison support? | Not needed | Existing experiment runs all configs and compares. New techniques = new configs, not replacements |
| 3 | Target augmentation: 6th level or replacement? | 6th granularity level (`role_augmented`) | Backward compatibility + direct comparison with base `role` |
| 4 | Cross-encoder + RRF additive or redundant? | Additive | Cross-encoder improves individual ranking quality; RRF combines multiple rankings. They work at different levels |
| 5 | Fine-tuned model: replace or add? | Add as 4th model | Direct A/B comparison with base bge-large on same test set |
| 6 | Training data: extend generate_rules.py or new script? | New `src/generate_training_data.py` | Different output format, different intent (training vs test gen), different data leakage constraints |

## Deferred Questions for Spec

1. Exact abbreviation dictionary contents — how many entries, sourced from where?
2. Instruction prefix exact strings per model (query-side vs target-side)
3. Target augmentation prompt template and expected output format
4. Cross-encoder model selection: ms-marco-MiniLM vs bge-reranker-base
5. RRF k constant: 60 (standard) or tuned on dev set?
6. Fine-tuning hyperparameter grid (lr, batch size, epochs, warmup ratio)
7. TSDAE noise ratio and epoch count
8. Exact config.yaml schema changes for new pipeline stages

---

## Phase Dependency Graph

```
Phase 1 (Quick Wins)
├── C1: Query Preprocessor (T2)         ─┐
├── C2: Instruction Prefix (T1)          │ No dependencies
└── C9: Orchestration (partial)          ─┘

Phase 2 (Pipeline Extensions) — evaluation-sequenced after Phase 1 (no code dependency)
├── C3: Target Augmentation (T3)         ─┐
├── C4: Cross-Encoder Reranker (T4)       │ Independent of each other
├── C5: RRF Score Fusion (T5)            ─┘ (C5 can fuse C4 output)
└── C9: Orchestration (remaining)

Phase 3 (ML Training) — evaluation-sequenced after Phase 2
├── C6: Training Data Generator          ─┐ C7 depends on C6 (code dep)
└── C7: Fine-Tuning Pipeline             ─┘

Phase 4 (Evaluation / Fallback) — gate: only if cumulative MRR < 0.75 after Phase 3
└── C8: BGE-M3 Integration              (depends on C5 for fusion)
```

**Critical path:** C1+C2 → evaluate → C3+C4+C5 → evaluate → C6→C7 → evaluate → C8 (conditional)

Each phase ends with a full evaluation run to measure cumulative improvement and decide whether to proceed.

---

## Technology Choices

| Choice | Technology | Rationale |
|--------|-----------|-----------|
| Cross-encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Well-benchmarked, fast inference, available via sentence-transformers |
| Training loss | `MultipleNegativesRankingLoss` | Standard for contrastive fine-tuning, efficient in-batch negatives |
| Domain adaptation | `DenoisingAutoEncoderLoss` (TSDAE) | Built into sentence-transformers, unsupervised, compatible with BERT-based BGE |
| Score fusion | RRF (k=60) | Rank-based (scale-agnostic), well-studied, no hyperparameter sensitivity |
| BGE-M3 library | `FlagEmbedding` | Only way to access sparse + ColBERT modalities (sentence-transformers only does dense) |
| Training framework | `sentence-transformers` 3.4.1 | Already installed, full training API including SentenceTransformerTrainer |

## Scope Exclusions

- No changes to evaluation metrics or test data
- No production deployment infrastructure
- No hyperparameter search automation (manual tuning on dev set)
- No incremental evaluation mode (deferred to futures)
- No cross-language support (deferred to futures)
- No confidence-based routing (deferred to futures)
