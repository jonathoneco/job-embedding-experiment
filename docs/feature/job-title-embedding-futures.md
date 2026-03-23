# Job Title Embedding — Full Futures Implementation

**Status:** In Progress
**Epic:** JEM-arn
**Tier:** 3 (Initiative)

## What

Implement 8 improvement techniques across 4 phases to push job title embedding matching from MRR 0.732/Top-3 0.795 toward production thresholds (MRR 0.75, Top-3 0.85). Techniques span query preprocessing, two-stage retrieval, score fusion, and model fine-tuning — all additive to the existing pipeline with independent measurability.

## Components

| # | Component | Technique | Scope |
|---|-----------|-----------|-------|
| C1 | Query Preprocessor | Abbreviation expansion | Small (~80 lines) |
| C2 | Instruction Prefix | BGE prompt kwarg | Trivial (~10 lines) |
| C3 | Target Augmentation | LLM-generated aliases | Moderate (~150 lines) |
| C4 | Cross-Encoder Reranker | Two-stage retrieval | Moderate (~120 lines) |
| C5 | RRF Score Fusion | Rank-based fusion | Small (~80 lines) |
| C6 | Training Data Generator | Synthetic pairs + corpus | Moderate (~200 lines) |
| C7 | Fine-Tuning Pipeline | Contrastive + TSDAE | Moderate (~250 lines) |
| C8 | BGE-M3 Integration | Dense+sparse+ColBERT | Moderate (~200 lines) |
| C9 | Orchestration Updates | Config + pipeline flow | Small (~50 lines) |

## Phases

1. **Quick Wins** (C1, C2, C9-partial): Instruction prefixing + abbreviation expansion
2. **Pipeline Extensions** (C3, C4, C5, C9-remaining): Augmentation + reranking + fusion
3. **ML Training** (C6, C7): Training data generation + fine-tuning
4. **Evaluation/Fallback** (C8): BGE-M3 — only if Phase 3 insufficient

## Key Decisions (from specs)

1. **Abbreviation dict**: 14 entries reversed from `generate_rules._ABBREVIATIONS`; compiled regex, word-boundary matching
2. **Instruction prefix**: `"Represent this sentence for searching relevant passages: "` for BGE models; query-side only
3. **Target augmentation**: LLM generates 5-10 aliases per role; `role_augmented` granularity with `role` field for eval compatibility
4. **Cross-encoder**: `ms-marco-MiniLM-L-6-v2` (22M params, sufficient for short text pairs); top-20 input, top-10 output
5. **RRF constant**: k=60 (standard, not tuned); primary use case: dense+sparse fusion on role granularity
6. **Fine-tuning**: 2-stage (TSDAE 10 epochs + contrastive 3 epochs); lr=2e-5, batch_size=32, warmup=10%
7. **TSDAE**: noise_ratio=0.6 (standard); corpus from taxonomy variants (~5.7K sentences)
8. **Config schema**: New `reranking`, `fusion`, `bgem3` sections all default disabled for backward compat
