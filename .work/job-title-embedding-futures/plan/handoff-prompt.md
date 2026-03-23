# Plan Handoff: job-title-embedding-futures

## What This Step Produced

Architecture document at `.work/job-title-embedding-futures/specs/architecture.md` defining 9 components across 4 phases, resolving all 6 open questions from research, and establishing the dependency graph for implementation.

## Architecture Document Location

`.work/job-title-embedding-futures/specs/architecture.md`

## Component List for Spec Writing

| # | Component | New Files | Modified Files | Phase |
|---|-----------|-----------|---------------|-------|
| C1 | Query Preprocessor | `src/preprocess.py` | `src/embed.py` (call site) | 1 |
| C2 | Instruction Prefix | — | `src/embed.py`, `config.yaml` | 1 |
| C3 | Target Augmentation | `src/augment.py`, `data/taxonomy/augmented_targets.json` | `src/targets.py` | 2 |
| C4 | Cross-Encoder Reranker | `src/rerank.py` | `src/embed.py` (initial_k param) | 2 |
| C5 | RRF Score Fusion | `src/fusion.py` | — | 2 |
| C6 | Training Data Generator | `src/generate_training_data.py`, `data/training/` | — | 3 |
| C7 | Fine-Tuning Pipeline | `src/fine_tune.py`, `models/bge-large-finetuned/` | `config.yaml` | 3 |
| C8 | BGE-M3 Integration | `src/bgem3.py` | `config.yaml` | 4 |
| C9 | Orchestration Updates | — | `config.yaml`, main runner | All |

## Key Design Decisions (from architecture)

1. All techniques are additive — existing 21 configs unchanged
2. New techniques appear as new configs/methods for independent measurement
3. Target augmentation is 6th granularity (`role_augmented`), not replacement
4. Fine-tuned model is 4th model, not replacement for bge-large
5. Training data generated from taxonomy (not test cases) to prevent leakage
6. Cross-encoder + RRF are additive (different levels of ranking improvement)
7. BGE-M3 is Phase 4 fallback only — requires all 3 modalities to justify dependency
8. Each phase ends with full evaluation to measure cumulative improvement

## Deferred Questions for Spec Step

1. Exact abbreviation dictionary contents — how many entries, sourced from where?
2. Instruction prefix exact strings per model (query-side vs target-side)
3. Target augmentation prompt template and expected output format
4. Cross-encoder model selection: ms-marco-MiniLM vs bge-reranker-base
5. RRF k constant: 60 (standard) or tuned on dev set?
6. Fine-tuning hyperparameter grid (lr, batch size, epochs, warmup ratio)
7. TSDAE noise ratio and epoch count
8. Exact config.yaml schema changes for new pipeline stages

## Instructions for Spec Step

Write one spec per component (C1-C9), plus a cross-cutting contracts spec (00). Each spec should:
- Define exact interface contracts (function signatures, data formats)
- List acceptance criteria per implementation step
- Specify files to create/modify
- Define testing strategy
- Reference the architecture for context

Dependency ordering for specs:
- C9 (orchestration) is cross-cutting — spec first or alongside C1/C2
- C1, C2 can be specced in parallel (no dependency)
- C3, C4, C5 can be specced in parallel (C5 may reference C4 output format)
- C6 must be specced before C7 (C7 consumes C6 output)
- C8 can be specced independently but should reference C5 (reuses fusion)

## Futures Discovered

No new futures discovered during planning. Existing futures from research remain:
- Evaluation Test Harness Improvements (next)
- Cross-Language Title Support (quarter)
- Confidence-Based Routing (someday)
