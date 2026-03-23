# Spec Handoff: job-title-embedding-futures

## What This Step Produced

10 specification documents (00-09) defining exact interface contracts, acceptance criteria, file lists, and testing strategies for all 9 components plus cross-cutting contracts.

## Spec Index

| Spec | Component | New Files | Modified Files | Phase |
|------|-----------|-----------|---------------|-------|
| 00 | Cross-cutting contracts | -- | -- | -- |
| 01 | Query Preprocessor (C1) | `src/preprocess.py` | `src/embed.py` | 1 |
| 02 | Instruction Prefix (C2) | -- | `src/embed.py`, `config.yaml` | 1 |
| 03 | Target Augmentation (C3) | `src/augment.py`, `data/taxonomy/augmented_targets.json` | `src/targets.py`, `src/evaluate.py` | 2 |
| 04 | Cross-Encoder Reranker (C4) | `src/rerank.py` | `config.yaml` | 2 |
| 05 | RRF Score Fusion (C5) | `src/fusion.py` | `config.yaml` | 2 |
| 06 | Training Data Generator (C6) | `src/generate_training_data.py`, `data/training/` | -- | 3 |
| 07 | Fine-Tuning Pipeline (C7) | `src/fine_tune.py`, `models/bge-large-finetuned/` | `config.yaml` | 3 |
| 08 | BGE-M3 Integration (C8) | `src/bgem3.py` | `config.yaml`, dependencies | 4 |
| 09 | Orchestration Updates (C9) | -- | `config.yaml`, `src/embed.py`, `src/evaluate.py` | All |

## Resolved Deferred Questions

| # | Question | Resolution |
|---|----------|-----------|
| 1 | Abbreviation dict contents | 14 entries reversed from `_ABBREVIATIONS` in `generate_rules.py` |
| 2 | Instruction prefix strings | BGE: `"Represent this sentence for searching relevant passages: "` (query-side only) |
| 3 | Augmentation prompt | LLM generates 5-10 aliases per role, batched by category (~42 API calls) |
| 4 | Cross-encoder model | `ms-marco-MiniLM-L-6-v2` — 22M params, fast, sufficient for short text |
| 5 | RRF k constant | k=60 (standard, not tuned) |
| 6 | Fine-tuning hyperparams | lr=2e-5, batch_size=32, epochs=3, warmup_ratio=0.1 |
| 7 | TSDAE params | noise_ratio=0.6, epochs=10 |
| 8 | Config schema | New `reranking`, `fusion`, `bgem3` sections, all default disabled |

## Key Interface Contracts

### Shared schemas (spec 00)
- Uniform ranking result: `{test_case_id, method, granularity, ranked_results: [{target_id, score}]}`
- Method naming: `model_label`, `model_label+rerank`, `fusion-x-y`, `bgem3-{modality}`
- Target schemas per granularity (including new `role_augmented` with `role` field)

### Critical integration points
- `encode_queries()` gains `prompt: str | None = None` parameter (spec 02)
- `run_embedding_model()` granularity loop gains `role_augmented` with `.get()` guard (spec 09)
- `_is_correct()` gains `"role_augmented"` in role-matching branch (spec 03)
- `load_model()` handles `revision=None` for local models (spec 09)
- Reranking and fusion are config-gated, run after all individual rankings collected (spec 09)

## Phase Dependency Graph for Decompose

```
Phase 1: specs 01 + 02 (parallel) + 09-partial
  No code dependencies. Both touch embed.py but different parts.

Phase 2: specs 03 + 04 + 05 (parallel) + 09-partial
  No code dependencies between 03, 04, 05.
  05 references 04's output format but doesn't depend on 04's code.
  09 orchestration wires them together.

Phase 3: spec 06 → 07 (sequential)
  07 consumes 06's output files.

Phase 4: spec 08 (conditional — only if MRR < 0.75 after Phase 3)
  Reuses 05's fusion module.
```

## Instructions for Decompose Step

Break specs into work items (beads issues). Key considerations:
- Phase 1 specs (01, 02) can be a single stream — both are small and touch embed.py
- Phase 2 specs (03, 04, 05) can be 2-3 streams — 03 is independent; 04+05 share ranking concerns
- Phase 3 specs (06, 07) are sequential — one stream
- Spec 09 orchestration changes are distributed across phases — split into per-phase work items
- Each work item should have clear file ownership (no file conflicts within a phase)

## Futures Discovered

No new futures discovered during spec writing.
