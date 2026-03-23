# Research Handoff: job-title-embedding-futures

## What This Step Produced
Comprehensive research across 4 topics covering all 8 improvement techniques, the current codebase architecture, and failure mode analysis. 4 research notes indexed in `research/index.md`.

## Key Findings

### Performance Context
- **Current best**: bge-large@role MRR 0.732, Top-3 0.795
- **Thresholds**: MRR 0.75, Top-3 0.85
- **Gap**: MRR -0.018 (2.4%), Top-3 -0.055 (6.5%)
- **#1 failure mode**: Category confusion (70% of top-10 failures)
- **#2 failure mode**: Abbreviation blindness (20-30% of hard failures)
- **Hard cases are the bottleneck**: 0.638 MRR vs 0.863 easy

### Technique Feasibility Summary

| # | Technique | Complexity | Expected MRR Gain | Addresses |
|---|-----------|-----------|-------------------|-----------|
| 1 | Instruction Prefixing | Trivial (2 lines) | +2-5% | General signal |
| 2 | Abbreviation Dictionary | Simple (new module) | +2-5% | Abbreviation blindness |
| 3 | Target Augmentation | Moderate (new granularity) | +5-20% recall | Alias coverage |
| 4 | Cross-Encoder Reranking | Moderate (new stage) | +5-15% ranking | Category confusion |
| 5 | Hybrid RRF Fusion | Simple (new module) | +3-8% | Surface variants |
| 6 | Contrastive Fine-Tuning | Moderate (training script) | +5-7% | Domain alignment |
| 7 | TSDAE + Fine-Tuning | Moderate (2-stage) | +5-8% combined | Language patterns |
| 8 | BGE-M3 Full Fusion | Moderate (new dep) | +5-8% | Model consolidation |

### Critical Integration Points
- **Query preprocessing** (before encode): abbreviation expansion (technique 2)
- **Query encoding** (prompt kwarg): instruction prefixing (technique 1)
- **Post-ranking** (after rank_targets): cross-encoder reranking (technique 4)
- **Score fusion** (after all rankings): RRF fusion (technique 5)
- **Target sets** (build_target_sets): augmentation adds granularity (technique 3)
- **Training scripts** (new): fine-tuning + TSDAE (techniques 6-7)
- **Model loading** (embed.py): BGE-M3 with FlagEmbedding (technique 8)

### Key Design Decisions for Planning
1. **Abbreviation dict is reversed**: Existing dict maps full→abbrev (for test gen). Query expansion needs abbrev→full. New module, not extension of existing dict.
2. **Target augmentation creates ~3.5-7K targets** (vs 692 for role). Significantly changes embedding cache sizes and encode time.
3. **Cross-encoder reranking requires rank_targets to return top-20** (currently top-10). Config-driven `initial_k` parameter.
4. **RRF is rank-based, not score-based**. Solves the [-1,1] vs [0,1] score incompatibility between embeddings and baselines.
5. **Training data must be separate from test data** to avoid leakage. Generate from taxonomy roles, not test cases.
6. **BGE-M3 dense-only is NOT worth it** — must use all 3 modalities (dense + sparse + ColBERT) to justify the FlagEmbedding dependency.
7. **TSDAE best used as Stage 1** before contrastive fine-tuning, not standalone.
8. **Contrastive fine-tuning vs BGE-M3** are alternative paths to the same goal. Recommend fine-tuning first (more targeted), BGE-M3 as fallback.

### Suggested Phasing for Plan Step
**Phase 1 (Quick Wins)**: Techniques 1-2 (instruction prefix + abbreviation dict) — minimal code, immediate measurable improvement
**Phase 2 (Pipeline Extensions)**: Techniques 3-5 (augmentation + reranking + fusion) — new modules, moderate integration
**Phase 3 (ML Training)**: Techniques 6-7 (contrastive + TSDAE) — new training infrastructure, GPU-dependent
**Phase 4 (Evaluation)**: Technique 8 (BGE-M3) — only if Phase 3 plateaus, new dependency

### Open Questions for Planning
1. Should quick wins (1-2) be evaluated independently before building on them, or implemented together?
2. Does the experiment orchestrator need to support A/B comparison of old vs new pipeline, or just replace?
3. For target augmentation: add as 6th granularity level, or replace existing role granularity with augmented version?
4. Cross-encoder + RRF fusion: are they additive or redundant? (Both improve ranking quality differently)
5. Fine-tuned model: should it replace bge-large in config, or be added as a 4th model for comparison?
6. Training data generation: extend existing generate_rules.py or create dedicated `src/generate_training_data.py`?

## Artifacts
- `research/01-codebase-architecture.md` — Pipeline flow, data contracts, integration points
- `research/02-failure-modes.md` — Performance gaps, failure→technique mapping
- `research/03-next-horizon.md` — 5 quick win techniques with integration details
- `research/04-quarter-horizon.md` — 3 ML techniques with feasibility assessments
- `research/index.md` — Research note index
- `futures.md` — 3 new futures discovered during research
