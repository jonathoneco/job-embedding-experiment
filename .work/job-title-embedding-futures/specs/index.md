# Spec Index: job-title-embedding-futures

| Spec | Title | Status | Dependencies |
|------|-------|--------|-------------|
| 00 | Cross-cutting contracts | complete | -- |
| 01 | Query Preprocessor (C1) | complete | 00 |
| 02 | Instruction Prefix (C2) | complete | 00 |
| 03 | Target Augmentation (C3) | complete | 00 |
| 04 | Cross-Encoder Reranker (C4) | complete | 00 |
| 05 | RRF Score Fusion (C5) | complete | 00, 04 (references output format) |
| 06 | Training Data Generator (C6) | complete | 00 |
| 07 | Fine-Tuning Pipeline (C7) | complete | 00, 06 (consumes training data) |
| 08 | BGE-M3 Integration (C8) | complete | 00, 05 (reuses fusion) |
| 09 | Orchestration Updates (C9) | complete | 00, all others (cross-cutting) |

## Dependency Ordering

```
Phase 1: 01, 02 (parallel) + 09 partial
Phase 2: 03, 04, 05 (parallel, 05 references 04 output) + 09 partial
Phase 3: 06 → 07 (sequential) + 09 partial
Phase 4: 08 (conditional) + 09 final
```
