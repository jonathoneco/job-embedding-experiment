# Spec Index

| Spec | Title | Status | Dependencies |
|------|-------|--------|-------------|
| 00 | Cross-cutting contracts | complete | — |
| 01 | Project Setup (C0) | complete | 00 |
| 02 | Taxonomy & Targets (C1) | complete | 00, 01 |
| 03 | Test Data (C2) | complete | 00, 02 |
| 04 | Matching Pipeline (C3) | complete | 00, 01, 02 |
| 05 | Evaluation & Reporting (C4) | complete | 00, 03, 04 |

## Dependency Order

```
00 (contracts)
 └── 01 (setup)
      ├── 02 (taxonomy) ──→ 03 (test data) ──┐
      └── 04 (matching) ←── 02 ──────────────┤
                                               └── 05 (evaluation)
```

**Parallel opportunities**:
- Specs 02 and 04 can be implemented in parallel after 01 (04 needs target sets from 02, but the pipeline code can be written before data exists)
- Spec 03 depends on 02 (needs taxonomy for validation)
- Spec 05 depends on both 03 and 04

## Deferred Questions Resolved

| Question | Resolution | Spec |
|----------|-----------|------|
| Cluster subclustering detail | Category-scaffold: <10 roles intact, 10-15→2, 16-25→2-3, 26+→3-4 subclusters. Hardcoded in clusters.py. ~96 total. | 02 |
| LLM prompt design | System + user prompts for descriptions (02), systematic + adversarial test data (03). Full prompts in specs. | 02, 03 |
| Dev set usage | Pipeline sanity check + threshold exploration only. NOT for metric reporting. | 00 |
| Error analysis format | JSON schema with failure modes (near-miss, category-confusion, surface-mismatch, rank-displacement, ambiguity). 10 worst cases in report. | 00, 05 |
| Config schema | Full YAML schema in spec 00 with experiment, taxonomy, models, embedding, test_data, generation, evaluation, report sections. | 00 |
