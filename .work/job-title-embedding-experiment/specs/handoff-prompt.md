# Spec Handoff → Decompose

## What This Step Produced

6 specification documents (00-05) plus an index, covering cross-cutting contracts and all 5 architecture components (C0-C4). All 5 deferred questions from the plan step were resolved.

## Spec Document Locations

| Spec | File | Component |
|------|------|-----------|
| 00 | `.work/job-title-embedding-experiment/specs/00-cross-cutting-contracts.md` | Shared schemas, config, conventions |
| 01 | `.work/job-title-embedding-experiment/specs/01-project-setup.md` | C0: Project Setup |
| 02 | `.work/job-title-embedding-experiment/specs/02-taxonomy-targets.md` | C1: Taxonomy & Targets |
| 03 | `.work/job-title-embedding-experiment/specs/03-test-data.md` | C2: Test Data |
| 04 | `.work/job-title-embedding-experiment/specs/04-matching-pipeline.md` | C3: Matching Pipeline |
| 05 | `.work/job-title-embedding-experiment/specs/05-evaluation-reporting.md` | C4: Evaluation & Reporting |
| Index | `.work/job-title-embedding-experiment/specs/index.md` | Dependency map |

## Dependency Order for Decompose

```
01 (setup)
 ├── 02 (taxonomy) ──→ 03 (test data) ──┐
 └── 04 (matching) ←── 02 ──────────────┤
                                          └── 05 (evaluation)
```

- **Phase 1**: Spec 01 (project setup)
- **Phase 2**: Specs 02 + 04 in parallel (taxonomy targets + matching pipeline code)
- **Phase 3**: Spec 03 (test data — needs taxonomy from 02)
- **Phase 4**: Spec 05 (evaluation — needs test data from 03 and rankings from 04)

Note: Spec 04 code can be written before 02's data exists (it consumes JSON files), but cannot be tested until 02 produces target_sets.json.

## Deferred Questions — Resolutions

| Question | Resolution |
|----------|-----------|
| Cluster subclustering detail | Category-scaffold: <10 roles intact, 10-15→2, 16-25→2-3, 26+→3-4. Hardcoded in `clusters.py`. ~96 total subclusters. Examples for 5 largest categories in spec 02. |
| LLM prompt design | Full system + user prompts in spec 02 (descriptions) and spec 03 (3-pass test data generation). |
| Dev set usage | Pipeline sanity check + threshold exploration only. NOT for metric reporting. Policy in spec 00. |
| Error analysis format | JSON schema with 5 failure modes. 10 worst cases per best model in report. Schema in spec 00, usage in spec 05. |
| Config schema | Full YAML in spec 00: experiment, taxonomy, models (with revision hashes), embedding, test_data, generation, evaluation, report. |

## Key Implementation Details for Decompose

- **Total files to create**: 14 (8 source modules, 3 scripts, 3 test files) + manual.json data file
- **API dependencies**: Claude API needed for `descriptions.py` (84 calls) and `generate_llm.py` (~7 calls)
- **GPU dependency**: Only `embed.py` needs GPU (garden-pop). All other code runs locally.
- **External data**: O*NET synonyms (25 cases in manual.json) sourced from onetonline.org during implementation
- **scipy dependency**: May need to add `scipy` to pyproject.toml for McNemar's test and Friedman test. Evaluate during decompose — or implement the binomial test manually.

## Instructions for Decompose Step

Read this handoff and the spec index. Break specs into executable work items. Key considerations:

1. **Phase ordering**: Setup → taxonomy/matching parallel → test data → evaluation
2. **Manual work items**: manual.json (50 hard cases + 25 O*NET), cluster definitions in clusters.py
3. **API-dependent items**: descriptions.py and generate_llm.py need ANTHROPIC_API_KEY
4. **GPU-dependent items**: Only embed.py (runs on garden-pop)
5. **Stream boundaries**: Consider separating prep work (local, Phases 1-3) from compute work (GPU, Phase 4)
