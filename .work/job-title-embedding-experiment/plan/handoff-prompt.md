# Plan Handoff → Spec

## What This Step Produced

Architecture document at `.work/job-title-embedding-experiment/specs/architecture.md` defining the full experiment structure: 5 components (C0-C4), data flow, execution model, technology choices, and directory layout.

## Architecture Document Location
`.work/job-title-embedding-experiment/specs/architecture.md`

## Component List for Spec Writing

| Component | Title | Scope | Dependencies |
|-----------|-------|-------|-------------|
| C0 | Project Setup | Dependencies, config, gitignore | None |
| C1 | Taxonomy & Targets | Parse taxonomy, build clusters, generate descriptions, build 5 target sets | C0 |
| C2 | Test Data | 750-case generation (rule-based + LLM + manual), validation, dev/test split | C1 |
| C3 | Matching Pipeline | 3 embedding models + 3 baselines, similarity computation | C0, C1 |
| C4 | Evaluation & Reporting | Metrics, statistics, visualization, markdown report | C2, C3 |

## Key Decisions Made in Planning

1. **Flat modular structure** under `src/` — not a pip-installable package, just importable modules
2. **`uv` for dependency management** — fast, reproducible, lockfile support
3. **LLM-generated descriptions** via Claude API for Role+Desc and Category+Desc granularity levels
4. **Markdown report** (not notebook) — reviewable anywhere, versionable, no rendering issues
5. **Two-phase execution**: prep locally (no GPU), compute on garden-pop (4080 GPU)
6. **Config via `config.yaml`** — seeds, model revision hashes, batch sizes, paths
7. **JSON for all data** — no database, files committed to git (except embedding cache)

## Questions Deferred to Spec

1. **Cluster subclustering detail**: Exact split rules per category — which get split, how many subclusters, what labels
2. **LLM prompt design**: Exact prompts for description generation and test data generation
3. **Dev set usage**: How the 100 dev cases are used (threshold tuning? prompt iteration?)
4. **Error analysis format**: What information per failure case in the report
5. **Config schema**: Exact YAML structure for `config.yaml`

## Instructions for Spec Step

Read this handoff and the architecture document. Write one spec per component (C0-C4) plus a cross-cutting contracts spec (00). Each spec should include:
- Implementation steps with acceptance criteria
- Interface contracts (what each module exposes and consumes)
- Files to create/modify
- Testing strategy

The architecture document has the full data flow, directory structure, and technology choices. The taxonomy source is `job-roles.md` in the project root (692 roles, 42 categories).
