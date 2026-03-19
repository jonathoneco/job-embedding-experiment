# Spec 00 — Cross-Cutting Contracts

## Overview

Shared schemas, conventions, and configuration consumed by all component specs. Every spec references this document for data formats and file conventions.

---

## File Path Conventions

All paths are relative to project root (`job-embedding-experiment/`). Code references use these constants from `config.yaml` — no hardcoded paths in source.

| Path | Contents | Git-tracked |
|------|----------|-------------|
| `job-roles.md` | Taxonomy source (read-only) | Yes |
| `config.yaml` | Experiment configuration | Yes |
| `data/taxonomy/roles.json` | Parsed taxonomy | Yes |
| `data/taxonomy/clusters.json` | Subclusters | Yes |
| `data/taxonomy/descriptions.json` | LLM-generated descriptions | Yes |
| `data/taxonomy/target_sets.json` | 5 granularity target sets | Yes |
| `data/test-cases/manual.json` | Manual + O*NET cases | Yes |
| `data/test-cases/raw_cases.json` | Pre-validation cases | Yes |
| `data/test-cases/dev.json` | Dev split (100 cases) | Yes |
| `data/test-cases/test.json` | Test split (650 cases) | Yes |
| `data/embeddings/<model-label>/` | Cached embeddings | No (.gitignored) |
| `results/report.md` | Final report | Yes |
| `results/figures/*.png` | Report charts | Yes |
| `results/metrics/*.json` | Raw metric outputs | Yes |

---

## JSON Schemas

### Role (from taxonomy parsing)

```json
{
  "role": "HR Business Partner",
  "category": "Human Resources"
}
```

All 692 roles follow this schema. `role` and `category` are exact strings from `job-roles.md`.

### Test Case

```json
{
  "id": "TC-0001",
  "input_title": "Sr. HRBP",
  "correct_roles": [
    {"role": "HR Business Partner", "category": "Human Resources"}
  ],
  "difficulty": "easy|medium|hard",
  "variation_type": "abbreviation|synonym|creative|jargon|misspelling|cross-category|combined-role|level-prefix",
  "source": "rule-based|llm-systematic|llm-adversarial|manual|onet",
  "notes": "Optional explanation"
}
```

- `id`: Zero-padded sequential ID (`TC-0001` through `TC-0750`)
- `correct_roles`: Accept set — any match counts as correct. Minimum 1 entry.
- `difficulty`: Exactly one of `easy`, `medium`, `hard`
- `variation_type`: Primary variation category (one of the listed values)
- `source`: Generation method

### Target

```json
{
  "id": "T-role-0001",
  "text": "HR Business Partner",
  "role": "HR Business Partner",
  "category": "Human Resources",
  "granularity": "role|role_desc|cluster|category_desc|category"
}
```

- `id`: Prefixed by granularity: `T-role-NNNN`, `T-rdesc-NNNN`, `T-clust-NNNN`, `T-cdesc-NNNN`, `T-cat-NNNN`
- `text`: The string that gets embedded/compared. Varies by granularity:
  - `role`: Role name only → `"HR Business Partner"`
  - `role_desc`: Role + description → `"HR Business Partner: Strategic HR advisor who aligns people strategy with business objectives"`
  - `cluster`: Subcluster label → `"HR Strategy & Business Partnership"`
  - `category_desc`: Category + key terms → `"Human Resources: recruitment, compensation, benefits, employee relations, talent management"`
  - `category`: Category name only → `"Human Resources"`
- For `role` and `role_desc` targets: include `role` (string) and `category` (string) fields.
- For `cluster`, `category_desc`, and `category` targets: the `role` field is **omitted**. Instead, include `roles` (array of role name strings) listing all member roles, and `category` (string).
- For `cluster` targets, additional field `cluster_label` with the subcluster label.

**Cluster/category target example**:
```json
{
  "id": "T-clust-0012",
  "text": "HR Strategy & Business Partnership",
  "roles": ["HR Business Partner", "HR Consultant", "People Analytics Analyst"],
  "category": "Human Resources",
  "cluster_label": "HR Strategy & Business Partnership",
  "granularity": "cluster"
}
```

### Ranking Result

```json
{
  "test_case_id": "TC-0001",
  "method": "bge-base|bge-large|minilm|tfidf|fuzzy|bm25",
  "granularity": "role|role_desc|cluster|category_desc|category",
  "ranked_results": [
    {"target_id": "T-role-0042", "score": 0.92},
    {"target_id": "T-role-0017", "score": 0.87}
  ]
}
```

- `ranked_results`: Sorted descending by score. Top 10 stored.
- `method`: Short label for the method (see Method Labels below)
- Baselines (`tfidf`, `fuzzy`, `bm25`) only produce results for `role` and `category` granularities.

### Metrics Result

```json
{
  "method": "bge-base",
  "granularity": "role",
  "split": "test",
  "metrics": {
    "mrr": 0.847,
    "top1": 0.78,
    "top3": 0.91,
    "top5": 0.95,
    "category_accuracy": 0.93,
    "mean_similarity_gap": 0.045
  },
  "by_difficulty": {
    "easy": {"mrr": 0.95, "top1": 0.92, "top3": 0.98, "top5": 0.99, "category_accuracy": 0.97},
    "medium": {"mrr": 0.82, "top1": 0.74, "top3": 0.89, "top5": 0.93, "category_accuracy": 0.91},
    "hard": {"mrr": 0.65, "top1": 0.52, "top3": 0.76, "top5": 0.82, "category_accuracy": 0.80}
  },
  "bootstrap_ci": {
    "mrr": [0.83, 0.86],
    "top1": [0.76, 0.80],
    "top3": [0.89, 0.93],
    "top5": [0.93, 0.97]
  }
}
```

### Error Analysis Case

```json
{
  "test_case_id": "TC-0042",
  "input_title": "People Ops Lead",
  "expected_roles": [{"role": "People Operations Specialist", "category": "Human Resources"}],
  "rank1": {"target_id": "T-role-0301", "text": "Operations Manager", "category": "Operations", "score": 0.82},
  "rank2": {"target_id": "T-role-0089", "text": "People Operations Specialist", "category": "Human Resources", "score": 0.79},
  "rank3": {"target_id": "T-role-0295", "text": "Operations Specialist", "category": "Operations", "score": 0.77},
  "correct_rank": 2,
  "similarity_gap": 0.03,
  "difficulty": "medium",
  "variation_type": "synonym",
  "failure_mode": "category-confusion"
}
```

Failure modes:
- `near-miss`: Correct category, wrong role (similar role within same category)
- `category-confusion`: Wrong category, semantically related role from another category
- `surface-mismatch`: Very low similarity to correct answer (score < 0.5)
- `rank-displacement`: Correct answer exists but ranked beyond top-5
- `ambiguity`: Multiple plausible categories; system chose a defensible but non-preferred one

---

## Method Labels

| Label | Full Name | Type |
|-------|-----------|------|
| `minilm` | `all-MiniLM-L6-v2` | Embedding |
| `bge-base` | `BAAI/bge-base-en-v1.5` | Embedding |
| `bge-large` | `BAAI/bge-large-en-v1.5` | Embedding |
| `tfidf` | Character n-gram TF-IDF | Baseline |
| `fuzzy` | Jaro-Winkler similarity | Baseline |
| `bm25` | BM25 (word tokenized) | Baseline |

---

## Granularity Labels

| Label | Target Count | Description |
|-------|-------------|-------------|
| `role` | 692 | Role name only |
| `role_desc` | 692 | Role name + one-line description |
| `cluster` | ~96 | Subcluster labels |
| `category_desc` | 42 | Category name + key terms |
| `category` | 42 | Category name only |

---

## Config Schema (`config.yaml`)

```yaml
experiment:
  name: "job-title-embedding-experiment"
  seed: 42

taxonomy:
  source: "job-roles.md"
  output_dir: "data/taxonomy"

models:
  - id: "sentence-transformers/all-MiniLM-L6-v2"
    revision: "<pinned-sha>"
    dim: 384
    label: "minilm"
  - id: "BAAI/bge-base-en-v1.5"
    revision: "<pinned-sha>"
    dim: 768
    label: "bge-base"
  - id: "BAAI/bge-large-en-v1.5"
    revision: "<pinned-sha>"
    dim: 1024
    label: "bge-large"

embedding:
  batch_size: 64
  cache_dir: "data/embeddings"

test_data:
  total: 750
  difficulty:
    easy: 250
    medium: 325
    hard: 175
  dev_size: 100
  output_dir: "data/test-cases"

generation:
  api_model: "claude-sonnet-4-20250514"
  max_tokens: 4096
  description_words: "10-15"

evaluation:
  top_k: [1, 3, 5]
  bootstrap_resamples: 1000
  production_thresholds:
    mrr: 0.75
    top3: 0.85

report:
  output: "results/report.md"
  figures_dir: "results/figures"
  metrics_dir: "results/metrics"
  error_analysis_count: 10
```

The `revision` fields are populated during project setup by querying HuggingFace for the current commit hash of each model.

---

## Error Handling Convention

**Fail-closed.** All modules raise exceptions on error — no silent fallbacks, no partial results, no default values for missing data. The orchestrator scripts (`scripts/*.py`) rely on Python's default exception propagation. If any step fails, the script exits with a traceback.

Specific patterns:
- Missing files → `FileNotFoundError` (Python default)
- Schema validation failures → `ValueError` with descriptive message
- API failures → re-raise `anthropic` exceptions (no retry logic — manual re-run)
- Empty results → `ValueError("No results produced for ...")`

---

## Naming Conventions

- Python modules: `snake_case.py` under `src/`
- JSON data files: `snake_case.json` under `data/`
- Figure files: `<chart-type>_<description>.png` (e.g., `heatmap_mrr_by_model_granularity.png`)
- Metric files: `metrics_<method>_<granularity>.json`
- Test case IDs: `TC-NNNN` (zero-padded to 4 digits)
- Target IDs: `T-<granularity-prefix>-NNNN`

---

## Dev Set Policy

The 100 dev cases are used for:
1. **Pipeline sanity checking**: Verify the full pipeline produces sensible results before running on test set
2. **Threshold exploration**: Examine similarity score distributions to inform production threshold recommendations (exploratory only — not used to tune cutoffs)
3. **Prompt iteration**: During test data generation, validate a small sample against dev set expectations

**Not used for**: Final metric reporting. All metrics in the report are computed on the 650-case test set only. Dev set results are mentioned in the report as a sanity check but never as primary evidence.
