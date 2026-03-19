# Spec 01 — Project Setup (C0)

**Dependencies**: None
**Refs**: Spec 00 (config schema, file paths)

---

## Overview

Initialize the project directory, dependency management, configuration, and gitignore. This component produces no runtime code — it creates the scaffold all other components build on.

---

## Files to Create

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies and project metadata |
| `config.yaml` | Experiment configuration (full schema in spec 00) |
| `.gitignore` | Exclude caches, embeddings, __pycache__ |
| `src/__init__.py` | Empty — makes src a package for relative imports |
| `tests/__init__.py` | Empty — makes tests discoverable by pytest |

---

## Implementation Steps

### Step 1: `pyproject.toml`

Create with `uv`-compatible format:

```toml
[project]
name = "job-embedding-experiment"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "sentence-transformers>=3.0,<4.0",
    "scikit-learn>=1.4,<2.0",
    "rapidfuzz>=3.6,<4.0",
    "rank-bm25>=0.2.2,<1.0",
    "matplotlib>=3.8,<4.0",
    "seaborn>=0.13,<1.0",
    "numpy>=1.26,<2.0",
    "anthropic>=0.40,<1.0",
    "scipy>=1.12,<2.0",
    "pyyaml>=6.0,<7.0",
    "pytest>=8.0,<9.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**Acceptance criteria**:
- `uv sync` succeeds and creates `uv.lock`
- All listed packages resolve without conflicts
- Python version constraint is 3.11+

### Step 2: `config.yaml`

Create with the full schema defined in spec 00. Model revision hashes are populated by looking up the current HEAD commit from HuggingFace for each model.

**Acceptance criteria**:
- All fields from spec 00 config schema are present
- File is valid YAML (parseable by `yaml.safe_load`)
- Model revision fields contain actual HuggingFace commit hashes (not placeholder strings)

### Step 3: `.gitignore`

```
__pycache__/
*.pyc
data/embeddings/
.venv/
*.egg-info/
uv.lock
```

Note: `uv.lock` is gitignored because this is a one-shot experiment, not a published package. Reproducibility comes from the pinned version ranges in `pyproject.toml` and the model revision hashes in `config.yaml`.

**Acceptance criteria**:
- `data/embeddings/` is excluded
- `__pycache__` is excluded
- Data files in `data/taxonomy/` and `data/test-cases/` are NOT excluded

### Step 4: Directory structure

Create empty directories and `__init__.py` files:

```
src/__init__.py
tests/__init__.py
data/taxonomy/
data/test-cases/
data/embeddings/
results/figures/
results/metrics/
scripts/
```

**Acceptance criteria**:
- All directories from the architecture's directory structure exist
- `src/__init__.py` and `tests/__init__.py` are empty files

---

## Interface Contract

**Exposes**: Project scaffold — all other components create files within this structure.

**Consumes**: Nothing.

---

## Testing Strategy

No unit tests for C0 itself. Validation:
- `uv sync` succeeds
- `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` succeeds
- `pytest --collect-only` succeeds (discovers test directory)
