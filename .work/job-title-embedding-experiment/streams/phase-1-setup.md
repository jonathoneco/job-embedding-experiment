# Stream: Phase 1 — Project Setup

**Work Items**: W-01 (JEM-l9q)
**Spec**: 01-project-setup.md
**Dependencies**: None (first phase)

---

## W-01: Project Scaffold (JEM-l9q)

**Files to create**:
- `pyproject.toml` — Dependencies and project metadata
- `config.yaml` — Full experiment configuration (spec 00 schema)
- `.gitignore` — Exclude caches, embeddings, __pycache__
- `src/__init__.py` — Empty package marker
- `tests/__init__.py` — Empty test package marker
- Directories: `data/taxonomy/`, `data/test-cases/`, `data/embeddings/`, `results/figures/`, `results/metrics/`, `scripts/`

**Key details**:
- Use `uv`-compatible pyproject.toml format with hatchling build system
- Dependencies: sentence-transformers, scikit-learn, rapidfuzz, rank-bm25, matplotlib, seaborn, numpy, anthropic, scipy, pyyaml, pytest
- Python >=3.11
- Config model `revision` fields: look up current HEAD commit from HuggingFace for each of the 3 models
- `.gitignore`: exclude `data/embeddings/`, `__pycache__/`, `.venv/`, `uv.lock`, `*.egg-info/`, `*.pyc`

**Acceptance criteria**:
- `uv sync` succeeds and creates `uv.lock`
- All packages resolve without conflicts
- `yaml.safe_load(open('config.yaml'))` succeeds
- `pytest --collect-only` discovers test directory
- All directories from architecture exist
- Model revision fields contain actual HuggingFace commit hashes

**Verification**:
```bash
uv sync
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
uv run pytest --collect-only
```

**Beads workflow**:
```bash
bd update JEM-l9q --status=in_progress
# ... implement ...
bd close JEM-l9q
```
