# Spec 06: Training Data Generator (C6)

**Technique:** T6/T7 prerequisite
**Phase:** 3 (ML Training)
**Scope:** Moderate (~200 lines)
**References:** [architecture.md](architecture.md) C6, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

New `src/generate_training_data.py` that generates (noisy_title, canonical_role) pairs for contrastive training plus a job title corpus for TSDAE. Training data is generated from taxonomy roles only — NOT from test cases — to prevent data leakage.

## Files

| Action | File | Changes |
|--------|------|---------|
| Create | `src/generate_training_data.py` | New script |
| Create | `data/training/pairs.jsonl` | Generated: contrastive training pairs |
| Create | `data/training/corpus.txt` | Generated: TSDAE corpus |

## Interface Contract

### Exposes (src/generate_training_data.py)

```python
def generate_contrastive_pairs(
    roles: list[dict],            # [{"role": str, "category": str}, ...]
    seed: int,
    output_path: str,             # path for pairs.jsonl
) -> int:
    """Generate (anchor, positive, hard_negative) triplets for contrastive training.

    Returns number of pairs generated.
    Idempotent: overwrites output_path if it exists.
    """

def generate_tsdae_corpus(
    roles: list[dict],
    output_path: str,             # path for corpus.txt
) -> int:
    """Generate job title corpus for TSDAE unsupervised pre-training.

    Returns number of corpus entries.
    """

def main():
    """CLI entry point. Reads config.yaml for taxonomy path and seed."""
```

### Output formats

**pairs.jsonl** — one JSON object per line:
```json
{"anchor": "Systems Administrator", "positive": "Sys Admin", "negative": "System Analyst"}
```

- `anchor`: canonical role name
- `positive`: noisy variant of the same role (abbreviation, reword, level prefix, etc.)
- `negative`: hard negative — a different role with similar surface form or from a confusable category

**corpus.txt** — one title per line:
```
Systems Administrator
Sys Admin
IT Administrator
Network Administrator
...
```

Contains canonical roles + all generated variants. Used for TSDAE denoising autoencoder pre-training.

### Consumes

- Taxonomy roles (loaded from `data/taxonomy/`)
- Transform rules from `generate_rules.py` (reused for variant generation)
- Seed from `config.yaml`

## Implementation Steps

### Step 1: Variant generation

**What:** For each of the 692 roles, generate 7-8 noisy variants using the same transform functions from `generate_rules.py`: level prefix, level suffix, word reorder, abbreviation, minor rewording. Use different random seeds than test generation to avoid overlap.

**Acceptance criteria:**
- [ ] Uses `_apply_level_prefix()`, `_apply_level_suffix()`, `_apply_word_reorder()`, `_apply_abbreviation()`, `_apply_minor_rewording()` from `generate_rules.py`
- [ ] Generates 7-8 variants per role (not all transforms apply to every role)
- [ ] Uses seed offset from config seed (e.g., `seed + 10000`) to avoid overlap with test data seeds
- [ ] Total: ~4,900-5,500 variants
- [ ] No variant is identical to any existing test case input_title (verified post-generation)

### Step 2: Hard negative mining

**What:** For each anchor role, select hard negatives — roles from different categories that share surface-form similarity.

**Strategy:**
1. Build a TF-IDF matrix over all 692 role names (character n-grams, same as baselines.py)
2. For each role, find top-5 most similar roles from OTHER categories
3. These become hard negatives for contrastive training

**Acceptance criteria:**
- [ ] Hard negatives are always from a different category than the anchor
- [ ] Each anchor has 1-5 hard negatives (some roles may have fewer similar cross-category matches)
- [ ] Similarity computed via TF-IDF cosine (reuses scikit-learn, already a dependency)
- [ ] No role is a hard negative for itself

### Step 3: Assemble contrastive pairs

**What:** Combine variants and hard negatives into (anchor, positive, negative) triplets.

**Acceptance criteria:**
- [ ] Each triplet: anchor = canonical role, positive = one variant, negative = one hard negative
- [ ] ~5,000 triplets total (692 roles x ~7 variants, each paired with a random hard negative)
- [ ] Written to `data/training/pairs.jsonl` as JSON lines
- [ ] Triplets are shuffled (deterministic with seed)

### Step 4: Generate TSDAE corpus

**What:** Collect all role names + all generated variants into a corpus file.

**Acceptance criteria:**
- [ ] Contains all 692 canonical role names
- [ ] Contains all ~5,000 generated variants
- [ ] One title per line
- [ ] Written to `data/training/corpus.txt`
- [ ] Deduplicated (no exact duplicate lines)

### Step 5: CLI entry point

**What:** `main()` function that loads config, reads taxonomy, calls both generators.

**Acceptance criteria:**
- [ ] Reads `config.yaml` for seed and taxonomy path
- [ ] Loads roles from taxonomy
- [ ] Creates `data/training/` directory if needed
- [ ] Calls `generate_contrastive_pairs()` and `generate_tsdae_corpus()`
- [ ] Prints summary: pair count, corpus size

## Data Leakage Prevention

**Critical constraint:** Training data must be generated from taxonomy roles only, never from test cases.

- Variant generation uses the same _transform functions_ as `generate_rules.py` but with different seeds
- No test case `input_title` values are used as training data
- Post-generation check: verify no exact match between training variants and test case inputs (warning, not blocking — statistical overlap of transform outputs is possible but unlikely given seed offset)

## Testing Strategy

- Unit test: verify variant generation produces expected count per role
- Unit test: verify hard negatives are from different categories
- Unit test: verify triplet assembly and JSONL format
- Unit test: verify corpus deduplication
- Integration: generate full training set, verify counts (~5K pairs, ~5.7K corpus entries)
