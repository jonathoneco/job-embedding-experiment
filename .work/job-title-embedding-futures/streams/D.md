# Stream D: Phase 3 — ML Training Pipeline

**Phase:** 3
**Work Items:** W-08 (JEM-eij), W-09 (JEM-222)
**Dependencies:** Phase 2 complete (Streams B + C)

---

## Overview

Generate training data from the taxonomy (not test cases), then fine-tune bge-large-en-v1.5 with TSDAE + contrastive learning. Produces a 4th model entry in config.

## File Ownership

| Action | File | Work Item |
|--------|------|-----------|
| Create | `src/generate_training_data.py` | W-08 |
| Create | `data/training/pairs.jsonl` | W-08 (generated) |
| Create | `data/training/corpus.txt` | W-08 (generated) |
| Create | `src/fine_tune.py` | W-09 |
| Create | `models/bge-large-finetuned/` | W-09 (generated) |
| Modify | `config.yaml` | W-09 (4th model entry) |
| Modify | `src/embed.py` | W-09 (load_model revision handling) |

## W-08: Training Data Generator (JEM-eij) — spec 06

**Spec:** `.work/job-title-embedding-futures/specs/06-training-data-generator.md`

**What:** Generate ~5K contrastive triplets and ~5.7K TSDAE corpus from taxonomy roles.

**Steps:**
1. **Variant generation**: For each of 692 roles, generate 7-8 noisy variants using transform functions from `generate_rules.py` (`_apply_level_prefix`, `_apply_level_suffix`, `_apply_word_reorder`, `_apply_abbreviation`, `_apply_minor_rewording`). Use seed offset (`seed + 10000`) to avoid test data overlap.
2. **Hard negative mining**: Build TF-IDF matrix over 692 role names (character n-grams). For each role, find top-5 most similar roles from OTHER categories. Use scikit-learn (already a dependency).
3. **Assemble contrastive pairs**: Create (anchor, positive, negative) triplets — anchor = canonical role, positive = variant, negative = hard negative from different category. ~5K triplets, shuffled with seed. Write to `data/training/pairs.jsonl`.
4. **TSDAE corpus**: Collect all role names + generated variants, deduplicate. One per line. Write to `data/training/corpus.txt`.
5. **CLI entry point**: `main()` reads config, loads taxonomy, creates `data/training/`, calls both generators, prints summary.

**CRITICAL: Data leakage prevention** — training data from taxonomy roles ONLY, never from test cases. Different seeds than test generation. Post-generation check: no exact match between training variants and test case inputs.

**Acceptance Criteria:**
- 7-8 variants per role, ~4,900-5,500 total
- Hard negatives always from different category
- ~5K triplets in JSONL format
- Corpus ~5.7K entries, deduplicated
- No test case input_title values in training data

**Tests:** Variant count per role; hard negatives cross-category; triplet JSONL format; corpus deduplication.

## W-09: Fine-Tuning Pipeline (JEM-222) — spec 07 + spec 09 step 7

**Spec:** `.work/job-title-embedding-futures/specs/07-fine-tuning-pipeline.md`
**Also:** `.work/job-title-embedding-futures/specs/09-orchestration-updates.md` step 7
**Depends on:** W-08 (JEM-eij)

**What:** Fine-tune bge-large with 2-stage training, add as 4th model in config, update `load_model()` for local paths.

**Steps:**
1. `train_tsdae(base_model_id, corpus_path, output_dir, epochs=10, batch_size=32, noise_ratio=0.6)` — load SentenceTransformer, create DenoisingAutoEncoderLoss, train with SentenceTransformerTrainer. Save to intermediate dir.
2. `train_contrastive(base_model_path, pairs_path, output_dir, epochs=3, batch_size=32, lr=2e-5, warmup_ratio=0.1)` — load model, read JSONL pairs, create MultipleNegativesRankingLoss, train. Save to final dir.
3. `main()` CLI — Stage 1 (TSDAE, skippable with `--skip-tsdae`), output to `models/bge-large-tsdae/`. Stage 2 (contrastive), output to `models/bge-large-finetuned/`.
4. Add 4th model to `config.yaml`: `id: "models/bge-large-finetuned"`, `revision: null`, `dim: 1024`, `label: "bge-large-ft"`, `instruction: "Represent this sentence for searching relevant passages: "`
5. Update `load_model()` in `embed.py` (spec 09 step 7): skip `revision` kwarg when value is None/missing.

**Acceptance Criteria:**
- TSDAE uses DenoisingAutoEncoderLoss with noise_ratio=0.6
- Contrastive uses MultipleNegativesRankingLoss
- SentenceTransformerTrainer API (not deprecated fit())
- Model saved to `models/bge-large-finetuned/`
- Config: 4th model entry, `revision: null`, inherits instruction
- `load_model()` handles `revision=None` — no kwarg passed

**Estimated training time:** ~60-90 min (TSDAE) + ~15-20 min (contrastive) on RTX 4080.

**Tests:** Data loading (corpus → dataset, pairs → dataset); mini training run (10 pairs, 1 epoch); smoke test load + encode.

## Completion

W-08 must complete before W-09 starts (data dependency). Run training on garden-pop (RTX 4080). After training, run full evaluation to measure MRR improvement.
