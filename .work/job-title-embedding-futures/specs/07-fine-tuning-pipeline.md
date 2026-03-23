# Spec 07: Fine-Tuning Pipeline (C7)

**Technique:** T6 — Contrastive Fine-Tuning, T7 — TSDAE Domain Adaptation
**Phase:** 3 (ML Training)
**Scope:** Moderate (~250 lines)
**Dependencies:** Spec 06 (training data)
**References:** [architecture.md](architecture.md) C7, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

New `src/fine_tune.py` that fine-tunes `bge-large-en-v1.5` using the sentence-transformers training API. Two-stage pipeline: optional TSDAE pre-training (Stage 1) then contrastive fine-tuning (Stage 2). Produces a model directory added to `config.yaml` as the 4th model.

## Files

| Action | File | Changes |
|--------|------|---------|
| Create | `src/fine_tune.py` | New training script |
| Create | `models/bge-large-finetuned/` | Output: fine-tuned model weights |
| Modify | `config.yaml` | Add fine-tuned model entry |

## Interface Contract

### Exposes (src/fine_tune.py)

```python
def train_tsdae(
    base_model_id: str,
    corpus_path: str,             # data/training/corpus.txt
    output_dir: str,              # intermediate model output
    epochs: int = 10,
    batch_size: int = 32,
    noise_ratio: float = 0.6,
) -> str:
    """Stage 1: TSDAE unsupervised domain adaptation.

    Returns path to the intermediate model directory.
    """

def train_contrastive(
    base_model_path: str,         # model ID or path (from Stage 1 or original)
    pairs_path: str,              # data/training/pairs.jsonl
    output_dir: str,              # final model output
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> str:
    """Stage 2: Contrastive fine-tuning with hard negatives.

    Returns path to the final fine-tuned model directory.
    """

def main():
    """CLI entry point. Runs Stage 1 (optional) then Stage 2."""
```

### Consumes

- `data/training/pairs.jsonl` (from spec 06) — contrastive training pairs
- `data/training/corpus.txt` (from spec 06) — TSDAE corpus
- Base model: `BAAI/bge-large-en-v1.5` (from config)

### Config entry for fine-tuned model

```yaml
models:
  # ... existing 3 models ...
  - id: "models/bge-large-finetuned"     # local path
    revision: null                         # no revision for local models
    dim: 1024
    label: "bge-large-ft"
    instruction: "Represent this sentence for searching relevant passages: "
```

## Hyperparameters

### Stage 1: TSDAE

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss | `DenoisingAutoEncoderLoss` | Built into sentence-transformers, unsupervised |
| Noise ratio | 0.6 | Standard TSDAE default — 60% of tokens replaced with `[MASK]` |
| Epochs | 10 | Standard for TSDAE; convergence observed at 8-10 for small corpora |
| Batch size | 32 | Fits RTX 4080 VRAM with bge-large (1024 dim) |
| Learning rate | 2e-5 | Standard for BERT-family fine-tuning |
| Warmup | 10% of steps | Prevents catastrophic forgetting |

**Estimated time:** ~60-90 minutes on RTX 4080 (corpus ~5.7K sentences x 10 epochs)

### Stage 2: Contrastive

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss | `MultipleNegativesRankingLoss` | Standard for contrastive learning, efficient in-batch negatives |
| Epochs | 3 | Short — prevents overfitting on ~5K pairs |
| Batch size | 32 | In-batch negatives scale with batch size |
| Learning rate | 2e-5 | Standard, with warmup to prevent catastrophic forgetting |
| Warmup ratio | 0.1 | 10% of training steps |

**Estimated time:** ~15-20 minutes on RTX 4080

## Implementation Steps

### Step 1: Implement `train_tsdae()`

**What:** Load base model, create TSDAE data loader, train with `DenoisingAutoEncoderLoss`.

**Acceptance criteria:**
- [ ] Loads base model via `SentenceTransformer(base_model_id)`
- [ ] Reads corpus from `corpus_path` (one sentence per line)
- [ ] Creates `DenoisingAutoEncoderLoss` with `noise_ratio=0.6`
- [ ] Uses `SentenceTransformerTrainer` (sentence-transformers 3.4.1 API)
- [ ] Saves model to `output_dir`
- [ ] Returns output directory path
- [ ] Prints training progress (epoch, loss)

### Step 2: Implement `train_contrastive()`

**What:** Load model (from Stage 1 or base), create contrastive data loader, train with `MultipleNegativesRankingLoss`.

**Acceptance criteria:**
- [ ] Loads model from `base_model_path` (local dir from Stage 1, or HuggingFace ID)
- [ ] Reads pairs from `pairs_path` (JSONL with anchor/positive/negative fields)
- [ ] Creates training dataset with `InputExample(texts=[anchor, positive, negative])`
- [ ] Uses `MultipleNegativesRankingLoss` — handles triplets natively (positive + hard negative per anchor)
- [ ] Configures `SentenceTransformerTrainer` with lr, warmup, batch_size, epochs
- [ ] Saves final model to `output_dir`
- [ ] Returns output directory path

### Step 3: CLI entry point

**What:** `main()` function that orchestrates both stages.

**Acceptance criteria:**
- [ ] Reads config for base model ID and seed
- [ ] Stage 1 (TSDAE): runs if `--skip-tsdae` flag not set
- [ ] Stage 1 output: `models/bge-large-tsdae/` (intermediate)
- [ ] Stage 2 (Contrastive): always runs, uses Stage 1 output as base (or original model if Stage 1 skipped)
- [ ] Stage 2 output: `models/bge-large-finetuned/` (final)
- [ ] Prints summary: stages run, model location, training time

### Step 4: Add fine-tuned model to config.yaml

**What:** Add 4th model entry pointing to local fine-tuned model.

**Acceptance criteria:**
- [ ] New model entry with `id: "models/bge-large-finetuned"`, `label: "bge-large-ft"`
- [ ] `revision: null` (local model, no HuggingFace revision)
- [ ] `dim: 1024` (same as base bge-large)
- [ ] `instruction` field set (same as bge-large — fine-tuned model inherits instruction compatibility)
- [ ] Added AFTER existing 3 models (not replacing any)

## Implementation Notes

**sentence-transformers 3.4.1 training API:**
```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from datasets import Dataset

# TSDAE
model = SentenceTransformer(base_model_id)
train_dataset = Dataset.from_dict({"sentence": corpus_lines})
loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=base_model_id)
args = SentenceTransformerTrainingArguments(output_dir=output_dir, num_train_epochs=10, ...)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
trainer.train()

# Contrastive
model = SentenceTransformer(tsdae_model_path)
train_dataset = Dataset.from_dict({"anchor": anchors, "positive": positives, "negative": negatives})
loss = losses.MultipleNegativesRankingLoss(model)
args = SentenceTransformerTrainingArguments(output_dir=output_dir, num_train_epochs=3, ...)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)
trainer.train()
```

**`load_model()` in embed.py already works for local paths** — `SentenceTransformer("models/bge-large-finetuned")` loads from local directory. The only change needed is `revision=None` handling (skip revision kwarg when null).

## Testing Strategy

- Unit test: verify TSDAE data loading (corpus.txt → dataset)
- Unit test: verify contrastive data loading (pairs.jsonl → dataset with correct fields)
- Integration: run training with tiny subset (10 pairs, 1 epoch) to verify pipeline completes
- Smoke test: load fine-tuned model, encode a query, verify embedding dimension matches (1024)
