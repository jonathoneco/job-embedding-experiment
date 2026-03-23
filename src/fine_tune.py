"""Two-stage fine-tuning pipeline for BGE-large embedding model.

Stage 1: TSDAE domain adaptation — unsupervised denoising autoencoder
Stage 2: Contrastive fine-tuning — triplet-based MultipleNegativesRankingLoss

Heavy dependencies (datasets, sentence-transformers training API) are
imported lazily so the module can be loaded in lightweight environments.
"""

import argparse
import json
import time

import yaml


def train_tsdae(
    base_model_id: str,
    corpus_path: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 32,
    noise_ratio: float = 0.6,
) -> str:
    """Stage 1: TSDAE domain adaptation.

    Trains a denoising autoencoder on the job title corpus to adapt the
    model to the job title domain before supervised fine-tuning.

    Args:
        base_model_id: HuggingFace model ID or local path.
        corpus_path: Path to one-title-per-line corpus file.
        output_dir: Directory to save the adapted model.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        noise_ratio: Fraction of tokens to delete for denoising.

    Returns:
        The output_dir path.
    """
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    model = SentenceTransformer(base_model_id, trust_remote_code=False)

    with open(corpus_path) as f:
        sentences = [line.strip() for line in f if line.strip()]

    train_dataset = Dataset.from_dict({"sentence": sentences})
    loss = losses.DenoisingAutoEncoderLoss(
        model,
        decoder_name_or_path=base_model_id,
        tie_encoder_decoder=True,
    )

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save(output_dir)
    return output_dir


def train_contrastive(
    base_model_path: str,
    pairs_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    warmup_ratio: float = 0.1,
) -> str:
    """Stage 2: Contrastive fine-tuning with triplet data.

    Uses MultipleNegativesRankingLoss on anchor/positive/negative triplets
    to learn discriminative embeddings for job title matching.

    Args:
        base_model_path: Path to TSDAE-adapted model or HuggingFace ID.
        pairs_path: Path to JSONL file with anchor/positive/negative fields.
        output_dir: Directory to save the fine-tuned model.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        lr: Learning rate.
        warmup_ratio: Warmup ratio for learning rate scheduler.

    Returns:
        The output_dir path.
    """
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments

    model = SentenceTransformer(base_model_path, trust_remote_code=False)

    anchors, positives, negatives = [], [], []
    with open(pairs_path) as f:
        for line in f:
            pair = json.loads(line)
            anchors.append(pair["anchor"])
            positives.append(pair["positive"])
            negatives.append(pair["negative"])

    train_dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": negatives,
    })

    loss = losses.MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save(output_dir)
    return output_dir


def main() -> None:
    """CLI entry point: run 2-stage fine-tuning pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune BGE-large for job title matching",
    )
    parser.add_argument("--skip-tsdae", action="store_true", help="Skip TSDAE stage")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    base_model_id = "BAAI/bge-large-en-v1.5"
    tsdae_dir = "models/bge-large-tsdae"
    final_dir = "models/bge-large-finetuned"

    if not args.skip_tsdae:
        print("Stage 1: TSDAE domain adaptation...")
        start = time.time()
        train_tsdae(base_model_id, "data/training/corpus.txt", tsdae_dir)
        print(f"Stage 1 complete in {time.time() - start:.0f}s")
        contrastive_base = tsdae_dir
    else:
        print("Skipping Stage 1 (TSDAE)")
        contrastive_base = base_model_id

    print("Stage 2: Contrastive fine-tuning...")
    start = time.time()
    train_contrastive(contrastive_base, "data/training/pairs.jsonl", final_dir)
    print(f"Stage 2 complete in {time.time() - start:.0f}s")
    print(f"Fine-tuned model saved to {final_dir}")


if __name__ == "__main__":
    main()
