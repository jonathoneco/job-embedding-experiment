# Futures

## Instruction Prefixing (Quick Win)
**Horizon:** next | **Domain:** experiment
BGE v1.5 models support query-side instruction prefix: `"Represent this sentence for searching relevant passages: "`. Not currently used. Trivial change, expected 2-5% improvement. Also test INSTRUCTOR model with task-specific template.

## Abbreviation Dictionary Expansion (Quick Win)
**Horizon:** next | **Domain:** data
Static lookup table mapping common partial abbreviations to full words (Mktg→Marketing, Dir→Director, Ops→Operations, TAM→Technical Account Manager). Applied to queries before embedding. No external infra needed.

## Target-Side Augmentation (Quick Win)
**Horizon:** next | **Domain:** data
LLM-generate 5-10 aliases per taxonomy role offline, precompute embeddings for all variants. At query time, match against expanded set and take best score. Expected 5-20% recall improvement. Runs entirely locally.

## Cross-Encoder Re-Ranking
**Horizon:** next | **Domain:** architecture
Two-stage: bi-encoder retrieves top-20, then `bge-reranker-base` (local model) re-scores pairs. Expected 5-15% ranking improvement. Adds ~20ms latency on GPU. Primarily addresses cross-category confusion.

## Hybrid Dense+Sparse Score Fusion
**Horizon:** next | **Domain:** experiment
Combine cosine similarity with BM25/TF-IDF scores via Reciprocal Rank Fusion. Experiment data already shows TF-IDF catches abbreviation patterns embeddings miss (TF-IDF hard MRR 0.604 vs bge-large 0.638). No external infra.

## Contrastive Fine-Tuning
**Horizon:** quarter | **Domain:** ml
Fine-tune bge-large with MultipleNegativesRankingLoss on synthetic (noisy_title, canonical_title) pairs. LLM generates variants offline. ~5K pairs showed 6.85% MRR gain in literature. Needs hard negative mining for cross-category confusion.

## TSDAE Domain Adaptation
**Horizon:** quarter | **Domain:** ml
Unsupervised pre-training on job title corpus (corrupts tokens, reconstructs). Adapts model to job title language without labels. 93.1% of supervised performance in literature. Built into Sentence Transformers. Combine with contrastive fine-tuning.

## BGE-M3 Evaluation
**Horizon:** quarter | **Domain:** experiment
Single model producing dense + sparse + ColBERT representations. Could replace current 3 models + 3 baselines with one unified model. Runs locally.
