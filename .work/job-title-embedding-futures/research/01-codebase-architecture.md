# Codebase Architecture

## Pipeline Flow
```
config.yaml → load models → encode targets (cached) → encode queries → rank (cosine sim) → evaluate → statistics → report
```

## Key Data Contract
All methods produce uniform ranking format:
```python
{"test_case_id": str, "method": str, "granularity": str, "ranked_results": [{"target_id": str, "score": float}]}
```

## Current Configuration Space
- 3 embedding models (minilm-384, bge-base-768, bge-large-1024) × 5 granularities = 15
- 3 baselines (TF-IDF, Jaro-Winkler, BM25) × 2 granularities (role, category) = 6
- Total: 21 configurations

## Integration Points for Futures
- **Query preprocessing**: Before `encode_queries()` in embed.py — abbreviation expansion goes here
- **Query encoding**: `model.encode()` supports `prompt` kwarg — instruction prefixing goes here
- **Post-ranking**: After `rank_targets()` — cross-encoder reranking goes here
- **Score fusion**: After both embedding + baseline rankings — RRF fusion goes here
- **Target sets**: `build_target_sets()` in targets.py — augmentation adds new granularity
- **Training**: New scripts using SentenceTransformerTrainer — fine-tuning/TSDAE

## Dependencies
- sentence-transformers >= 3.4.1 (supports prompt kwarg, CrossEncoder, training API)
- RTX 4080 on garden-pop (24GB VRAM — sufficient for all techniques)

## Caching
- Target embeddings cached at `{cache_dir}/{model_label}/targets_{granularity}.npy`
- Queries not cached (query-specific)
- New granularities/models get new cache files automatically
