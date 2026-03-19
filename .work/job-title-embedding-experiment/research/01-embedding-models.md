# Embedding Model Selection

## Three-Tier Model Selection

All models run locally on a 4080 (16GB VRAM) via `sentence-transformers`. No API keys needed.

| Tier | Model | Params | Dim | STS Score | VRAM | Throughput |
|------|-------|--------|-----|-----------|------|------------|
| Small | `all-MiniLM-L6-v2` | 22.7M | 384 | ~82.0 | ~300 MB | 5-10K sent/s |
| Medium | `BAAI/bge-base-en-v1.5` | 109M | 768 | ~85.4 | ~1 GB | 1.5-3K sent/s |
| Large | `BAAI/bge-large-en-v1.5` | 335M | 1024 | ~87.1 | ~3 GB | 500-1.2K sent/s |

## Key Decisions

- **Similarity metric**: Cosine similarity (all models trained with contrastive loss optimizing for cosine). Pre-normalize embeddings, then use dot product for speed.
- **No input prefixes**: For symmetric similarity (both sides are short job titles), v1.5 BGE models work well without prefixes. Worth a quick ablation test.
- **No larger models needed**: 7B+ embedding models (e5-mistral, gte-Qwen2) are 50-100x slower with only 1-2 STS points improvement on short text. Not worth complexity.
- **BGE family consistency**: Using BGE for both medium and large tiers isolates the impact of model capacity (same training methodology, different sizes).

## Throughput Notes

For ~400 taxonomy entries × ~750 test cases, the entire experiment completes in seconds on any tier. Throughput is not a practical concern at this scale.

## Implementation

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=128)
similarities = embeddings @ taxonomy_embeddings.T  # dot product = cosine sim
```
