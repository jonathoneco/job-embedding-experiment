# Quarter Horizon Techniques (3 ML Improvements)

## 6. Contrastive Fine-Tuning
**Feasibility: HIGH (9/10)**
- sentence-transformers 3.4.1 has full training API: SentenceTransformerTrainer + MultipleNegativesRankingLoss
- Data format: `{"anchor": noisy_title, "positive": canonical_role}` pairs
- ~5K pairs needed (692 roles × 7-8 variants each)
- Generate variants by extending existing generate_rules.py transforms
- Hard negative mining critical for cross-category confusion: roles from different categories with similar semantics
- Training: ~15-20 min on RTX 4080, batch size 32, 3 epochs
- Base model: bge-large-en-v1.5 (current best)
- Expected: +5-7% MRR (literature: 6.85% on 5K pairs)
- **Risk**: Catastrophic forgetting if learning rate too high — use 2e-5 with warmup

## 7. TSDAE Domain Adaptation
**Feasibility: MEDIUM (6/10)**
- Unsupervised: corrupts 60% of tokens, trains encoder to reconstruct
- DenoisingAutoEncoderLoss built into sentence-transformers
- Corpus: ~2K-3K job titles from taxonomy (692 roles) + descriptions + test cases
- Training: ~60-90 min on RTX 4080, 10 epochs, batch 32
- **Best used as Stage 1** before contrastive fine-tuning (Stage 2)
- Combined pipeline: TSDAE (1 hr) → contrastive (20 min) = ~1.5 hrs total
- Expected alone: +1-3% MRR; combined with contrastive: +5-8% MRR
- **Challenge**: Decoder compatibility — BGE models (BERT-based) are compatible
- **Challenge**: Hyperparameter sensitivity (noise ratio, epochs, LR)

## 8. BGE-M3 Evaluation
**Feasibility: MEDIUM (6/10) for full; HIGH (9/10) for dense-only**
- Single model producing dense + sparse + ColBERT representations
- Dense-only: drop-in replacement via sentence-transformers, but ~same performance as bge-large
- **Full value** requires FlagEmbedding library (new dependency) for sparse + ColBERT extraction
- Sparse output captures lexical patterns (like learned BM25) — addresses abbreviation weakness
- ColBERT: fine-grained token-level interactions
- Fusion of all 3 modalities via RRF could reach MRR 0.77-0.80
- **Dense-only NOT worth it** — must use all 3 modalities to justify switch
- Expected (full fusion): +5-8% MRR over current best
- **Strategic question**: Does BGE-M3 full fusion replace or complement techniques 1-5?

## Decision Points for Planning
1. **Contrastive fine-tuning vs BGE-M3**: Both target ~same improvement. Fine-tuning is more targeted (leverages domain data). BGE-M3 is more general. Recommend: fine-tuning first, BGE-M3 as alternative if fine-tuning plateaus.
2. **TSDAE worth the complexity?**: Only if contrastive fine-tuning alone doesn't hit thresholds. Budget 1.5 hrs GPU time for the combined pipeline.
3. **Training data**: Must generate separate from test data to avoid data leakage. Use taxonomy roles as seed, not test cases.
4. **Evaluation fairness**: Fine-tuned models must be evaluated on same held-out test set. Dev set used for hyperparameter tuning only.
