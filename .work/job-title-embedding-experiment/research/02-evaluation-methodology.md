# Evaluation Methodology

## Metrics

| Metric | Purpose | Production Threshold |
|--------|---------|---------------------|
| **MRR** (primary) | Best single summary metric | >= 0.75 |
| Top-1 Accuracy | Headline number | >= 0.65 |
| Top-3 Accuracy | Shortlist viability (human-in-loop) | >= 0.85 |
| Top-5 Accuracy | Generous shortlist | >= 0.92 |
| Category Accuracy | Coarse correctness | >= 0.90 |
| Mean Similarity Gap | Confidence signal | Report only |

## Granularity Levels (5 levels, not 3)

Research suggests expanding from 3 to 5 granularity levels:

| Level | # Targets | Representation |
|-------|-----------|---------------|
| Role | ~400 | Role name only |
| Role+Desc | ~400 | Role name + one-line description |
| Cluster | ~80-120 | Cluster label + description (agglomerative clustering) |
| Category+Desc | ~35 | Category name + key terms (5-10 terms) |
| Category | ~35 | Category name only |

Role+Desc and Category+Desc are cheap to add and often decisive — they test whether richer text representations help at the same granularity level.

## Baselines (Non-Embedding)

Include at minimum:
1. **TF-IDF + cosine similarity** (character 3-5 grams) — classic IR baseline
2. **Fuzzy string matching** (Jaro-Winkler via `rapidfuzz`) — catches typos/abbreviations
3. **BM25** (via `rank_bm25`) — "just use search" approach

Optional ceiling comparison: LLM-as-classifier (send title + taxonomy, ask to classify).

## Statistical Testing

- **McNemar's test** for pairwise approach comparison (paired binary outcomes, Bonferroni correction)
- **Bootstrap CIs** (1000 resamples) for MRR and accuracy — more informative than p-values
- **Friedman test** + Nemenyi post-hoc for 3+ simultaneous comparisons

## Result Presentation

1. **Model × Granularity matrix** (one table per metric)
2. **Grouped bar charts** (x=granularity, grouped=models, y=MRR)
3. **Heatmap** (models × granularity, color=MRR)
4. **Performance by difficulty tier** (easy/medium/hard breakdown)
5. **Similarity score distributions** (violin plots: correct vs incorrect matches)
6. **Category confusion matrix** (which categories get confused)

## Ground Truth Design

- Accept-set approach: primary label + acceptable alternatives
- Report metrics both with and without accept set
- At least 2 annotators on a 100-case subset, measure Cohen's kappa (target >= 0.7)
