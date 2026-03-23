# Failure Modes & Performance Gaps

## Current Performance
| Config | MRR | Top-1 | Top-3 | Top-5 | Cat.Acc |
|--------|-----|-------|-------|-------|---------|
| bge-large@role | 0.732 | 0.649 | 0.795 | 0.841 | 0.741 |
| bge-large@role_desc | 0.731 | 0.642 | 0.806 | 0.845 | 0.772 |
| bge-base@role_desc | 0.721 | 0.628 | 0.785 | 0.854 | 0.734 |

## Production Threshold Gap
- MRR: 0.732 vs 0.75 target (-0.018, -2.4%)
- Top-3: 0.795 vs 0.85 target (-0.055, -6.5%)

## Difficulty Breakdown (bge-large@role)
- Easy: 0.863 MRR | Medium: 0.681 | Hard: 0.638
- **Hard cases are the bottleneck** — 23 percentage point gap vs easy

## Top Failure Modes (ranked by impact)
1. **Category confusion (70% of top-10 failures)**: Model ranks roles from wrong category highest. "Revenue Cycle Analyst" → Finance instead of Operations.
2. **Abbreviation blindness (~20-30% of hard failures)**: TAM, Mktg, Dir. CS Ops — semantic embeddings can't decode abbreviations.
3. **Rank displacement (15%)**: Correct role exists but buried below top-10 by similar-sounding alternatives.
4. **Alias/synonym coverage**: Creative rewrites miss because only canonical name is in target set.
5. **Seniority/scope modifiers (10-15% of misranks)**: Lead/Manager/Director treated as independent concepts.

## Failure → Technique Mapping
| Failure Mode | Primary Fix | Secondary Fix |
|-------------|------------|---------------|
| Category confusion | Cross-encoder reranking | Contrastive fine-tuning (hard negatives) |
| Abbreviation blindness | Abbreviation dictionary | Hybrid dense+sparse fusion |
| Rank displacement | Cross-encoder reranking | Target augmentation |
| Alias coverage | Target augmentation | Contrastive fine-tuning |
| Seniority confusion | Instruction prefixing | TSDAE domain adaptation |

## Key Insight
TF-IDF hard MRR (0.604) is only 3.4% behind bge-large hard MRR (0.638). Embeddings don't have decisive advantage on surface variants — sparse methods catch patterns embeddings miss. This validates hybrid fusion.
