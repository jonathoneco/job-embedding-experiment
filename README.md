# Job Title Embedding Experiment

Can embedding models reliably map free-text job titles to a structured taxonomy of 692 roles across 42 categories?

## Bottom Line

**Embeddings are a strong foundation but not a complete solution.** Generic sentence embeddings (bge-large) achieve MRR 0.733 and Top-3 accuracy of 80.6% against production thresholds of MRR 0.75 / Top-3 85%. They handle ~80% of job title variations reliably but hit a hard ceiling on abbreviations, domain jargon, and creative titles.

## What We Tested

| Technique | Impact | Verdict |
|-----------|--------|---------|
| 3 embedding models (MiniLM, BGE-base, BGE-large) | BGE-large best by +0.05 MRR | BGE-large is the baseline |
| 3 baselines (TF-IDF, BM25, fuzzy) | All below embeddings | Embeddings clearly superior |
| Abbreviation expansion ("Sr Dev" -> "Senior Developer") | +0.001 to +0.009 MRR | Modest positive, kept |
| BGE instruction prefixing | -0.003 to -0.050 MRR | Negative for short text, removed |
| Cross-encoder reranking (ms-marco) | -0.005 to -0.050 on top models | Wrong domain, removed |
| LLM-augmented targets (6,288 aliases) | -0.009 to -0.043 MRR | Dilutes signal, not useful |

Best result: **bge-large @ role_desc = MRR 0.733, Top-3 0.806**

## What Works

Embeddings reliably handle surface-level title transformations and many semantic mappings:

| Variation | Rank-1 Rate | Examples |
|-----------|-------------|---------|
| Case changes, reordering | 100% | "SECURITY ANALYST" -> Security Analyst |
| Level suffixes/prefixes | 84-100% | "Senior Accountant" -> Accountant |
| Minor rewording | 94% | "The Corporate Counsel" -> Corporate Counsel |
| Cross-category roles | 75% | "Cloud Solutions Architect" -> Cloud Solution Architect |
| Combined/dual roles | 70% | "QA/DevOps Engineer" -> DevOps Engineer |
| Synonyms | 61% | "Distribution Center Manager" -> Warehouse Manager |

The model demonstrates genuine semantic understanding beyond word overlap: "Faculty" -> Professor, "Building Information Modeling Specialist" -> BIM Coordinator, "Head of Enterprise Risk" -> Enterprise Risk Manager.

62% of correct matches score above 0.85 confidence, suggesting a threshold could reliably flag uncertain cases for human review.

## What Doesn't Work

The 12% miss rate and 8% rank 4-10 bracket represent the hard ceiling of generic embeddings:

| Variation | Miss Rate | Why |
|-----------|-----------|-----|
| Abbreviations | 20% | "TAM", "VP Eng" carry no semantic signal. Three-letter acronyms are unrecoverable. |
| Domain jargon | 20% | "Revenue Cycle Analyst" = "Bursar Operations Analyst" is a knowledge mapping, not similarity. |
| Creative/slang titles | 19% | "Chief Storyteller" = "Communications Manager" requires understanding job function. |
| Misspellings | 24% | "Mktg Autom Ops Wiz" combines abbreviation + misspelling + slang. |

## Sector Mapping (Job Title -> Category)

Direct category-level matching across 42 sectors. We score two ways:

- **Strict:** The model's top pick exactly matches the labeled category.
- **Relaxed:** The top pick is in an equivalent category group (e.g., Finance ~ Banking & Financial Services, Engineering ~ Software Engineering ~ IT). We define 13 groups of adjacent categories where the boundary is organizational rather than semantic.

| Approach | Strict | Relaxed | Notes |
|----------|--------|---------|-------|
| Match to category name | 36% | — | Category names too abstract |
| Match to category + keywords | **50%** | **61%** | 11% of "failures" are adjacent-category matches |
| Implicit via role matching | **77%** | **82%** | Best: find the role, read its category |

The strict-to-relaxed gap reveals that **~11% of sector mapping "errors" are taxonomy ambiguity, not model failures.** The model picks a reasonable adjacent sector; the test data just labels a different one.

**Equivalent category groups used for relaxed scoring:**

| Group | Categories Treated as Equivalent |
|-------|--------------------------------|
| Finance | Finance, Banking & Financial Services, Insurance |
| Engineering | Engineering, Software Engineering, Information Technology |
| Content | Communications, Marketing, Agency & Advertising |
| Operations | Operations, Supply Chain & Logistics, Manufacturing, Frontline Management |
| Construction | AEC, Real Estate & Construction, Engineering |
| Media | Media & Entertainment, Sports & Entertainment, Communications |
| Healthcare | Healthcare, Pharma & Life Sciences |

**When sector mapping works**, the domain keyword is obvious:

| Query | Matched Sector | Score |
|-------|---------------|-------|
| "Telecom Network Capacity Planner" | Telecom | 0.746 |
| "Brand Marketing Manager" | Marketing | 0.745 |
| "Supply Chain Operations Analyst" | Operations | 0.736 |
| "Junior Brand Finance Analyst (CPG)" | Consumer Packaged Goods (CPG) | 0.732 |
| "Sports/Venue Ops Maestro" | Sports & Entertainment | 0.736 |

**Taxonomy-ambiguous cases** (strict fail, relaxed pass — not real errors):

| Query | Expected | Got Instead | Adjacent? |
|-------|----------|-------------|-----------|
| "Endpoint Support Engineer" | Information Technology | Software Engineering | Yes — same engineering group |
| "Financial Systems Analyst" | Finance, IT | Banking & Financial Services | Yes — same finance group |
| "Brand Design Lead" | Design | Marketing | Yes — same content/design group |
| "Technology Project Manager" | IT, Operations, Software Eng | Project Management | Yes — same operations group |
| "Automation Program Manager" | Engineering, IT, PM | Software Engineering | Yes — same engineering group |

**Genuine failures** (39% of cases — truly wrong sector):

| Query | Expected | Got Instead | Why |
|-------|----------|-------------|-----|
| "Happiness Engineer" | Customer Service | Software Engineering (0.673) | "Engineer" dominates; creative title has no CS signal |
| "Operations Analyst, Payments" | Banking & Financial Services | Operations (0.695) | "Operations" keyword overpowers "Payments" domain |
| "commodity manager" | Supply Chain & Logistics | Agriculture (0.673) | "Commodity" has agricultural connotations |
| "Brand Protection Specialist" | Legal | Consumer Packaged Goods (0.663) | "Brand" pulls toward CPG; legal function invisible |
| "Revenue Management Specialist" | Hospitality & Travel | Sales (0.665) | "Revenue" signals Sales; hospitality context missing |

The pattern: genuine failures happen when the functional keywords in a title (Operations, Engineering, Brand) overpower the domain context. The embedding sees "Operations Analyst" and picks the Operations category, ignoring that "Payments" implies Banking. Role-level matching then reading the category remains the best path to sector classification — it achieves 77% strict / 82% relaxed.

## Granularity Findings

| Granularity | Targets | MRR | Top-3 | Takeaway |
|-------------|---------|-----|-------|----------|
| role_desc | 692 | **0.733** | **0.806** | Best — descriptions disambiguate similar roles |
| role | 692 | 0.729 | 0.797 | Strong baseline |
| category_desc | 42 | 0.631 | 0.727 | Weak for direct sector classification |
| category | 42 | 0.496 | 0.578 | Not viable |
| cluster | 90 | 0.490 | 0.559 | Not viable — abstract labels |

**Why do descriptions help?** In 86 cases, adding AI-generated descriptions rescued a match that bare role names missed:

| Query | Role (bare name) | Role + Description |
|-------|-----------------|-------------------|
| "Diversity, Equity & Inclusion Manager" | rank >10 (matched: Policy & Governance Manager) | rank 1: "DEI Program Manager: Develops and executes diversity, equity, and inclusion initiatives..." |
| "Security Operations Center Lead" | rank >10 (matched: Operations Specialist) | rank 1: "SOC Manager: Manages security operations center and oversees threat detection..." |
| "Account-Based Marketing Manager" | rank >10 (matched: Account Manager) | rank 1: "ABM Manager: Develops targeted marketing campaigns for specific high-value enterprise accounts" |
| "Product Lifecycle Management Specialist" | rank >10 (matched: Technical Product Manager) | rank 1: "PLM Analyst: Manages product lifecycle management systems..." |
| "Accounts Receivable Specialist" | rank >10 (matched: Order Management Specialist) | rank 1: "AR Specialist: Manages customer invoicing, tracks receivables..." |

The descriptions expand abbreviations ("DEI", "SOC", "ABM", "PLM", "AR") and explain functional scope, giving the embedding model richer text to match against.

**When do bare role names win?** In 94 cases, descriptions actually hurt — mostly with misspellings and non-English input where the added description text creates noise:

| Query | Role (bare) | Role + Desc |
|-------|-------------|-------------|
| "Datenanalyst" (German) | rank 2 (Data Analyst nearby) | rank >10 (description text pushes Data Steward higher) |
| "Projekct Manger" (misspelled) | rank 2 (close to Project Manager) | rank >10 (description noise overwhelms the fuzzy match) |
| "Revenue Operations Manager" | rank 1 (Operations Manager) | rank >10 (descriptions pull toward Marketing Operations) |

## Production Recommendation

**For a "top 3-5 suggestions" UX:** Embeddings are ready. 80% Top-3 accuracy with confidence scoring to flag uncertain matches. Users review suggestions rather than trusting auto-match.

**For automated single-match:** Not yet viable. The 12% miss rate and 23% wrong-category rate are too high for unsupervised mapping. Options to close the gap:

1. **Contrastive fine-tuning** (most promising) — domain-adapt bge-large on the taxonomy to learn "Sr Dev" ~ "Software Engineer". Training pipeline ready, needs GPU.
2. **Hybrid approach** — embedding retrieval for candidates, LLM-in-the-loop for disambiguation of low-confidence matches.
3. **Two-stage with human review** — auto-match above 0.85 confidence (~62% of cases), route the rest to human review.

## Repository Structure

```
src/
  embed.py              # Embedding engine (encode, rank, orchestrate)
  preprocess.py         # Query abbreviation expansion
  baselines.py          # TF-IDF, BM25, fuzzy matching
  evaluate.py           # MRR, Top-K, category accuracy metrics
  augment.py            # LLM-generated role aliases
  rerank.py             # Cross-encoder reranking
  fusion.py             # Reciprocal Rank Fusion
  fine_tune.py          # TSDAE + contrastive fine-tuning pipeline
  generate_training_data.py  # Training data from taxonomy

scripts/
  run_experiment.py     # Main experiment orchestrator

results/
  summary.md            # Detailed results with qualitative analysis
  run-baseline/         # Original experiment results
  run-abbrev-rerank/    # Abbreviation + reranking experiment
  run-abbrev-augmented/ # Abbreviation + augmented targets experiment
  metrics/              # Live results (current code state)
```

## Running

```bash
uv sync                           # Install dependencies
python scripts/run_experiment.py  # Run full experiment (~30s on CPU)
```
