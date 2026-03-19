# Test Data Generation

## Target: 750 Test Cases

| Difficulty | Count | Proportion | Description |
|-----------|-------|-----------|-------------|
| Easy | 225 | 30% | Abbreviations, level prefixes, minor rewording |
| Medium | 300 | 40% | Synonyms, informal language, industry jargon |
| Hard | 150 | 20% | Ambiguous, creative, cross-functional, multi-label |
| Impossible | 75 | 10% | Genuinely unmappable (too vague, creative, niche) |

## Variation Types to Cover

1. **Abbreviations**: Sr., VP, Mgr, HR, IT, PM, DevOps
2. **Creative/vanity titles**: Growth Hacker, People Ops Ninja, Chief Happiness Officer
3. **Company-specific**: SDE III, Noogler, Engagement Manager (consulting)
4. **Combined roles**: Developer/Designer, Sales & Marketing Manager
5. **Level prefixes/suffixes**: Lead, Head of, Principal, I/II/III
6. **Industry jargon**: Quant, Attending, Of Counsel
7. **Misspellings**: Adminstrator, Enginear, Cordinator
8. **Emerging titles**: Prompt Engineer, Head of Remote, AI Whisperer

## Generation Strategy (Mix)

| Method | Count | Difficulty | Effort |
|--------|-------|------------|--------|
| Rule-based transforms | 100-150 | Easy | Low |
| LLM Pass 1 (systematic) | 250-350 | Easy/Medium | Low-Med |
| LLM Pass 2-3 (adversarial) | 100-150 | Medium/Hard | Medium |
| LLM Pass 4 (impossible) | 30-50 | N/A | Low |
| Manual/curated | 30-50 | Hard | High |
| **Total** | **560-850** | Mixed | |

## Output Schema

```json
{
  "id": "TC-0001",
  "input_title": "Sr. HRBP",
  "correct_roles": [
    {"role": "HR Business Partner", "category": "Human Resources"}
  ],
  "difficulty": "easy",
  "variation_type": "abbreviation",
  "notes": "Standard abbreviation of HR Business Partner"
}
```

## Quality Control

1. Automated validation: schema, deduplication, taxonomy membership
2. Spot-check 50-75 cases manually (stratified sample)
3. Difficulty calibration: second pass on 30-case sample
4. Multi-label: expect 15-25% of cases to have legitimate multiple mappings

## Dev/Test Split

- Dev set: 100 cases (for tuning thresholds, prompt engineering)
- Test set: 650 cases (final evaluation only)

## Category Coverage

Approximately uniform: ~20 cases per category (750/35 ≈ 21). Larger categories (IT, Engineering) may get 25-30; niche (Mining, Agriculture) get 10-15. Uniform is better than real-world distribution for experiment validity.
