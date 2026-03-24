# Futures: Curated Target Library Evaluation

## Deferred Enhancements

### F1: Curated role_desc and category_desc granularities
Add description-augmented target sets for the curated library. The filtering logic is identical; just needs curated versions of the `role_desc` and `category_desc` target builders. Deferred because three granularities cover the primary question.

### F2: Coverage-aware test case generation
Generate additional test cases specifically targeting curated roles that are underrepresented in the current 750-case test set. Would improve statistical power for curated evaluation.

### F3: Automated curation suggestions
Build a tool that suggests roles to include/exclude based on semantic overlap scores, usage frequency data, or cluster coverage analysis. Currently curation is fully manual.

### F4: Paired statistical comparison (full vs curated)
Run bootstrap paired tests on the subset of test cases that are valid in both target sets. Requires careful methodology since the population differs. Deferred because descriptive comparison is sufficient for the initial evaluation.

### F5: Multiple curated library variants
Support evaluating against several curated libraries (e.g., "small org", "tech company", "enterprise") to understand how library composition affects accuracy. Would require parameterizing the curated config to accept multiple files.
