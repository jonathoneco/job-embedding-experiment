# Futures Discovered During Research

## Evaluation Test Harness Improvements
**Horizon:** next | **Domain:** testing
Current experiment re-runs all 21 configs from scratch. Add incremental evaluation mode that only re-runs changed methods/granularities while preserving existing results. Would significantly speed up iterative development.

## Cross-Language Title Support
**Horizon:** quarter | **Domain:** data
Error analysis shows foreign language titles (German "Datenanalyst", Spanish "Especialista en Cumplimiento AML") in test set. Current models are English-only. BGE-M3 is multilingual — could address this if cross-language support becomes a requirement.

## Confidence-Based Routing
**Horizon:** someday | **Domain:** architecture
When similarity gap (rank-1 minus rank-2 score) is below threshold, route to human review or secondary classification. Current `mean_similarity_gap` metric already computed but not used for routing decisions.
