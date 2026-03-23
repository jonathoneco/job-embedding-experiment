# Decompose Handoff: job-title-embedding-futures

## What This Step Produced

10 work items across 5 streams, organized in 4 phases with a clear concurrency map and dependency graph. Each stream has a self-contained agent prompt with file ownership, acceptance criteria, and spec references.

## Work Item Index

| W# | Beads ID | Title | Stream | Phase | Spec |
|----|----------|-------|--------|-------|------|
| W-01 | JEM-1ub | Query Preprocessor | A | 1 | 01 |
| W-02 | JEM-2me | Instruction Prefix | A | 1 | 02 |
| W-03 | JEM-r7o | Target Augmentation Module | B | 2 | 03 |
| W-04 | JEM-g92 | Augmentation Integration | B | 2 | 03 |
| W-05 | JEM-rac | Cross-Encoder Reranker | C | 2 | 04 |
| W-06 | JEM-e1x | RRF Score Fusion | C | 2 | 05 |
| W-07 | JEM-mjz | Phase 2 Orchestration | C | 2 | 09 |
| W-08 | JEM-eij | Training Data Generator | D | 3 | 06 |
| W-09 | JEM-222 | Fine-Tuning Pipeline | D | 3 | 07 |
| W-10 | JEM-40z | BGE-M3 Integration | E | 4 | 08 |

## Concurrency Map

```
Phase 1:  ┌─── Stream A (W-01, W-02) ───┐
          └──────────────────────────────┘
                        │
                   [Eval Gate]
                        │
Phase 2:  ┌─── Stream B (W-03, W-04) ───┐
          │                              │
          │  ┌─── Stream C (W-05→W-06→W-07) ─┐
          └──┴───────────────────────────────┘
                        │
                   [Eval Gate]
                        │
Phase 3:  ┌─── Stream D (W-08 → W-09) ──┐
          └──────────────────────────────┘
                        │
                [MRR < 0.75 Gate]
                        │
Phase 4:  ┌─── Stream E (W-10) ──────────┐
          └──────────────────────────────┘
```

**Parallel execution:**
- Phase 1: Stream A only (small, single agent)
- Phase 2: Streams B and C run in parallel (no file overlap)
- Phase 3: Stream D only (sequential W-08 → W-09)
- Phase 4: Stream E only (conditional)

## File Ownership Per Phase

### Phase 1 (Stream A)
- Creates: `src/preprocess.py`
- Modifies: `src/embed.py`, `config.yaml`

### Phase 2 — Stream B
- Creates: `src/augment.py`, `data/taxonomy/augmented_targets.json`
- Modifies: `src/targets.py`, `src/evaluate.py`

### Phase 2 — Stream C
- Creates: `src/rerank.py`, `src/fusion.py`
- Modifies: `config.yaml`, `src/embed.py`

### Phase 3 (Stream D)
- Creates: `src/generate_training_data.py`, `data/training/pairs.jsonl`, `data/training/corpus.txt`, `src/fine_tune.py`, `models/bge-large-finetuned/`
- Modifies: `config.yaml`, `src/embed.py`

### Phase 4 (Stream E)
- Creates: `src/bgem3.py`
- Modifies: `config.yaml`, dependencies

**No file conflicts within any phase.** Streams B and C (the only parallel pair) have zero file overlap.

## Dependency Graph

```
W-01 ─┬──→ W-03 ──→ W-04 ─┬──→ W-08 ──→ W-09 ──→ W-10
      │                    │
W-02 ─┼──→ W-05 ──→ W-07 ─┘
      │                │
      └──→ W-06 ───────┘
```

**Critical path:** W-01/W-02 → W-05/W-06 → W-07 → W-08 → W-09 → W-10

## Stream Execution Documents

| Stream | File | Phase |
|--------|------|-------|
| A | `.work/job-title-embedding-futures/streams/A.md` | 1 |
| B | `.work/job-title-embedding-futures/streams/B.md` | 2 |
| C | `.work/job-title-embedding-futures/streams/C.md` | 2 |
| D | `.work/job-title-embedding-futures/streams/D.md` | 3 |
| E | `.work/job-title-embedding-futures/streams/E.md` | 4 |

## Instructions for Implement Step

1. **Phase 1**: Spawn single agent for Stream A. After completion, run Phase 1 evaluation gate.
2. **Phase 2**: Spawn two parallel agents — Stream B and Stream C. After both complete, run Phase 2 evaluation gate.
3. **Phase 3**: Spawn single agent for Stream D (sequential: W-08 then W-09). Run training on garden-pop (RTX 4080). After completion, run Phase 3 evaluation gate. Check if MRR >= 0.75.
4. **Phase 4** (conditional): If MRR < 0.75, spawn agent for Stream E. Otherwise skip.

Each agent receives its stream execution doc + relevant specs. Agents claim issues with `bd update <id> --status=in_progress` and close with `bd close <id>`.

Phase gating follows the Inter-Step Quality Review Protocol: Phase A (artifact validation) + Phase B (code quality review) after each phase.

## Key Context for Agents

- **Base commit:** `c68fd09d59544e64301412bb8cd4d60fd06d86ed`
- **Epic:** JEM-arn
- **Specs directory:** `.work/job-title-embedding-futures/specs/`
- **Cross-cutting contracts:** `.work/job-title-embedding-futures/specs/00-cross-cutting-contracts.md`
- **Uniform ranking format:** `{test_case_id, method, granularity, ranked_results: [{target_id, score}]}`
- **GPU:** RTX 4080 on garden-pop (for Phase 3 training)
- **Production thresholds:** MRR >= 0.75, Top-3 >= 0.85
