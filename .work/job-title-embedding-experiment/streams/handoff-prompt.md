# Decompose Handoff → Implement

## What This Step Produced

8 work items across 4 streams and 4 phases, with beads issues and dependency tracking.

## Work Item Summary

| ID | Beads | Stream | Phase | Spec | Title |
|----|-------|--------|-------|------|-------|
| W-01 | JEM-l9q | phase-1 | 1 | 01 | Project scaffold |
| W-02 | JEM-h0r | stream-a | 2 | 02 | Taxonomy parser + cluster definitions + tests |
| W-03 | JEM-3ky | stream-a | 2 | 02 | Descriptions + targets + prep orchestrator |
| W-04 | JEM-3ft | stream-b | 2 | 04 | Embedding engine + baselines |
| W-05 | JEM-ccr | stream-a | 3 | 03 | Rule-based generator + validator + tests |
| W-06 | JEM-g2w | stream-a | 3 | 03 | LLM generator + manual cases + orchestrator |
| W-07 | JEM-04t | stream-d | 4 | 05 | Metrics engine + statistical analysis + tests |
| W-08 | JEM-zqj | stream-d | 4 | 05 | Report generator + main orchestrator |

## Dependency DAG

```
W-01 (setup)
 ├── W-02 (taxonomy) → W-03 (descriptions/targets) → W-05 (validators) → W-06 (LLM data) ─┐
 └── W-04 (matching pipeline) ─────────────────────────────────────────────────────────────────┤
                                                                                               └── W-07 (metrics) → W-08 (report)
```

## Concurrency Map

```
Phase 1:  [W-01 setup]
Phase 2:  [W-02 → W-03]  ||  [W-04]     (2 parallel streams)
Phase 3:  [W-05 → W-06]                   (1 stream, blocked on Phase 2 Stream A)
Phase 4:  [W-07 → W-08]                   (1 stream, blocked on Phase 3 + Phase 2 Stream B)
```

**Critical path**: W-01 → W-02 → W-03 → W-05 → W-06 → W-07 → W-08 (7 sequential items)
**Parallelism**: Only W-04 runs in parallel (during Phase 2, alongside W-02/W-03)

## Phase Gating

- **Phase 1 → 2**: After W-01 closes, run validation. Then launch Stream A and Stream B agents in parallel.
- **Phase 2 → 3**: Stream A (W-02, W-03) must both close. Stream B (W-04) may still be in progress — that's OK, Phase 3 doesn't depend on it.
- **Phase 3 → 4**: W-05, W-06 must close. W-04 must also be closed (Phase 4 depends on matching pipeline code).

## Stream Execution Documents

| Stream | File | Work Items |
|--------|------|-----------|
| Phase 1 Setup | `streams/phase-1-setup.md` | W-01 |
| Stream A (Data) | `streams/stream-a.md` | W-02, W-03, W-05, W-06 |
| Stream B (Pipeline) | `streams/stream-b.md` | W-04 |
| Stream D (Eval) | `streams/stream-d.md` | W-07, W-08 |

## API & GPU Dependencies

- **ANTHROPIC_API_KEY required**: W-03 (descriptions), W-06 (LLM test cases)
- **GPU required (garden-pop)**: Only the actual execution of `run_experiment.py` (W-08). All code can be written locally.

## File Ownership (no conflicts)

No file appears in more than one stream within the same phase. `tests/test_taxonomy.py` appears in both W-02 and W-03 but they are sequential within Stream A.

## Instructions for Implement Step

1. Execute Phase 1 (W-01) first — single agent, project scaffold
2. Launch Phase 2 with two parallel agents: Stream A (W-02→W-03) and Stream B (W-04)
3. After Phase 2 Stream A completes, launch Phase 3 (W-05→W-06) — can start even if Stream B is still running
4. After Phase 3 + Stream B complete, launch Phase 4 (W-07→W-08)
5. Each phase completion triggers quality review before proceeding
6. GPU compute (running the full experiment) happens after all code is written
