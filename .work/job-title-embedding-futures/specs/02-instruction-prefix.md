# Spec 02: Instruction Prefix (C2)

**Technique:** T1 — Instruction Prefixing
**Phase:** 1 (Quick Wins)
**Scope:** Trivial (~10 lines config + 2 lines code)
**References:** [architecture.md](architecture.md) C2, [00-cross-cutting-contracts.md](00-cross-cutting-contracts.md)

---

## Overview

Add `prompt` kwarg to `encode_queries()` calls in `embed.py`. BGE models expect instruction prefixes to improve encoding quality. Config-driven per model — models without instructions get `prompt=None` (no-op in sentence-transformers).

## Files

| Action | File | Changes |
|--------|------|---------|
| Modify | `src/embed.py` | Pass `prompt` kwarg to `model.encode()` in `encode_queries()` |
| Modify | `config.yaml` | Add `instruction` field per model |

## Interface Contract

### Config schema (per model)

```yaml
models:
  - id: "sentence-transformers/all-MiniLM-L6-v2"
    revision: "c9745ed1..."
    dim: 384
    label: "minilm"
    # No instruction field — defaults to None

  - id: "BAAI/bge-base-en-v1.5"
    revision: "a5beb1e3..."
    dim: 768
    label: "bge-base"
    instruction: "Represent this sentence for searching relevant passages: "

  - id: "BAAI/bge-large-en-v1.5"
    revision: "d4aa6901..."
    dim: 1024
    label: "bge-large"
    instruction: "Represent this sentence for searching relevant passages: "
```

**Instruction string:** `"Represent this sentence for searching relevant passages: "` — the standard BGE retrieval instruction. This is what BGE models were trained with for asymmetric search tasks.

**Query-side only:** Instruction prefixes are only applied to query encoding. Target encoding remains un-prefixed (research finding: short job title targets perform better without prefix).

### Modified function signature

```python
def encode_queries(
    model: SentenceTransformer,
    queries: list[str],
    batch_size: int,
    prompt: str | None = None,    # NEW parameter
) -> np.ndarray:
```

**The `prompt` kwarg is passed directly to `model.encode()`**, which prepends it to each input string. sentence-transformers handles this natively — no manual string concatenation needed.

## Implementation Steps

### Step 1: Add `instruction` to config.yaml

**What:** Add `instruction` field to each BGE model entry. MiniLM gets no instruction field.

**Acceptance criteria:**
- [ ] `bge-base` and `bge-large` entries have `instruction: "Represent this sentence for searching relevant passages: "`
- [ ] `minilm` entry has no `instruction` field
- [ ] Existing config fields unchanged

### Step 2: Modify `encode_queries()` signature

**What:** Add `prompt: str | None = None` parameter. Pass it to `model.encode()`.

**Acceptance criteria:**
- [ ] `encode_queries()` accepts `prompt` kwarg
- [ ] When `prompt` is not None, passed as `prompt=prompt` to `model.encode()`
- [ ] When `prompt` is None, `model.encode()` called without `prompt` kwarg (or with `prompt=None` — sentence-transformers treats both the same)
- [ ] L2 normalization still applied after encoding

### Step 3: Thread instruction through `run_embedding_model()`

**What:** Read `model_config.get("instruction")` and pass to `encode_queries()`.

**Acceptance criteria:**
- [ ] `instruction` read from `model_config` with `.get("instruction")` (returns `None` if absent)
- [ ] Passed as `prompt=instruction` to `encode_queries()`
- [ ] NOT passed to `encode_targets()` (query-side only)
- [ ] Method name unchanged (instruction is transparent — see spec 00, S0.2)

## Implementation Notes

The key code change in `encode_queries()` is adding one kwarg to the `model.encode()` call:
```python
embeddings = model.encode(queries, batch_size=batch_size, prompt=prompt, normalize_embeddings=True)
```

Note: `normalize_embeddings=True` may already be handled separately. Check existing code — if normalization is done manually after encode, keep it that way and just add the `prompt` kwarg.

## Testing Strategy

- Verify config loads correctly with new `instruction` field
- Verify `encode_queries()` produces different embeddings when `prompt` is provided vs None (for BGE models)
- Verify `encode_targets()` is NOT affected (no instruction passed)
- Verify existing 21 configs still produce valid results (backward compatibility)
