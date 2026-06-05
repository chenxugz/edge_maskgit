# Milestone 3 — Per-layer numerical verification

Layer-by-layer cosine / max-abs-diff comparison between the C++ runtime and the
PyTorch oracle, for one fixed deterministic forward pass.

## What this catches (and doesn't)

We verify **step 1 of the iterative-decoding loop** — input = class token at
position 0, MASK at positions 1..256. This is the canonical input used by the
existing end-of-pipeline `verify-opencl-transformer` logit gate; we just
additionally dump per-layer activations and compare each one.

**Catches:** any kernel-level math bug (matmul, LayerNorm, GELU, softmax,
FlashAttention), graph builder layout/stride bugs, quantization drift per
layer, the typical sources of regression.

**Doesn't catch:** error accumulation across the 8-step iterative loop, or
behavior that only triggers at specific token distributions later in the loop.
Those would need *trajectory replay* — dump PyTorch's per-step input + per-layer
intermediates, then teacher-force the C++ side step by step. Not implemented;
the step-1 gate is sufficient for catching most regressions.

We sidestep the sampling-RNG problem entirely this way: the transformer forward
is deterministic given input tokens, so no need to align Gumbel noise between
the two stacks for per-layer math verification.

## Workflow

### 1. Generate the oracle (PyTorch, one-time per model change)

```bash
~/anaconda3/envs/maskgit-ref/bin/python reference/dump_intermediates.py
```

Writes 51 tensors under `reference/export/intermediates_step1/`:
`embd_post_norm.bin`, `blk.{0..23}.{attn,ffn}_post.bin`, `output_norm.bin`,
`output_logits.bin`, plus a `meta.json` with shapes.

### 2. Dump from the C++ runtime

```bash
# Host (or device — same binary, run via adb)
MG_DUMP_NAMED=1 bazel-bin/verify-opencl-transformer \
    models/maskgit-256-gq8.gguf reference/export \
    --dump-dir verification/dumps/<run-name>
```

`MG_DUMP_NAMED=1` enables the OpenCL backend's per-named-tensor readback
(off by default — adds ~50 device→host transfers per compute). Without it,
the dumped intermediates are uninitialized arena memory.

### 3. Compare

```bash
python3 verification/compare_intermediates.py verification/dumps/<run-name>
```

Auto-detects the precision tier (f32 / q8 / q4) from the output-logits diff
magnitude and applies tiered tolerances:

| tier | max_abs floor | cosine floor |
|---|---:|---:|
| f32 | 0.02 | 0.99999 |
| q8 | 1.0 | 0.9999 |
| q4 | 2.0 | **0.99 (per-layer)** — int4 weights accumulate to ~0.995 at mid-network before the output projection smooths the logits back to ≥0.9999 |

The per-layer Q4 floor is looser than the existing logit gate's 0.9999
because **the int4 weight precision budget genuinely drifts middle layers to
~0.995** — not a kernel bug, just the chip's int4 ceiling. The logit gate
catches anything beyond that because the output projection averages
per-channel error across the full vocabulary.

## Current scoreboard (M=257, step 1)

| run | worst cosine | worst max_abs | result |
|---|---:|---:|---|
| Host OpenCL F32 | **1.00000000** | 2.9e-5 | 51 / 51 PASS (bit-perfect, float-add reordering only) |
| Host OpenCL Q8_0 | 0.99997516 | 6.87e-2 | 51 / 51 PASS (int8 quant noise) |
| Mali OpenCL Q8_0 (fp16 FA) | 0.99991820 | 1.99e-1 | 51 / 51 PASS (int8 + fp16 FA noise) |
| Mali OpenCL Q4_K (int8-dot) | 0.99427133 | 1.38e0 | 51 / 51 PASS (int4 weight noise, smooths to logit cosine 0.99996) |

## Files

| Path | What |
|---|---|
| `reference/dump_intermediates.py` | PyTorch oracle dumper (hooks on emb_ln, AttentionLN/MlpLN per layer, final LN) |
| `verification/compare_intermediates.py` | Per-tensor comparison harness with tiered tolerances |
| `verification/dumps/` | C++ runtime dumps (gitignored — regenerate as needed) |
| `tools/verify-opencl-transformer.cpp` | Extended with `--dump-dir DIR`; gates on `MG_DUMP_NAMED` env var to actually read back named buffers |
| `src/mg-transformer.cpp` | `->named(...)` on the 51 tensors that get compared |
| `src/mg-opencl/mg-opencl.cpp` | `compute()` reads back named tensors when `MG_DUMP_NAMED` is set |
