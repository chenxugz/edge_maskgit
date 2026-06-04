# Sequence-Length Sweep: testing "does GPU win at larger M?"

The M6 analysis claimed the GPU/CPU gap was structural to small-M prefill (M=257 in
MaskGIT-256) and would close at longer sequences:

> At longer prefills (M ≥ 1024, e.g. MaskGIT-512×512) the launch overhead amortizes
> and attention's O(M²) makes the GPU's parallelism start to pay off; the gap is
> expected to close, and may flip near M ≥ 2048.

This sweep tests that claim directly. Architecture is held fixed (24L, 768d, 16h,
3072 FFN, vocab 2025); only the positional-embedding row count and the VQGAN
latent grid vary. Weights are random — the model produces garbage images, but
kernel shapes are correct and latency is the same as a real-weights run.

## Setup

| Param | Value |
|---|---|
| Sequence lengths M (= n_tokens + 1) | 65, 257, 1025, 4097 |
| Backends | XNNPACK Q8 (host M1 Max) · OpenCL GQ8 (Pixel 9 Mali-G715) |
| Steps per generate | 8 (MaskGIT default) |
| Quantization | int8 (per-channel for XNNPACK, ggml Q8_0 for OpenCL) |

Synthetic GGUFs fabricated by `tools/make_synthetic_gguf.py` — reads metadata
+ tensor info from an existing real GGUF, overrides `pos_embd.weight` shape +
`maskgit.n_tokens`, regenerates per-type random data (sane fp16 scales so
quantized kernels don't see NaN scales and short-circuit).

```bash
python tools/make_synthetic_gguf.py --base models/maskgit-256-q8.gguf \
    --n-tokens 1024 -o models/synth/synth-n1024-q8.gguf
bazel-bin/mg-generate --bench --n-runs 3 --warmup 1 \
    -m models/synth/synth-n1024-q8.gguf --backend xnnpack
```

For the on-device leg the bench skips VQGAN (`MG_BENCH_SKIP_VQGAN=1`) at large M
because the bump-allocator's transformer scratch arena + VQGAN scratch arena
together exceed the phone's 12 GB RAM at M ≥ 1025. The "transformer (×8)"
component is what we want anyway — VQGAN runs once per generate, doesn't scale
with M, and isolating it removes a confounding variable.

## Results — transformer-only latency (×8 steps)

| M | host CPU (XNNPACK Q8) | device Mali (OpenCL GQ8) | **device / host** |
|---:|---:|---:|---:|
| **65**   | 699 ms      | 1 733 ms     | **2.48×** |
| **257**  | 3 018 ms    | 5 157 ms     | **1.71×** |
| **1025** | 20 804 ms   | 38 693 ms    | **1.86×** |
| **4097** | 207 177 ms  | OOM ※        | —         |

※ Device M=4097 needs ~79 GB scratch arena (M²·heads·layers·4 bytes); the bump
allocator never frees within a forward pass, so peak host memory ≫ phone RAM.

## Reading

**The crossover doesn't happen on this hardware.** The GPU/CPU ratio drops from
2.48× (M=65) to 1.71× (M=257) as the launch overhead amortizes — that part of
the M6 narrative held. But from M=257 → M=1025, the ratio plateaus at ~1.8×
rather than continuing to close. At M=1025 the GPU is still 1.86× slower than
the CPU, not faster.

### Why the gap stays open at large M

Three pieces of the per-op profile (from the M=257 device run, in
`benchmark/analysis.md` §M6 finale):

1. **The FC matmul (M-linear, int8, 48–51% of compute) is structurally pinned at
   ~3× CPU-favored.** Mali-G715 has only `arm_dot_acc` (4 MAC/instr); Cortex-X4
   has `SMMLA` (8×8 = 64 MAC/instr). That's a ~3× int8 throughput ceiling
   regardless of M — making the prefill longer doesn't change it.

2. **Attention (M²·D, F32 on both backends — quant unsafe here) scales the same
   shape on both sides.** At M=257 attention is ~14% of CPU, ~18% of GPU; at
   M=1025 it's a much bigger fraction (~64% of CPU, ~70% of GPU) — but the
   *ratio* doesn't budge much because both sides see the same O(M²) growth.

3. **The GPU does better than naive O(M²) extrapolation at M=1025** — predicted
   ~57 s from M=257 scaling, measured 39 s. So the per-head shapes ARE
   filling the GPU's cores better at larger M (4× the per-row work). That's
   a real wins on the attention side. But it isn't enough: the FC's 3×
   int8-throughput penalty dominates.

So the GPU win that O(M²) attention would deliver is canceled by the FC's
int8-throughput-ceiling penalty. The two effects roughly cancel from M=257
onward.

### What would change this story

- **Mali int8 wider than `arm_dot_acc`.** No such instruction on G715
  (`cl_arm_matrix_multiply` is the same 4-MAC). Would need a future GPU
  generation.
- **F32 attention path optimized on GPU** — flash-attention-style fusion
  (probed in M6 #7 as a negative result on small M; might pay off at M=1025+
  where the kernel becomes ALU-bound). Not pursued here.
- **Decode-style M=1.** GPU loses by much more (no parallelism). MaskGIT
  doesn't decode autoregressively, so this isn't relevant.

## Scaling shape (host XNNPACK Q8, all on M1 Max big cores)

| M | transformer (×8) ms | ratio vs. M=257 | predicted O(M) | predicted O(M²) |
|---:|---:|---:|---:|---:|
| 65    | 699       | 0.23× | 0.25× | 0.06× |
| 257   | 3 018     | 1.0×  | 1.0×  | 1.0×  |
| 1025  | 20 804    | 6.9×  | 4.0×  | 16×   |
| 4097  | 207 177   | 68.6× | 16×   | 256×  |

The transformer scales somewhere between linear (FC) and quadratic (attention).
Attention's share grows from ~14% at M=257 to ~91% at M=4097 — exactly the
regime the M6 hypothesis predicted would help the GPU. The actual on-device
measurement (where we have it, up to M=1025) shows the GPU benefiting from
the bigger per-head work — but not enough to flip the gap.

## Conclusion

The earlier claim — **"M ≥ 2048 may flip the gap"** — is **not supported by
this measurement**. On Mali-G715 vs Cortex-X4 SMMLA, the GPU stays ~1.7-1.9×
slower across the whole tested range (M ∈ {257, 1025}). The fundamental cause
is the chip's int8 throughput ceiling on the M-linear FC matmul, which
dominates even when attention's M² growth fills a much larger fraction of
total compute.

Updating the M6 wrap-up text accordingly.

## Files

| Path | What |
|---|---|
| `tools/make_synthetic_gguf.py` | GGUF synth at arbitrary n_tokens |
| `models/synth/synth-n*-{q8,gq8}.gguf` | the 8 synth GGUFs (gitignored) |
| `host-n{64,256,1024,4096}-q8.txt` | full bench output, host CPU |
| `device-n{64,256,1024}-gq8.txt` | full bench output, Pixel 9 |
