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
| Steps per generate | 8 (MaskGIT default) |
| Quantization | int8 (per-channel for XNNPACK, ggml Q8_0 for OpenCL) |
| **Primary comparison** | **Pixel 9 Cortex-X3 + A715 (XNNPACK Q8, SMMLA) vs Pixel 9 Mali-G715 (OpenCL GQ8, arm_dot_acc)** — same chip, same RAM, same thermal envelope |
| Secondary cross-check | M1 Max XNNPACK Q8 (host) — shows the device CPU is within ~7% of M1 at M=257-1025, so the on-device gap isn't a Pixel-9-CPU-is-slow artifact |

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

**Primary: Pixel 9 CPU vs Pixel 9 GPU (same chip, same RAM).**

| M | Device CPU (XNNPACK Q8) | Device GPU (OpenCL GQ8) | **GPU / CPU** |
|---:|---:|---:|---:|
| **65**   | 773 ms      | 1 733 ms    | **2.24×** |
| **257**  | 2 985 ms    | 5 157 ms    | **1.73×** |
| **1025** | 22 198 ms   | 38 693 ms   | **1.74×** |
| **4097** | 319 657 ms  | OOM ※       | —         |

※ Device GPU M=4097 needs ~79 GB scratch arena (M²·heads·layers·4 bytes); the
bump allocator never frees within a forward pass, so peak host memory ≫ phone RAM.

**Cross-check: M1 Max XNNPACK (host) confirms the Pixel-9 CPU isn't a bottleneck.**

| M | Host CPU (XNNPACK Q8) | Device CPU (XNNPACK Q8) | Device/Host |
|---:|---:|---:|---:|
| 65   | 699 ms      | 773 ms      | 1.11× |
| 257  | 3 018 ms    | 2 985 ms    | 0.99× |
| 1025 | 20 804 ms   | 22 198 ms   | 1.07× |
| 4097 | 207 177 ms  | 319 657 ms  | 1.54× (thermal) |

The Pixel 9 Cortex-X3 SMMLA path matches M1 Max XNNPACK to within ~7% at
M ≤ 1025 — they're both compute-bound on int8 matmul. M=4097 on device shows
thermal throttling kicking in over 5+ minutes of sustained compute; the GPU
gap measurement would have the same caveat, so the M=4097 device GPU
infeasibility doesn't bias the conclusion at M ≤ 1025.

## Reading

**The crossover doesn't happen on this hardware.** The GPU/CPU ratio drops from
2.24× (M=65) to 1.73× (M=257) as the launch overhead amortizes — that part of
the M6 narrative held. But from M=257 → M=1025, the ratio plateaus at ~1.74×
rather than continuing to close. At M=1025 the GPU is still 1.74× slower than
the CPU, not faster.

### Per-kernel comparison (M=257 vs M=1025)

The CPU per-op profile uses XNNPACK's `XNN_FLAG_BASIC_PROFILING` — actual
kernel wall time. The GPU per-op profile uses `clFinish` serialization to
attribute time per op-type, which inflates absolute ms but preserves shares;
GPU ms below have been de-inflated by `(true_transformer_total /
per_op_sum)`.

| M | kernel    | CPU ms | GPU ms | **GPU/CPU** |
|---:|---|---:|---:|---:|
| 257  | FC        | 2 019  | 3 110  | **1.54×** |
| 257  | Attention | 582    | 1 093  | **1.88×** |
| 257  | SoftMax   | 124    | 182    | **1.47×** |
| 1025 | FC        | 7 235  | 11 604 | **1.60×** |
| 1025 | Attention | 9 692  | 18 007 | **1.86×** |
| 1025 | SoftMax   | 2 735  | 5 979  | **2.19×** |

**Per-kernel ratios are essentially flat** (FC 1.54 → 1.60, Attn 1.88 → 1.86)
and SoftMax actually gets worse (1.47 → 2.19). The expected "GPU catches up
as compute dominates" effect does not show up in this measurement.

### Why the gap stays open at large M on THIS implementation

The standard intuition — "longer M → compute-bound → GPU wins" — assumes the
GPU has surplus compute throughput that gets activated when the kernel work
is large enough. On Mali-G715 + Cortex-X3 with the current kernels, three
reasons that intuition doesn't materialize:

1. **Mali shares system RAM; "high FP32 throughput" doesn't translate when
   attention is bandwidth-bound.** Per-step attention at M=1025 touches
   ~150 MB per layer × 24 layers × 8 steps ≈ 30 GB across the run. Mali has
   no dedicated VRAM — it competes with the CPU for the same ~50 GB/s
   LPDDR5X. The CPU keeps its working set in L2/L3 caches and pays less in
   DRAM traffic per FLOP. The 1.3 TFLOPs of Mali FP32 compute is moot if the
   data can't reach it fast enough.

2. **The GPU attention kernel here is a naive `MulMat(f32)` — no
   flash-attention, no tiling, no shared-memory reuse.** Even at M=1025
   per-head matrices it stays at 1.86× CPU-slower, suggesting the kernel
   isn't pulling away. A flash-attention rewrite (or any memory-tiled
   attention) would let the GPU show its theoretical advantage. We probed an
   attention fp16 fusion in M6 #7 and got a negative result on small M; the
   right test would have been a full flash-attention rewrite, not just a
   dtype cast. **We did not do that work.**

3. **Int8 FC has no GPU headroom on this chip.** Cortex-X3 SMMLA puts the CPU
   int8 throughput in the same league as Mali's `arm_dot_acc`. There's no
   theoretical gap to close — both backends are within a small constant
   factor of the chip's int8 ceiling.

So the conclusion is empirically robust **for this implementation on this
hardware**, but it's an implementation statement, not a fundamental one. A
flash-attention rewrite might flip the gap at large M; this measurement
can't predict that. What we can say:

- **As shipped today**, the on-device CPU is faster at every measured M.
- **The per-kernel ratios don't show the GPU pulling ahead** as M grows,
  which means the M ≥ 2048 "flip" forecast from the earlier M6 narrative is
  not happening with the current kernel set.
- **A different GPU attention kernel** (flash-attention with tiling) is the
  obvious next experiment if we wanted to challenge that conclusion.

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

## Scaling shape (Device CPU XNNPACK Q8, Pixel 9 Cortex-X3)

| M | transformer (×8) ms | ratio vs. M=257 | predicted O(M) | predicted O(M²) |
|---:|---:|---:|---:|---:|
| 65    | 773      | 0.26× | 0.25× | 0.06× |
| 257   | 2 985    | 1.0×  | 1.0×  | 1.0×  |
| 1025  | 22 198   | 7.4×  | 4.0×  | 16×   |
| 4097  | 319 657  | 107×  | 16×   | 256×  |

The transformer scales somewhere between linear (FC) and quadratic (attention).
Attention's share grows from ~14% at M=257 to ~91% at M=4097 — exactly the
regime the M6 hypothesis predicted would help the GPU. The on-device
measurement (where we have it, up to M=1025) shows the GPU benefiting from
the bigger per-head work — but not enough to flip the gap.

## Update (2026-06-04): fp16 K/V tiles inside flash-attention — another 7-22%

After flipping FA on as the default, the obvious next lever is **fp16 K/V tile
storage** inside the FA kernel. Mali-G715 has `cl_khr_fp16` with 2× ALU rate
on half-precision multiply, and shrinking K/V tiles halves __local memory
pressure. Q row is cast to fp16 once at workgroup entry; K, V are cast on
load into __local; the inner Q·K and P·V dots run at fp16-mul → fp32-accumulate
(Mali's native MAD path). Softmax stats (m_i, l_i) and the output
accumulator stay fp32. Auto-dispatched to `k_flash_attention_h` when
`cl_khr_fp16` is present (Mali yes; M1 OpenCL no — falls back to fp32 kernel).

**Device sweep update:**

| M | CPU XNN Q8 | GPU FA fp32 | GPU FA fp16 | + strided FA | + LN-affine | **+ VQGAN GN+SiLU** | **vs CPU** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 257  | 2 985 ms   | 4 882 ms    | 4 775 ms    | 4 751 ms    | 4 653 ms    | **4 547 ms**    | 1.52× slower |
| 1025 | 22 198 ms  | 23 405 ms   | 21 736 ms   | 21 047 ms   | 19 103 ms   | **19 147 ms**   | **0.86× — GPU 16% faster** |
| 4097 | 319 657 ms | 147 628 ms  | 115 488 ms  | 114 711 ms  | 109 737 ms  | **111 828 ms**  | **0.35× — GPU 2.86× faster** |

The VQGAN GN+SiLU column shows transformer-only at M=1025/4097 (M-quadratic doesn't
touch VQGAN), so the wins land on the M=257 column where VQGAN is 25% of wall: end-
to-end **6 466 → 6 119 ms (−5.4%)**, VQGAN decode **1 671 → 1 366 ms (−18%)**.

## Q4_K sweep (Step 16 — int8-dot for Q4_K)

After bringing Q4_K up onto the same int8-dot kernel as Q8_0, the same crossover
pattern shows up — and at large M the win over the CPU is **even bigger**, because
the CPU's XNN int4 path scales worse than int8 (`maskgit-256-q4.gguf` at M=4097 takes
**484 s** on CPU vs Q8's 320 s):

| M | CPU XNN Q4 | GPU gq4 | **gq4 / CPU Q4** | CPU XNN Q8 | **gq4 / CPU Q8** |
|---:|---:|---:|---:|---:|---:|
| 65   | 681 ms      | 1 744 ms    | 2.56× slower    | 773 ms     | 2.26× slower |
| 257  | 3 189 ms    | 5 129 ms    | 1.61× slower    | 2 985 ms   | 1.72× slower |
| 1025 | 27 640 ms   | 20 308 ms   | **0.73× — GPU 36% faster** | 22 198 ms | **0.91× — GPU 9% faster** |
| 4097 | 483 522 ms  | 116 823 ms  | **0.24× — GPU 4.14× faster** | 319 657 ms | **0.37× — GPU 2.74× faster** |

GPU gq4 is only 4–13% slower than gq8 across the sweep — the int8-dot path closed
almost all of the gap that the F32 dequant kernel had created. At M=4097 the GPU
beats the CPU by 4.14× (vs 2.74× for gq8) — the larger Q4-vs-Q8 win on the GPU side
reflects that the int8 dot product treats Q4_K and Q8_0 with the same hardware path;
the larger Q8-vs-Q4 *loss* on the CPU side reflects KleidiAI's int4 micro-kernel
being less mature than its int8 one.

**So Q4_K isn't just smaller (216 MB vs 298 MB) — at the M ≥ 1025 prefill regime it's
also clearly the right choice for the GPU on this device, and the GPU is the right
device for that regime.**

The crossover lands even earlier: **GPU now beats CPU at M=1025** (was tied
with fp32 FA), and at M=4097 the GPU is **2.77× faster** (was 2.17×).

**Cosine on Mali** (direct measurement via cross-compiled
verify-opencl-transformer): fp16 FA = **0.99999929**, fp32 naive chain =
0.99999930. Difference = 10⁻⁸ (fp32 comparison round-off). Bit-equivalent at
logit precision. Quick-5 IS/top-k unchanged within small-sample noise.

## Update (2026-06-03): flash-attention is the missing piece — now default-on

**Status as of commit `b3df…`:** flash-attention is the default for the OpenCL
backend. Set `MG_NO_FLASH_ATTN=1` to fall back to the naive
`MulMat(K,Q) → SoftMax → MulMat(Vᵀ, scores)` chain for A/B comparison.
Quick-5 IS/top-k regression: within noise (IS −0.005, Top-1 2/40 image flip,
Top-5 1/40 flip, PSNR −0.03 dB — all FP32 summation-order effects). See
`evaluation/README.md` for the full quality table.


The conclusion above held for the **naive `MulMat(f32) + SoftMax`** attention
kernel. Adding a tiled flash-attention-v2 OpenCL kernel that never materializes
the M×M scores tensor changes the picture completely:

| M | CPU XNNPACK | GPU baseline | GPU flash-attn | FA speedup | **FA / CPU** |
|---:|---:|---:|---:|---:|---:|
| 65   | 773 ms      | 1 733 ms    | 1 697 ms    | −2%   | 2.20× |
| 257  | 2 985 ms    | 5 157 ms    | 4 882 ms    | −5%   | 1.64× |
| 1025 | 22 198 ms   | 38 693 ms   | 23 405 ms   | **−39%** | **1.05× (tied)** |
| 4097 | 319 657 ms  | OOM ※       | 147 628 ms  | (was infeasible) | **0.46× (GPU 2.17× faster)** |

※ Baseline OOM'd because the bump allocator reserved the M²·heads·layers·4 B
scores tensor in main memory (~79 GB at M=4097). Flash-attn never allocates
that tensor — scores live in workgroup-local memory inside the kernel.

**The crossover happens at M ≈ 1025 and the GPU pulls decisively ahead at
M=4097.** At M=4097 the per-op profile shows FlashAttn at 80 664 ms vs the
naive MulMat(f32)+SoftMax which would have been ~190 s (from the M=1025
extrapolation), so the kernel itself is roughly 2.3× faster than the naive
chain at this M. End-to-end, the GPU now beats the CPU by 2.17× because
attention has become the dominant cost and the GPU has the compute throughput
to handle it once the memory pressure is removed.

### Why flash-attention flips the result

Per-step at M=1025, the naive path moves through DRAM:
- Q · Kᵀ → write M² scores (~67 MB/layer)
- SoftMax → read M² + write M² (134 MB/layer)
- S · V → read M² + write M·D (~70 MB/layer)
- Total: ~270 MB DRAM traffic per layer × 24 layers × 8 steps ≈ **52 GB across the run**

Flash-attention keeps scores and softmax in workgroup-local memory inside a
single kernel — the M² tensor never touches DRAM. Only Q, K, V, and the
final output are read/written from global memory. Per-layer DRAM traffic
drops to ~10 MB, total run traffic drops to ~2 GB — **a 26× DRAM bandwidth
reduction at M=1025**. This is precisely what flash-attention was designed
to do, and the reason LiteRT-LM (which uses fused attention kernels) reports
GPU >> CPU on similar workloads.

### Per-op profile, M=1025, GPU side

| op | baseline ms | flash-attn ms |
|---|--:|--:|
| MulMat(q) FC      | 15 522 | 13 346 (cooler thermals?) |
| MulMat(f32) attention | 23 981 | — (subsumed) |
| SoftMax            |  8 075 | — (subsumed) |
| **FlashAttn**      | —      | **6 467** |
| total transformer  | 40 416 | 23 405 |

Attention block went from 32 056 ms (23 981 + 8 075) → 6 467 ms. That's the
**5× reduction in the attention block** that closes the gap with CPU.

## Conclusion (revised)

The earlier claim was **"with naive kernels, M ≥ 2048 may flip the gap"** —
that turned out to be wrong: with the naive `MulMat(f32) + SoftMax` chain,
the gap doesn't close with M. **But the underlying intuition was right**:
attention is the dominant kernel at large M, and the GPU does have the
compute headroom to win — once the memory pattern is fixed.

With flash-attention shaders, the crossover is **M ≈ 1025** (essentially
tied) and the GPU **decisively beats the CPU at M = 4097** (2.17× faster).
For larger prefill workloads (LLMs, longer-resolution image models) the
GPU will dominate further.

The original M6 wrap claim — "for this class of phone, the CPU is the
right tool" — was true *for the kernels we had at the time*. With
flash-attention, the cross-over point lands inside the regime that
real on-device workloads will see, and **the GPU becomes the right tool
for M ≥ ~1000**. This brings us in line with the LiteRT-LM ecosystem,
which relies on the same memory-tiled attention pattern.

**Two layers of why:**

1. **Implementation:** our GPU attention is a naive `MulMat(f32)` with no
   tiling / shared-memory reuse / flash-attention. The "GPU wins at large
   M because it's compute-bound" intuition assumes a compute-bound kernel.
   Ours stays bandwidth- or launch-bound. A flash-attention rewrite is
   the test that would actually challenge this conclusion.
2. **Hardware:** Mali-G715 shares system RAM with the CPU (no dedicated
   VRAM), so the GPU's theoretical FP32 throughput advantage is partially
   neutralized by competing for the same ~50 GB/s LPDDR5X bandwidth.
   This is a structural mobile-GPU constraint, not a kernel issue.

**Practical takeaway:** for MaskGIT as shipped today, the on-device CPU
is the right tool on this class of phone. The GPU work in M6 was still
valuable — device gq8 went 111 s → 7.1 s — and the profiler-guided
journey transfers. The "longer sequences flip it" forecast was wishful
*given the kernels we wrote*; it might still hold with a properly
memory-tiled GPU attention, but that's an unverified hypothesis until
we build and measure it.

## Files

| Path | What |
|---|---|
| `tools/make_synthetic_gguf.py` | GGUF synth at arbitrary n_tokens |
| `models/synth/synth-n*-{q8,gq8,q4,gq4}.gguf` | synth GGUFs at each (M, quant) (gitignored) |
| `device-xnn-n{64,256,1024,4096}-q8.txt` | bench transcripts, Pixel 9 CPU XNN Q8 |
| `device-xnn-n{64,256,1024,4096}-q4.txt` | bench transcripts, Pixel 9 CPU XNN Q4 |
| `device-n{64,256,1024}-gq8.txt`         | bench transcripts, Pixel 9 Mali Q8 (pre-FA) |
| `device-fa-n{64,256,1024,4096}-gq8.txt` | bench transcripts, Pixel 9 Mali Q8 (post-FA) |
| `device-fa-n{64,256,1024,4096}-gq4.txt` | bench transcripts, Pixel 9 Mali Q4_K (int8-dot) |
| `host-n{64,256,1024,4096}-q8.txt`       | secondary: M1 Max XNNPACK cross-check |
