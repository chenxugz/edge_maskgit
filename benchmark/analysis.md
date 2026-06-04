# Milestone 5 — Benchmark & Profiling: analysis

## Roofline — why the GPU is far from the CPU on device (`benchmark/roofline.py`)

Mali-G715 (Tensor G4) order-of-magnitude specs: **~1.3 TFLOP/s FP32 / ~2.6 TFLOP/s
FP16**, **~60 GB/s** shared LPDDR5X. MaskGIT-256 work:

| stage | FLOP | ideal (compute roofline) | measured (Mali) | **achieved efficiency** |
|---|--:|--:|--:|--:|
| Transformer ×8 steps | ~890 GFLOP (111/fwd) | **~0.34 s** (fp16) | 15.0 s | **~2%** |
| VQGAN decode (once) | ~188 GFLOP | **~0.14 s** (fp32) | 6.3 s | **~2%** |

The roofline says the whole thing *could* run in **well under 1 s** on this GPU —
neither compute nor bandwidth (weight reads ~0.15 s even with tiled re-reads) is the
wall. We achieve only **~2–7%** of peak. For contrast, XNNPACK int8 on the same CPU
(~0.7 s ideal at ~0.6 TOP/s i8mm, ~3 s measured) runs at **~25%** of *its* roofline.

So the device GPU↔CPU gap is **mostly a kernel-efficiency / software-maturity gap, not
a hardware gap.** The rest of this section derives each number above and then attributes
the gap. Run `python3 benchmark/roofline.py` to reproduce every figure.

### What a roofline model is

A roofline bounds achievable performance by the **smaller** of two hardware ceilings:

- **Peak compute** — `P` FLOP/s the ALUs can retire (here ~2.6 TFLOP/s fp16).
- **Peak memory bandwidth** — `B` byte/s to/from DRAM (here ~60 GB/s).

For a kernel that does `W` FLOP while moving `Q` bytes, its **arithmetic intensity** is
`I = W/Q` (FLOP per byte). The attainable rate is `min(P, B·I)`: below the **ridge
point** `I* = P/B` the kernel is **memory-bound** (bandwidth is the wall, rate `= B·I`);
above it the kernel is **compute-bound** (rate `= P`). For this GPU
`I* = 2.6e12 / 60e9 ≈ 43 FLOP/byte` — a kernel must do ≥43 FLOP per byte fetched to be
compute-bound. The **ideal time** for a stage is therefore `max(W/P, Q/B)`, and the
**achieved efficiency** we quote is simply `ideal_time / measured_time` (equivalently
`measured_FLOP/s / roofline_FLOP/s`). 100% means we hit the relevant ceiling; ~2% means
the kernel spends 98% of its wall time neither computing nor streaming usefully.

### Transformer FLOPs (≈111 GFLOP / forward)

A matmul of an `M×K` by `K×N` matrix is `M·K·N` multiply-accumulates (MACs) = `2·M·K·N`
FLOP (1 mul + 1 add). For a linear layer over `S` tokens this is `2·S·(in·out)` FLOP, so
across a whole transformer **`FLOP ≈ 2 · params · tokens`** — the standard rule of thumb.
Per layer, with `E=768` hidden, `FFN=3072`, `H=16` heads, `HD=48` head-dim, `S=257`
tokens (256 image + 1 class), `VOCAB=2025`, `L=24` layers:

| weight-matmul (per layer, MACs) | expression | value |
|---|---|--:|
| Q,K,V,O projections | `4·E²` | 2.36 M |
| FFN up + down | `2·E·FFN` | 4.72 M |
| output proj (logits, once at end) | `E·VOCAB` | 1.56 M |

The attention score + value matmuls carry **no weights** — they are activation×activation:
`scores = Q·Kᵀ` and `A·V`, each `H·S·S·HD` MACs, so `2·H·S·S·HD` MACs/layer.

Putting it together (`×S` tokens, `×L` layers for the FC part; output proj is once):

```
FC_MAC   = S · (4·E² + 2·E·FFN + E·VOCAB)            ≈ 53.2 G  (×2 = 106.5 GFLOP, ×L folded in below)
attn_MAC = H·S²·HD·2                                  per layer
FC_FLOP   = 2·FC_MAC·L         = 106.5 GFLOP
attn_FLOP = 2·attn_MAC·L       =   4.9 GFLOP
total     = FC_FLOP + attn_FLOP ≈ 111.4 GFLOP / forward
```

The FC projections dominate (~96%); attention scores are tiny because `S=257` is short
(`S²` is small relative to `E·FFN`). Over 8 decoding steps: **~890 GFLOP**. This
**supersedes the design doc's "4.6 GFLOP/forward"** (CLAUDE.md §2.2), which under-counted —
it used `E=768`/8 heads and omitted the FFN and output-projection contributions; the real
figure is ~24× larger.

### Transformer memory traffic & arithmetic intensity

The quantized FC weights are Q8_0: 32 int8 + one fp16 scale per 32-value block →
`~1.0625 bytes/param`. Weight params per layer = `4·E² + 2·E·FFN + E·VOCAB`, so

```
W_bytes(Q8_0) = (4·E² + 2·E·FFN + E·VOCAB)·L · 1.0625 ≈ 220 MB  (read ≥ once / forward)
```

A tiled GEMM stages the weight in `64×64` output tiles; with only `S=257` rows the weight
is re-streamed once per row-tile, a factor `ceil(S/64) = 5`. So effective weight traffic
is `220 MB × 5 ≈ 1.10 GB/forward`. Arithmetic intensity for the FC part is then
`I = 106.5 GFLOP / 1.10 GB ≈ 97 FLOP/byte` — **above** the ridge point (≈43), so even
with the 5× re-read the transformer is **(just) compute-bound**. Both rooflines are well
under a second per 8 steps:

| bound | per forward | ×8 steps |
|---|--:|--:|
| compute (fp16, 2.6 TFLOP/s) | 42.8 ms | **0.34 s** |
| weight memory (re-read ×5, 60 GB/s) | 18.3 ms | 0.15 s |

Ideal transformer ≈ **0.34 s**. Measured on Mali = **15.0 s** (8 forwards) ⇒
`0.34/15.0 ≈ **2.3%** efficiency`.

### VQGAN FLOPs (≈188 GFLOP)

The decoder is a conv pyramid: each level runs `(num_res_blocks+1) = 3` residual blocks ×
`2` convs each, every conv a `3×3`, `C→C` over an `R×R` map = `R²·C²·9` MACs. Summing
`18·R²·C²·9`... per level over `R²·C²·9 × (3·2)` for the levels `(R,C)`:

| level (R, C) | conv MAC = `6·R²·C²·9` |
|---|--:|
| (16, 512)  | 0.7 G |
| (32, 256)  | 0.7 G |
| (64, 256)  | 2.9 G |
| (128, 128) | 2.9 G |
| (256, 128) | 11.6 G |

`vq_FLOP = 2·Σ ≈ **188 GFLOP**`, dominated by the high-resolution low-channel tail
(`R=256`). The F32 conv weights (~72 M params, but stored/streamed F32) are ~288 MB; at
60 GB/s that is only ~5 ms, and `I` is far above the ridge, so VQGAN is **deeply
compute-bound**. Ideal (fp32, 1.3 TFLOP/s) ≈ **0.14 s** vs measured **6.3 s** ⇒ **~2%**.

### Hardware-spec caveat

The Mali-G715 / Tensor-G4 numbers (~1.3 TFLOP/s fp32, ~2.6 TFLOP/s fp16, ~60 GB/s) are
**vendor-undocumented order-of-magnitude estimates** — ARM/Google publish neither the
shader-core FLOP/s nor the effective LPDDR5X bandwidth for this part. They are good enough
because the *conclusion is robust to large error*: even if peak compute were 2× lower
(0.7 s ideal) or bandwidth 2× lower (0.30 s weight reads), the ideal is still **≪ 1 s** and
the achieved efficiency is still **low single digits**. The 60–100× gap to measured swamps
any 2× spec uncertainty.

### The efficiency gap, attributed per cause

We achieve only **~2–7%** of peak. For contrast, XNNPACK int8 on the same CPU
(~0.7 s ideal at ~0.6 TOP/s i8mm, ~3 s measured) runs at **~25%** of *its* roofline.

The from-scratch GPU kernels lose efficiency to (each tied to the per-op profile in
`results/device.md`):

- **No native int8 matmul.** `MulMat(q)` is **44%** of device time (13.3 s, 1160
  dispatches). The CPU's KleidiAI uses ARMv8.6 `SMMLA` (i8mm: an int8 8×8 outer-product
  per instruction); our GPU kernel *dequantizes* every weight to fp16/fp32 and does float
  FMAs — for Q8_0 that is one extra mul-add per loaded weight (≈2× the inner-loop ALU),
  worse for Q4_K's super-block unpack — and **none of that dequant ALU is in the 111 GFLOP
  count**, so the *real* compute the GPU performs is well above the roofline numerator.
- **Small-M (257 tokens).** A 257-row GEMM barely fills the GPU and **wastes the last
  64-wide tile**: `ceil(257/64)=5` row-tiles cover 320 rows, so the 5th tile is ~98% idle
  lanes, and even the full tiles under-fill a GPU that wants thousands of in-flight
  work-items. The CPU's big caches + few strong cores handle small-M far better — this is
  the main reason the *phone* CPU beats the *phone* GPU on the same model.
- **No fusion / one kernel per graph node.** The forward graph is **~829 ops** (cf. the
  825/961-count internal nodes and the `Add` 2003 / `Norm` 400 / `Mul` 425 dispatch counts
  in the profile); each kernel writes its result to global memory and the next re-reads it.
  XNNPACK instead fuses bias+activation+residual and keeps tensors in cache/registers, so
  it never pays those round-trips. `Add` alone is **5%** (1655 ms) and `Mul`/`Gelu`/`Silu`
  another ~3% — almost pure memory-traffic overhead a fused library would erase.
- **Un-fused small norm/softmax kernels + DVFS.** `GroupNorm` 6% + `Norm` 4% + `SoftMax`
  3% are each tiny launches that don't amortize dispatch/occupancy ramp, and Mali's
  DVFS/thermal governor holds the sustained clock below the peak the roofline assumes.
  Together these drag the *average* efficiency down to ~2%.

This is why a mature **fused int8 library (XNNPACK) reaches ~25%** while our from-scratch
per-node float kernels reach **~2%**: same silicon, ~10× the software maturity.

### CPU contrast (int8 i8mm)

The transformer's int8 MAC work over 8 steps is `FC_MAC·L·8 ≈ 426 G` MACs. The Tensor-G4
big core(s) with `i8mm`/`SMMLA` sustain ~0.6 TOP/s int8 (conservative, 1–2 cores), giving
an **ideal ~0.7 s** vs **~3 s** measured ⇒ **~25%** of its roofline — an order of magnitude
better utilization than the GPU's float path, on the very same chip.

**Implication for M6:** the remaining wins are **kernel fusion** and a **GPU int8 dot
path** (`dot8` / `cl_arm_integer_dot_product`) — both large efforts — not more tile tuning.
Run `python3 benchmark/roofline.py` to reproduce these estimates.

---

The benchmark/profiling tool is built into the runtime as a mode:

```
mg-generate -m model.gguf --backend opencl|xnnpack|reference [--quant ...] \
            --bench --n-runs N --warmup K
```

It reports, for one config: model-load time; end-to-end latency percentiles
(p50/p90/p99, mean±sd, min) over N runs; a **per-component** breakdown (transformer
forwards / host sampling / VQGAN decode, from `GenStats`); peak RSS; and, for the
OpenCL backend, a **per-op-type GPU profile**. The per-op profile runs one extra
`clFinish`-serialized pass (so its absolute total inflates vs the real overlapped run,
but the *relative* split is reliable), and splits `MulMat` into the quantized FC
(`MulMat(q)`) vs the F32 attention/conv-matmul (`MulMat(f32)`).

All numbers below: class 207, seed 42, 8 steps, 256×256.

## End-to-end (per the table in the README)

| backend / precision | host M1 Max | Pixel 9 Mali |
|---|--:|--:|
| OpenCL ggml Q8_0 | 2.8 s | 26 s |
| OpenCL ggml Q4_K | 2.9 s | 27 s |
| XNNPACK int8 (CPU) | 3.9 s | 4.2 s |

## Where the time goes — per-op GPU profile

**Pixel 9 / Mali-G715 / gq8** (the deployment target):

| op | dispatches | ms | % |
|---|--:|--:|--:|
| **MulMat(q)** — quantized FC | 1160 | 13295 | **44%** |
| **Conv2D** — VQGAN decoder | 32 | 9078 | **30%** |
| GroupNorm — VQGAN | 25 | 1704 | 6% |
| Add (residuals/bias) | 2003 | 1655 | 5% |
| MulMat(f32) — attention | 392 | 1512 | 5% |
| Norm (LayerNorm) | 400 | 1156 | 4% |
| SoftMax | 192 | 809 | 3% |
| Mul, Gelu, Silu, …, views | — | <1.4 s | ~6% |

**M1 Max / gq8** (for contrast — VQGAN dominates more, since the FC is already fast there):

| op | ms | % |
|---|--:|--:|
| Conv2D | 885 | 23% |
| MulMat(q) | 817 | 22% |
| GroupNorm | 778 | 20% |
| Add | 559 | 15% |

Component view (Mali gq8): transformer 57% (15.0 s) / VQGAN 42% (11.0 s) / sampling 1%.

## Findings → targets

1. **MulMat(q) = 44% on device.** This is the quantized FC, already the most-optimized
   kernel (tiled → 4×4 register micro-tile → fp16 for Q8_0). It is far from Mali's peak,
   most likely **register-pressure-limited**: the 4×4 micro-tile uses 16 accumulators per
   work-item, which suits the register-rich M1 but caps Mali occupancy (Mali gained far
   less from micro-tiling than the host did). Next: autotune the tile (smaller micro-tile
   / workgroup) for Mali.
2. **Conv2D = 30% on device, and it is still the naive one-thread-per-output-pixel
   kernel** (`k_conv2d`) — completely unoptimized, re-reading inputs and weights with no
   reuse. This is the clearest remaining win: a tiled / local-memory direct conv (im2col
   is memory-prohibitive at 256×256). High value, well-understood method.
3. **The F32 attention matmul is only 5%.** Optimizing it (which earlier prose wrongly
   called the bottleneck) would barely move end-to-end — the tool corrected that guess.
4. **GroupNorm (6%) + Norm (4%) + SoftMax (3%)** are each modest; not worth it before the
   two big items.

**Conclusion:** the next two optimizations, in priority order, are (a) a tiled VQGAN
**Conv2D** (30%, wide open) and (b) Mali tile-autotuning for the **quantized FC** (44%,
diminishing). The CPU stays ahead on device because XNNPACK's KleidiAI i8mm int8
micro-kernels are purpose-built for this small-M GEMM; the GPU is competitive on host.

## M6 hill-climb log

| # | optimization | host effect (cosine) | device effect |
|---|---|---|---|
| 1 | **Tiled implicit-GEMM Conv2D** (`k_conv2d_t`) — replaces the naive direct conv; 16×16 local-memory tile, im2col column gathered on the fly | VQGAN decode 1.77→**1.29 s**; Conv2D 23%→**12%** (885→416 ms); end-to-end gq8 2.84→**2.36 s**; cosine **1.0** | **Conv2D 30%→18%** (9.1→**4.4 s**); VQGAN 11→**6.3 s**; **end-to-end gq8 26.2→21.5 s** |
| 2 | **`MulMat(q)` tile-autotune** — micro-tile (WPTM×WPTN) made build-time tunable (`-DGEMM_WPTM/N`, env `MG_GEMM_WPTM/N`), single source of truth shared by kernel + dispatch | n/a | **no win — 4×4 is already optimal.** Mali gq8 forward sweep: 4×4 **3.73 s** (best), 6×6 3.82, 4×8 3.84, 2×4 3.87, 8×4 3.91, 4×2 4.18, 8×8 5.62. Register-pressure hypothesis disproved; reuse/intensity wins. |
| 3 | **Matmul-epilogue fusion** (`mul_mat_ex`) — fold bias-add + GELU/SiLU + residual-add into the matmul kernel (reference + all OpenCL matmul variants); removes those as separate graph nodes | ~4% (gq8 2.36→2.26 s); cosine **1.0** | **~wash.** `Add` 1655→538 ms and `Gelu` removed (−~1.3 s) but `MulMat(q)` rose 13.3→14.2 s (epilogue now runs in the matmul) → net ≈ −0.4 s (~2%), within thermal noise (e2e 21.5→22.5 s). |

**M6 #3 conclusion:** fusion is correct (≈1550 `Add` + all `Gelu` launches removed, cosine
unchanged) and architecturally cleaner, but only **marginally** faster — exactly as the
roofline predicts. The bottleneck is `MulMat(q)`'s dequant *compute* (56% of device
time), which fusion doesn't touch; it only relocates the cheap ~15% periphery, and the
epilogue work (bias/residual loads, GELU eval) partly offsets the launch/round-trip
savings. Kept for the cleaner graph; the real lever remains a GPU int8-dot path.

| 4 | **int8-dot matmul** (`arm_dot_acc`) for Q8_0 — quantize the activation to int8 and use Mali's `cl_arm_integer_dot_product_accumulate_int8` (4 int8 MACs/op, the GPU analog of i8mm) instead of dequant→float; new `k_quantize_q8` + tiled `k_mul_mat_q8_i8` | n/a (Mali-only ext) | **big win.** cosine 0.99999929. Single forward only ~14% (2.9 vs 3.5 s) but over the 8-step loop **transformer 16.1→6.4 s (2.5×), end-to-end gq8 22.4→12.8 s (1.75×)** |

| 5 | **int8 VQGAN conv** (`k_conv2d_i8`, `arm_dot`) — pre-quantized int8 weights + per-tensor-quantized input, implicit-GEMM gather as int8 | n/a (Mali) | **21% but lossy.** matched-thermal: Conv2D 4.38→1.56 s, end-to-end gq8 12.8→**10.1 s**; BUT VQGAN cosine 1.0→**0.9984** (per-tensor activation too coarse; ≈ XNNPACK int8 conv 0.9994). **Off by default** (`MG_ARM_CONV=1`). |

| 5b | **per-block activation quant for the int8 conv** — gather-time per-(pixel, 32-block) inside `k_conv2d_i8` (replaces the per-tensor scale; removes the amax/quantize pre-passes) | n/a | **fixes the accuracy.** cosine 0.9984→**0.99997**; matched-thermal end-to-end **12.8→9.7 s**; **now default-on** (`MG_NO_ARM_CONV=1` opts out). |
| 6 | **Workgroup-parallel reductions** for `k_norm` / `k_soft_max` / `k_group_norm` — 64/256-thread workgroups with local-memory tree reduction (the originals were 1 thread/row sequential; GroupNorm did ~262 k elements per thread at 256×256) | n/a | **big.** GroupNorm 1713→**189 ms** (9.1×), Norm 1409→**337 ms** (4.2×), SoftMax 1025→**385 ms** (2.7×), VQGAN 3.1→**1.66 s**, **end-to-end gq8 9.79→7.08 s (−28%)**, cosine bit-identical. |
| — | **probe**: `cl_arm_matrix_multiply` Mali built-ins are `arm_matrix_multiply(char4,char4,int)` etc. — the *same* 4-MAC int8 dot as `arm_dot_acc`. No wider matmul primitive on this Mali. | — | informational |
| 7 | **F32 attention matmul** — two cheap attempts: fp16 cast on load, and TSK 16→32 (halve barriers) | n/a | **negative result.** fp16 regressed (+31%); TSK=32 a wash. Attention is overhead-bound, not ALU/barrier-bound — further wins need flash-attention-style fusion. (DEEP_DIVE §13.3 Step 9.) |
| — | **XNN per-op profile added to `--bench`** (`XNN_FLAG_BASIC_PROFILING`); enables the side-by-side comparison below. | — | tooling |

### M6 finale — CPU vs GPU per-op (Mali-G715 GPU vs XNNPACK i8mm CPU, Pixel 9, gq8, 1 run profile)

| op | CPU int8 (ms) | GPU int8 (ms) | **GPU/CPU** | room? |
|---|--:|--:|--:|---|
| **MulMat — FC** | **2019 (48%)** | **5995 (51%)** | **3.0×** | HW int8 ceiling (Mali only has 4-MAC `arm_dot`; CPU `SMMLA` is 64 MAC/instr) |
| **MulMat — attention** | 582 (14%) | 2089 (18%) | **3.6×** | overhead-bound; needs flash-attention fusion |
| Conv2D | 521 (13%) | 1303 (11%) | 2.5× | already int8 per-block; same int8 ceiling |
| Deconv (upsample) | 245 (6%) | — | n/a | XNN fuses these; we use a separate `Upscale` op |
| **Activation** (Gelu/Silu) | 209 (5%) | **155** | **0.7× ✓** | GPU slightly *wins* |
| SoftMax | 186 (4%) | 350 (3%) | 1.9× | small absolute |
| Mul/Scale | 123 (3%) | 425 (4%) | 3.5× | small absolute |
| Norm + GroupNorm | 79 (2%) | 508 (5%) | 6.4× | small absolute; XNN fuses; ours is now workgroup-parallel |
| Add | 75 (2%) | 548 (5%) | 7.3× | small absolute; XNN fuses bias-adds into the matmul epilogue |
| **end-to-end p50** | **4.1 s** | **7.1 s** | **1.7×** | |

### Why the gap doesn't close further (and when it would)

This is a small-M *prefill* — every transformer step processes all 257 tokens in parallel, which is the regime where CPUs win and GPUs are at their worst:

- **Hardware peak int8 throughput** on this chip: Mali-G715 MP7 ~1 TOPS via `arm_dot_acc` (4 MAC/instr); Cortex-X4/A720 with `SMMLA` ~3 TOPS (64 MAC/instr). The ~3× per-op gap we measure is the chip's actual int8 throughput ratio — not kernel inefficiency.
- **Small M=257** under-utilizes the GPU's parallelism while still paying the launch / sync / memory overhead.
- **Attention is O(M²)** — but at M=257 it's only ~5% of compute and the per-head shapes are too small for the GPU to fill its cores.

At longer prefills (M ≥ 1024, e.g. MaskGIT-512×512) the launch overhead amortizes and attention's O(M²) makes the GPU's parallelism start to pay off; the gap is expected to close, and may flip near M ≥ 2048. At decode-style M=1 the GPU loses by much more — but MaskGIT doesn't decode autoregressively.

> **Empirical correction, then flip (2026-06-03).** Naive attention: the
> ratio plateaus at ~1.74× from M=257 → M=1025; the GPU never closes.
> Implementing **tiled flash-attention** (the kernel pattern LiteRT-LM
> relies on) flips this entirely:
>
> | M | CPU XNNPACK | GPU baseline | GPU flash-attn | FA / CPU |
> |---:|---:|---:|---:|---:|
> | 65   | 773 ms     | 1 733 ms    | 1 697 ms    | 2.20× |
> | 257  | 2 985 ms   | 5 157 ms    | 4 882 ms    | 1.64× |
> | 1025 | 22 198 ms  | 38 693 ms   | 23 405 ms   | **1.05× (tied)** |
> | 4097 | 319 657 ms | OOM         | 147 628 ms  | **0.46× (GPU 2.17× FASTER)** |
>
> Flash-attention reduces DRAM traffic ~26× per layer at M=1025 by
> keeping the M² scores tensor in workgroup-local memory; the naive
> path wrote, read, and re-wrote that tensor to global memory. The
> attention block alone drops from 32 s (MulMat+SoftMax) to 6.5 s
> (single fused FlashAttn op).
>
> **Crossover at M ≈ 1025, GPU decisively wins at M ≥ 4097.** The
> original M6 wrap statement "for this class of phone, the CPU is the
> right tool" held *with the kernels we had*. With flash-attention
> shaders, **the GPU becomes the right tool for the prefill regime that
> real workloads will see** (LLMs with 1k+ prompt tokens, larger
> generative image models). See
> [`benchmark/seqlen-sweep/README.md`](seqlen-sweep/README.md) for the
> flash-attn implementation, per-op profile, and DRAM-traffic analysis.

For this model on this chip, **the CPU is the right tool**. The GPU work in M6 is still valuable: device gq8 went **111 s → 7.1 s** (a 16× improvement), with the cosine-clean accuracy-safe int8 path, all gated and reversible. The journey + the negative results are the most transferable artifact (see DEEP_DIVE §13).

**M6 #5 conclusion:** the int8 conv is a real 21% end-to-end win but hits an accuracy
wall — a per-tensor activation scale clips the wide-magnitude im2col blocks (the FC
avoided this via per-32-block activation quant, which the conv's implicit im2col makes
hard). Gated off to preserve the F32 conv's cosine-1.0 image quality; per-block/per-column
gather-time activation quant is the path to ship it by default. (Debug: a cold run again
mis-ranked it *slower* end-to-end — only the matched bench A/B showed the 21%.) Component
gap that motivated it: GPU VQGAN 6.2 s vs CPU 0.7 s (9×, because the CPU conv is int8);
GPU transformer 5.8 s vs CPU ~3 s (2×, the small-M gap).

**M6 #4 conclusion:** the int8-dot path is the largest M6 win. The *per-forward* gain is
modest, but the int8 datapath draws less power, so over the sustained 8-step decode loop
it avoids the **thermal throttling** that crushes the fp16 path — the win compounds
(14% → 2.5×). A cautionary measurement note: a single cold-device forward first showed
int8 *slower* (4.1 vs 3.6 s); only a matched-thermal multi-run A/B revealed the real
ranking — single-shot device timings are unreliable. After #4 the device cost splits
~50/50 between the transformer and the (already-tiled) VQGAN conv. Off-by-default escape
hatch: `MG_NO_ARM_DOT=1`. Q8_0 only (Q4_K/F32 keep the dequant path).

**M6 #2 conclusion:** the quantized-FC micro-tile is already at its sweet spot (4×4);
tile geometry is exhausted. The remaining MulMat(q) gap to the CPU is *structural*, not
a tuning issue: the GPU dequantizes to float and does float FMAs, whereas the CPU runs
native int8 matmul (ARMv8.6 `SMMLA`/i8mm). Closing it would need a GPU int8 dot-product
path (e.g. `cl_arm_integer_dot_product` / `dot8`) — a large, separate effort with
uncertain Mali support. The build-time tile param + env override remain as reusable
autotuning infrastructure (default 4×4).

After #1 the **device** profile re-ranks (Mali gq8): **MulMat(q) 52%** (12.8 s), Conv2D
18% (4.4 s), GroupNorm 7%, MulMat(f32) 6%. The quantized FC is now the clear #1 — the
next lever is **Mali tile-autotuning for `MulMat(q)`** (the 4×4 micro-tile = 16
accumulators is register-pressure-limited on Mali). Conv2D is no longer worth a second
pass; GroupNorm (7%) is small.

## Reproduce

Host sweep: `RUNS=8 ./benchmark/run_bench.sh` → `benchmark/results/host.md`.
Device: build `scripts/build_android_generate_opencl.sh`, `adb push`, then run the same
`--bench` flags on the phone (see `run_bench.sh` comments). Raw captures are in
`benchmark/results/`.
