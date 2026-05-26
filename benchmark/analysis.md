# Milestone 5 — Benchmark & Profiling: analysis

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
