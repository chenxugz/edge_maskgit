# Host benchmark (run_bench.sh, RUNS=6)

## XNNPACK int8 (CPU)

### Benchmark: xnnpack / q8 / host CPU

| metric | value |
|---|---|
| model load | 2 ms |
| end-to-end p50 | 3802 ms |
| end-to-end p90 | 3813 ms |
| end-to-end p99 | 3813 ms |
| end-to-end mean +/- sd | 3792 +/- 19 ms |
| end-to-end min | 3767 ms |
| peak RSS | 876 MB |

**Component breakdown (mean per run, 8 steps):**

| component | ms | % |
|---|--:|--:|
| transformer (x8) | 3046 | 80% |
| sampling/masking | 6 | 0% |
| VQGAN decode | 739 | 19% |


## OpenCL ggml Q8_0 (GPU)

### Benchmark: opencl / gq8 / Apple M1 Max

| metric | value |
|---|---|
| model load | 7 ms |
| end-to-end p50 | 2844 ms |
| end-to-end p90 | 2862 ms |
| end-to-end p99 | 2862 ms |
| end-to-end mean +/- sd | 2841 +/- 14 ms |
| end-to-end min | 2818 ms |
| peak RSS | 36 MB |

**Component breakdown (mean per run, 8 steps):**

| component | ms | % |
|---|--:|--:|
| transformer (x8) | 1126 | 40% |
| sampling/masking | 6 | 0% |
| VQGAN decode | 1708 | 60% |

**GPU per-op-type profile (1 run, clFinish-serialized; relative split is the signal):**

| op | dispatches | ms | % |
|---|--:|--:|--:|
| Conv2D | 32 | 896 | 23% |
| MulMat(q) | 1160 | 850 | 21% |
| GroupNorm | 25 | 787 | 20% |
| Add | 2003 | 614 | 15% |
| Norm | 400 | 209 | 5% |
| MulMat(f32) | 392 | 181 | 5% |
| Mul | 425 | 136 | 3% |
| SoftMax | 192 | 129 | 3% |
| Gelu | 200 | 80 | 2% |
| op16 | 193 | 55 | 1% |
| Silu | 25 | 15 | 0% |
| GetRows | 9 | 8 | 0% |
| Upscale | 4 | 3 | 0% |
| op13 | 825 | 0 | 0% |
| op15 | 961 | 0 | 0% |


## OpenCL ggml Q4_K (GPU)

### Benchmark: opencl / gq4 / Apple M1 Max

| metric | value |
|---|---|
| model load | 5 ms |
| end-to-end p50 | 3045 ms |
| end-to-end p90 | 3085 ms |
| end-to-end p99 | 3085 ms |
| end-to-end mean +/- sd | 3050 +/- 22 ms |
| end-to-end min | 3030 ms |
| peak RSS | 41 MB |

**Component breakdown (mean per run, 8 steps):**

| component | ms | % |
|---|--:|--:|
| transformer (x8) | 1333 | 44% |
| sampling/masking | 7 | 0% |
| VQGAN decode | 1711 | 56% |

**GPU per-op-type profile (1 run, clFinish-serialized; relative split is the signal):**

| op | dispatches | ms | % |
|---|--:|--:|--:|
| MulMat(q) | 1160 | 1004 | 25% |
| Conv2D | 32 | 883 | 22% |
| GroupNorm | 25 | 777 | 20% |
| Add | 2003 | 556 | 14% |
| Norm | 400 | 198 | 5% |
| MulMat(f32) | 392 | 169 | 4% |
| Mul | 425 | 121 | 3% |
| SoftMax | 192 | 121 | 3% |
| Gelu | 200 | 73 | 2% |
| op16 | 193 | 50 | 1% |
| Silu | 25 | 14 | 0% |
| GetRows | 9 | 9 | 0% |
| Upscale | 4 | 3 | 0% |
| op13 | 825 | 0 | 0% |
| op15 | 961 | 0 | 0% |


