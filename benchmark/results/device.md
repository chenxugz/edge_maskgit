# Device benchmark — Pixel 9 / Mali-G715 (adb)

Captured via `mg-generate-opencl --backend opencl --bench` (class 207, seed 42, 8 steps).

## OpenCL ggml Q8_0 (GPU)

### Benchmark: opencl / gq8 / Mali-G715 r0p0

| metric | value |
|---|---|
| model load | 6 ms |
| end-to-end p50 | 26170 ms |
| end-to-end p90 | 26302 ms |
| end-to-end p99 | 26302 ms |
| end-to-end mean +/- sd | 26187 +/- 66 ms |
| end-to-end min | 26099 ms |
| peak RSS | 2185 MB |

**Component breakdown (mean per run, 8 steps):**

| component | ms | % |
|---|--:|--:|
| transformer (x8) | 15014 | 57% |
| sampling/masking | 173 | 1% |
| VQGAN decode | 10999 | 42% |

**GPU per-op-type profile (1 run, clFinish-serialized; relative split is the signal):**

| op | dispatches | ms | % |
|---|--:|--:|--:|
| MulMat(q) | 1160 | 13295 | 44% |
| Conv2D | 32 | 9078 | 30% |
| GroupNorm | 25 | 1704 | 6% |
| Add | 2003 | 1655 | 5% |
| MulMat(f32) | 392 | 1512 | 5% |
| Norm | 400 | 1156 | 4% |
| SoftMax | 192 | 809 | 3% |
| Mul | 425 | 400 | 1% |
| Gelu | 200 | 224 | 1% |
| Silu | 25 | 176 | 1% |
| Cont | 193 | 119 | 0% |
| GetRows | 9 | 54 | 0% |
| Upscale | 4 | 39 | 0% |
| views (Reshape/Permute) | — | <0.1 | 0% |
