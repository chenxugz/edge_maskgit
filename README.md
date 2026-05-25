# MaskGIT On-Device — Custom C/C++ Runtime

Bringing [MaskGIT](https://github.com/google-research/maskgit) (class-conditional
ImageNet image generation) to a from-scratch **C++17** on-device inference runtime
for Android (ARM CPU + GPU), using ggml / stable-diffusion.cpp as architectural
references. Full design and milestone roadmap: [`CLAUDE.md`](CLAUDE.md).

**Docs:** [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) — implementation deep dive
(tensor library, kernels, transformer/VQGAN graphs, XNNPACK backend, quantization,
verification, and the latency/memory rationale). [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md) — open items.

## Status

| Milestone | State |
|---|---|
| **M1 — Reference oracle** | ✅ PyTorch, signed off ([`reference/`](reference/)) |
| **M2 — C/C++ runtime & kernels** | ✅ tensor lib, F32 kernels, GGUF, transformer + VQGAN, decode loop, CLI · ✅ **XNNPACK backend** (int8/int4) · ✅ **Android** (arm64-v8a) · ✅ **OpenCL GPU backend** (ggml Q8_0/Q4_K), full pipeline on host + Pixel 9 Mali-G715 |
| M3 — Numerical verification | 🔄 component-boundary verification in place (see below); per-layer harness pending |
| M4 — Evaluation framework | 🔲 not started |
| M5 — Benchmark tool | 🔲 not started |

### Deviations from the roadmap (intentional)

- **PyTorch reference, not JAX.** Google's `gs://maskgit-public` checkpoint bucket
  was decommissioned (`AccessDenied`), so M1 uses the genuine **official weights
  converted to PyTorch** ([hmorimitsu/maskgit-torch](https://github.com/hmorimitsu/maskgit-torch)).
  The model is identical. See [`reference/README.md`](reference/README.md).
- **Build system is Bazel**, not CMake (`bazelisk` via bzlmod).
- The from-scratch reference kernels are the default backend and **correctness
  oracle**; **XNNPACK** is an opt-in accelerated backend (`--backend xnnpack`).

## Results: latency, memory & samples

The journey from the from-scratch F32 reference kernels → XNNPACK F32 → int8/int4
quantization → on-device → small quantized model files. All runs generate **class
207 (golden retriever), seed 42, 8 steps, 256×256**; images are in [`samples/`](samples/).

| Kernel | Precision | Model file | Machine | Latency | Peak RSS | Sample |
|---|---|---|---|---|---|---|
| reference (scalar) | F32 | 775 MB | M1 Max | 731 s | 5451 MB | `dog207_reference_f32_host.png` |
| XNNPACK | F32 | 775 MB | M1 Max | 13.8 s | 2398 MB | `dog207_xnnpack_f32_host.png` |
| XNNPACK | **int8** | 288 MB | M1 Max | **3.9 s** | **928 MB** | `dog207_xnnpack_int8_host.png` |
| XNNPACK | **int4** | 207 MB | M1 Max | **4.1 s** | **767 MB** | `dog207_xnnpack_int4_host.png` |
| XNNPACK | F32 | 775 MB | Pixel 9 | 15.4 s | 1977 MB | `dog207_xnnpack_f32_device.png` |
| XNNPACK | **int8** | 288 MB | Pixel 9 | **4.2 s** | **839 MB** | `dog207_xnnpack_int8_device.png` |
| XNNPACK | **int4** | 207 MB | Pixel 9 | **4.0 s** | **676 MB** | `dog207_xnnpack_int4_device.png` |
| OpenCL (GPU) | F32 | 775 MB | M1 Max | 27.3 s | 4699 MB | `dog207_opencl_f32_host.png` |
| OpenCL (GPU) | **ggml Q8_0** | 298 MB | M1 Max | **5.6 s** | 4698 MB | `dog207_opencl_gq8_host.png` |
| OpenCL (GPU) | **ggml Q4_K** | 216 MB | M1 Max | **5.7 s** | 4700 MB | `dog207_opencl_gq4_host.png` |
| OpenCL (GPU) | F32 | 775 MB | Pixel 9 (Mali) | 347 s | 4633 MB | `dog207_opencl_f32_device.png` |
| OpenCL (GPU) | **ggml Q8_0** | 298 MB | Pixel 9 (Mali) | 111 s | 3953 MB | `dog207_opencl_gq8_device.png` |
| OpenCL (GPU) | **ggml Q4_K** | 216 MB | Pixel 9 (Mali) | 76 s | 4262 MB | `dog207_opencl_gq4_device.png` |

- **M1 Max** = macOS arm64 host; **Pixel 9** = Tensor G4, Android 16 (`adb`); **Pixel 9
  (Mali)** = the same phone's Mali-G715 GPU via OpenCL.
- Latency is end-to-end (class id → PNG); peak RSS via `getrusage`.
- The reference row's 5.4 GB is mostly its unoptimized bump-allocator arenas
  (1.5 GB transformer + 3 GB VQGAN, never freed mid-graph) — an artifact of the
  correctness-first reference path, not a fundamental requirement.
- XNNPACK quantized rows load the small quantized GGUF (transformer FC stored
  int8/int4; VQGAN conv kept F32 and quantized on load — see the quantization note
  below). The OpenCL rows use **ggml block quantization** (Q8_0 / Q4_K, dequant-fused
  in the matmul kernel) with VQGAN conv in F32; `--backend opencl` runs the *whole*
  pipeline (8 transformer steps + VQGAN) on the GPU.
- **GPU peak RSS is high (~4.7 GB)** because the OpenCL path reuses the reference
  graph builders and their conservative arenas (the 1.5 GB + 3 GB above) *plus* the
  mmap'd weights *plus* GPU buffers (unified memory on M1) — it is not memory-tuned
  like the XNNPACK path. On host the GPU also trails XNNPACK CPU (5.6 s vs 3.9 s):
  the kernels are still one-thread-per-output with a full VQGAN readback. On Mali the
  naive kernels are far from the CPU (Q4_K 76 s vs XNNPACK int4 4 s) — local-memory
  tiling is the lever (see the GPU section). The point of these rows is that the
  from-scratch OpenCL backend runs the **entire** pipeline correctly on a phone GPU,
  not that it is yet the fast path.
- Reference vs XNNPACK F32 is bit-identical (the two F32 samples are the same file).
  Across precisions/backends/machines the *image composition* varies — tiny logit
  differences flip a few discrete token samples — but every sample is a clean,
  class-correct golden retriever.

### Sample gallery (Pixel 9)

XNNPACK CPU:

| F32 (15.4 s) | int8 (4.2 s) | int4 (4.0 s) |
|---|---|---|
| ![](samples/dog207_xnnpack_f32_device.png) | ![](samples/dog207_xnnpack_int8_device.png) | ![](samples/dog207_xnnpack_int4_device.png) |

OpenCL on the Mali-G715 GPU (full pipeline on-device):

| F32 (347 s) | ggml Q8_0 (111 s) | ggml Q4_K (76 s) |
|---|---|---|
| ![](samples/dog207_opencl_f32_device.png) | ![](samples/dog207_opencl_gq8_device.png) | ![](samples/dog207_opencl_gq4_device.png) |

### Numerical verification (vs the PyTorch oracle, fixed inputs)

Each component is run on a fixed input and compared to the PyTorch reference
(`reference/dump_verify.py` dumps; `tools/verify-*` reproduce these on the M1 Max
host). Metrics isolate kernel/quantization error from sampling RNG.

| Component | Kernel / precision | mean_abs_diff | cosine |
|---|---|---|---|
| Transformer logits | reference F32 (C++) | 2.6e-6 | 1.0000000 |
| Transformer logits | XNNPACK F32 | 2.6e-6 | 1.0000000 |
| Transformer logits | XNNPACK int8 | 2.9e-2 | 0.9999951 |
| Transformer logits | XNNPACK int4 | 2.0e-1 | 0.9998129 |
| VQGAN image | reference F32 (C++) | 5.2e-7 | 1.0000000 |
| VQGAN image | XNNPACK F32 | 4.1e-7 | 1.0000000 |
| VQGAN image | XNNPACK int8 conv | 1.5e-2 | 0.9993878 |

F32 (reference and XNNPACK) matches the oracle to ~1e-5 (`cosine = 1.0`). int8 is
near-lossless (`cosine ≈ 0.9999`); int4 keeps `cosine ≈ 0.9998` — enough that the
sampled tokens, and thus image class, track the F32 result. VQGAN conv is int8 in
both the q8 and q4 models (XNNPACK conv is int8-only), so VQGAN int4 == the int8 row.

## Prerequisites

```bash
# 1. Python reference env (M1 oracle + weight export + verification dumps)
conda create -n maskgit-ref python=3.11 -y && conda activate maskgit-ref
pip install -r reference/requirements.txt

# 2. Bazel (build system)
brew install bazelisk            # macOS; provides `bazel`

# 3. XNNPACK (only needed for --backend xnnpack); builds a static lib locally
./scripts/build_xnnpack.sh
```

## Build the model file

The runtime loads a GGUF file produced from the PyTorch checkpoints:

```bash
cd reference && ./download_checkpoints.sh           # ~880 MB of official weights
python export_weights.py --verify                   # -> reference/export/*.npz (+ lossless round-trip)
cd ..
python tools/convert_to_gguf.py                     # -> models/maskgit-256-f32.gguf
```

## Run

```bash
bazel build //:mg-generate

# reference kernels (slow, ~13 min) — the correctness oracle
./bazel-bin/mg-generate -m models/maskgit-256-f32.gguf --class-id 207 --seed 42 --steps 8 -o out.png

# XNNPACK backend (~14 s)
./bazel-bin/mg-generate -m models/maskgit-256-f32.gguf --class-id 207 --seed 42 --steps 8 --backend xnnpack -o out_xnn.png
```

## On-device (Android, arm64-v8a)

The runtime cross-compiles to Android with the NDK and runs as a standalone
`adb` binary. Validated on a **Pixel 9 (Tensor G4, Android 16)**: 15.1 s for the
XNNPACK backend, output **bit-identical** to the host.

```bash
# 1. Cross-compile XNNPACK for Android, then the runtime (NDK r27).
#    Set ANDROID_NDK_HOME if not at ~/Library/Android/sdk/ndk/27.0.12077973
./scripts/build_xnnpack_android.sh      # -> third_party/xnn/lib-android/
./scripts/build_android.sh              # -> build-android/mg-generate (arm64 PIE)

# 2. Push the binary + model and run on-device.
adb push build-android/mg-generate /data/local/tmp/
adb shell chmod 755 /data/local/tmp/mg-generate
adb push models/maskgit-256-f32.gguf /data/local/tmp/
adb shell "cd /data/local/tmp && ./mg-generate -m maskgit-256-f32.gguf \
    --class-id 207 --seed 42 --steps 8 --backend xnnpack -o dog.png"
adb pull /data/local/tmp/dog.png .
```

Notes: the binary is self-contained (`-static-libstdc++`). The F32 model is
~866 MB plus a few hundred MB of XNNPACK workspace — fine on an 8 GB+ device;
smaller devices will need the quantized models (next milestone). The Android
build currently uses a direct NDK clang script (`scripts/build_android.sh`); a
Bazel `--config=android` target is a planned convenience.

---

## Validating the runtime

Verification compares the C++ runtime against the PyTorch oracle (M1). The key
idea is to feed **fixed inputs** so kernel/graph correctness is isolated from the
stochastic sampling RNG (which differs between PyTorch and C++). Comparisons are
made at component **boundaries** — the full transformer logits and the full VQGAN
image — plus kernel-level unit tests and an end-to-end backend diff.

All commands run from the repo root (so the binaries see the checkpoint/export
paths). Build everything first: `bazel build //...`.

### 0. Generate the reference tensors (from the PyTorch oracle)

```bash
cd reference && python dump_verify.py && cd ..
```

Writes fixed-input reference tensors to `reference/export/` (raw little-endian):

| File | Shape | What it is |
|---|---|---|
| `verify_tokens.bin` | int32 `[S=257]` | fixed transformer input (class 207: label + all-mask) |
| `verify_logits.bin` | float32 `[S, vocab=2025]` | PyTorch transformer logits for that input |
| `verify_grid.bin` | int32 `[16, 16]` | fixed VQGAN token grid |
| `verify_image.bin` | float32 `[256, 256, 3]` | PyTorch VQGAN decode of that grid |

### 1. Kernel unit tests

Each F32 kernel is checked against an independent in-test reference (catches
stride/indexing bugs):

```bash
bazel test //:test_kernels --test_output=all
```

Expect `ALL PASS` (max diffs at F32 noise level, ≤ 4e-6).

### 2. GGUF loader round-trip

Confirms the loader reads back exactly what the converter wrote (per-tensor
float64 checksums + hyperparameters):

```bash
./bazel-bin/mg-model-info models/maskgit-256-f32.gguf
```

The printed checksums match those from `python tools/convert_to_gguf.py --check`.

### 3. Transformer output tensors (logits) vs PyTorch

Runs the C++ transformer on `verify_tokens.bin` and compares logits to
`verify_logits.bin` (max-abs diff, mean-abs diff, cosine similarity):

```bash
./bazel-bin/verify-transformer     models/maskgit-256-f32.gguf reference/export   # reference kernels
./bazel-bin/verify-xnn-transformer models/maskgit-256-f32.gguf reference/export   # XNNPACK subgraph
```

Pass criteria: `max_abs_diff < 2e-2` and `cosine > 0.99999`. Observed:
`max_abs_diff ≈ 1.9e-5`, `cosine = 1.00000000` for both backends.

### 4. VQGAN output tensors (image) vs PyTorch

Decodes `verify_grid.bin` and compares the image to `verify_image.bin`:

```bash
./bazel-bin/verify-vqgan     models/maskgit-256-f32.gguf reference/export   # reference kernels
./bazel-bin/verify-xnn-vqgan models/maskgit-256-f32.gguf reference/export   # XNNPACK subgraph
```

Pass criteria: `max_abs_diff < 2e-2`, `cosine > 0.99999`. Observed:
`max_abs_diff ≈ 8e-6 … 1.2e-5`, `cosine = 1.00000000`.

### 5. End-to-end backend diff

With a fixed seed, the reference and XNNPACK pipelines should produce the same
image. Swapping only the transformer is **bit-identical**; the full XNNPACK path
differs by at most ±1/255 on a handful of pixels (uint8 rounding of ~1e-6 float
differences):

```bash
./bazel-bin/mg-generate -m models/maskgit-256-f32.gguf --class-id 207 --seed 42 --steps 8                  -o out_ref.png
./bazel-bin/mg-generate -m models/maskgit-256-f32.gguf --class-id 207 --seed 42 --steps 8 --backend xnnpack -o out_xnn.png

python - <<'PY'
import numpy as np; from PIL import Image
a=np.asarray(Image.open("out_ref.png").convert("RGB")).astype(int)
b=np.asarray(Image.open("out_xnn.png").convert("RGB")).astype(int)
d=np.abs(a-b); print("max px diff", d.max(), " differing px", int((d.any(-1)).sum()), "/", a.shape[0]*a.shape[1])
PY
```

### GPU backend (OpenCL, experimental)

A from-scratch OpenCL backend (`src/mg-opencl/`) executes the same `mg::Graph` on
the GPU — one kernel per node, stride-aware `mul_mat`/`cont` so the attention's
permuted views run directly. The **full transformer and VQGAN** run on the GPU
and match the PyTorch oracle:

| Component | M1 Max GPU (OpenCL) | vs reference scalar | cosine |
|---|---|---|---|
| Transformer forward (F32) | 2.81 s | 54 s | 1.0000000 |
| Transformer forward (**ggml Q8_0**) | **0.54 s** | 54 s | 0.99999979 |
| Transformer forward (**ggml Q4_K**) | **0.55 s** | 54 s | 0.99995951 |
| VQGAN decode (F32) | 1.80 s | 326 s | 1.0000000 |

Intermediates stay on the GPU (only the graph output is read back), which is why
the quantized transformer is ~0.5 s. F32 / VQGAN are now compute-bound on the
naive (one-thread-per-output) `mul_mat`/`conv2d` — tiled/local-memory kernels are
the next perf lever.

**On-device (Pixel 9 Mali-G715):** the same OpenCL backend cross-compiles and runs
on the phone's GPU (`scripts/build_android_opencl.sh` for the component verifier;
`scripts/build_android_generate_opencl.sh` for the full `mg-generate -m … --backend
opencl`), linking the device's `libOpenCL.so`. The per-component forward passes are
cosine bit-identical to the host GPU, and the **entire class-id → image pipeline (8
transformer steps + VQGAN decode) now runs end-to-end on Mali** — see the OpenCL
rows in the results table above (Q4_K 76 s, Q8_0 111 s, F32 347 s; every output a
clean golden retriever).

| Transformer forward | Mali-G715 (naive) | Mali-G715 (**M-blocked**) | cosine |
|---|---|---|---|
| ggml Q8_0 | 18.7 s | **14.3 s** (−24%) | 0.99999979 |
| ggml Q4_K | — | **9.5 s** | 0.99995951 |

The quantized `mul_mat` kernels are **M register-blocked**: each work-item computes
4 output columns for one weight row, dequantizing each weight block *once* and
reusing it across the 4 activation columns — ¼ the weight-byte traffic, which is
the bottleneck on bandwidth-bound mobile GPUs. This is a per-device choice: the
desktop M1 Max has so many threads that the simpler one-thread-per-output kernel
(more parallelism) wins, so the runtime selects the scalar variant there and the
register-blocked `*_b4` variant only for Mali/Adreno — no host regression. Mali is
still slower than its CPU; local-memory tiling of the activation row is the next
lever. (Block factor 4 must be a compile-time constant so `acc[4]` register-allocates
and the inner loops unroll — a runtime factor spills the accumulator to private
memory and is ~5× slower.)

**ggml Q8_0 on GPU** (block-32, fp16 scale + 32 int8) is the block quantization
earmarked for the GPU path: a dequant-fused `mul_mat` kernel reads ¼ the weight
bytes, so it's ~4× faster than F32-on-GPU *and* more accurate than XNNPACK's
per-channel int8 (block-wise scales are finer). `python tools/convert_to_gguf.py
--quant gq8` (or `gq4` for Q4_K super-blocks: 256-weight blocks, 6-bit packed
scales/mins + 4-bit quants) writes the file; `verify-opencl-transformer
models/maskgit-256-gq{8,4}.gguf` reproduces it.

Reproduce component verification: `./bazel-bin/verify-opencl-transformer
models/maskgit-256-f32.gguf reference/export` and `verify-opencl-vqgan`; full
generation: `./bazel-bin/mg-generate -m models/maskgit-256-gq8.gguf --class-id 207
--backend opencl -o out.png`. Done: keep-intermediates-on-GPU, ggml Q8_0/Q4_K
dequant-fused matmul, M register-blocking, full on-device pipeline. Still naive
(one thread per output, full VQGAN readback) so it trails XNNPACK — local-memory
tiling of `mul_mat`/`conv2d` and a memory-tuned arena are the next levers.

### Intermediate (per-layer) tensors — current state & plan

Today verification is at component **boundaries** (full transformer logits, full
VQGAN image), which already pins down correctness because errors at any internal
layer propagate to these outputs (and the F32 kernels match the reference to
~1e-5). **Per-layer intermediate dumping is not yet implemented** — it is the
Milestone 3 work and the deferred `--dump-intermediates` flag:

- `reference/run_reference.py` will gain `--dump-intermediates` to save per-layer
  activations (post-norm, Q/K/V, attention scores/output, FFN, per-decoder-stage
  feature maps) to `.npy` with a canonical naming scheme.
- The C++ runtime will dump the same-named tensors from graph execution.
- `verification/verify.py` will match tensors by name and report per-layer
  max/mean diff + cosine against a tolerance spec (`CLAUDE.md` §M3).

Until then, to probe a specific intermediate you can add a temporary external
output to the relevant graph builder (`src/mg-transformer.cpp` / `src/mg-vqgan.cpp`)
and dump it the way `tools/verify-*.cpp` dump the final tensors.

---

## Repository layout

```
CLAUDE.md            design & milestone roadmap
MODULE.bazel .bazelrc BUILD.bazel   Bazel (bzlmod) build
scripts/build_xnnpack.sh            builds XNNPACK static lib -> third_party/xnn/

reference/           M1 — PyTorch oracle, weight export, dump_verify.py, reference images, tests
include/             mg-tensor / mg-model / mg-transformer / mg-vqgan / mg-generate / mg-xnn[-vqgan].hpp
src/                 tensor lib, F32 CPU kernels (mg-cpu/), GGUF loader, graph builders,
                     decode loop, XNNPACK subgraphs (mg-xnn*.cpp)
tools/               convert_to_gguf.py, mg-generate CLI (main.cpp), mg-model-info, verify-*.cpp
third_party/         stb (PNG writer), xnn (prebuilt XNNPACK — gitignored)
tests/               test_kernels.cpp (F32 kernel unit tests)
verification/        M3 — per-layer comparison harness (planned)
evaluation/ benchmark/   M4 / M5 (planned)
```
