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
| **M4 — Evaluation framework** | ✅ IS + top-1/top-5 + paired PSNR (one InceptionV3 pass) across reference / xnnpack-q8 / opencl-gq8 on Quick-5 — no ImageNet download ([`evaluation/`](evaluation/)) |
| **M5 — Benchmark tool** | ✅ `--bench` mode: latency percentiles + per-component + per-op profile for **both** GPU and XNNPACK ([`benchmark/`](benchmark/)) |
| **M6 — Perf hill-climbing** | ✅ device Q8_0 **111 s → 7.1 s** via 9 profiler-guided steps; the remaining 1.7× CPU gap is the chip's ~3× int8 throughput ratio ([`benchmark/analysis.md`](benchmark/analysis.md), [`docs/DEEP_DIVE.md`](docs/DEEP_DIVE.md) §13) |

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
| OpenCL (GPU, tiled) | F32 | 775 MB | M1 Max | 2.9 s | 30 MB | `dog207_opencl_f32_host.png` |
| OpenCL (GPU, tiled) | **ggml Q8_0** | 298 MB | M1 Max | **2.4 s** | 30 MB | `dog207_opencl_gq8_host.png` |
| OpenCL (GPU, tiled) | **ggml Q4_K** | 216 MB | M1 Max | **2.7 s** | 30 MB | `dog207_opencl_gq4_host.png` |
| OpenCL (GPU, tiled) | F32 | 775 MB | Pixel 9 (Mali) | 33 s | 2406 MB | `dog207_opencl_f32_device.png` |
| OpenCL (GPU, **int8-dot**) | **ggml Q8_0** | 298 MB | Pixel 9 (Mali) | **7.1 s** | 2228 MB | `dog207_opencl_gq8_device.png` |
| OpenCL (GPU, tiled) | **ggml Q4_K** | 216 MB | Pixel 9 (Mali) | 25 s | 1886 MB | `dog207_opencl_gq4_device.png` |

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
- The GPU numbers above are the result of a series of profiler-guided optimizations
  (M5/M6 — see [`benchmark/analysis.md`](benchmark/analysis.md)): tiled local-memory
  GEMM, 2D register micro-tiling + fp16 for the quantized FC, a **tiled implicit-GEMM
  VQGAN conv**, matmul-epilogue fusion, and an **int8 matmul via the ARM dot-product
  extension** (`arm_dot_acc` — the GPU analog of the CPU's i8mm — which quantizes the
  activation to int8 and uses Mali's native int8 datapath, for **both** the FC and the
  VQGAN conv), plus **workgroup-parallel reductions** for Norm/SoftMax/GroupNorm (the
  originals were one-thread-per-row sequential — 9× / 4× / 3× wins respectively).
  Together these took Mali end-to-end Q8_0 from 111 s (naive kernels) to **7.1 s**, and
  host Q8_0 to **2.4 s — faster than the XNNPACK CPU (3.9 s)**. The Tensor G4 CPU is
  still ~1.7× faster *on device* (KleidiAI i8mm, on a small-M workload that favors the
  CPU), but the gap has closed substantially. The int8 paths keep cosine ≥0.99997
  (per-block activation quant); `MG_NO_ARM_DOT` / `MG_NO_ARM_CONV` opt back to F32.
- **Peak RSS** was ~4.5 GB across the board until the arena was made non-zeroing: the
  `Context` bump-allocator used to zero-fill its whole (over-provisioned) 1.5 GB +
  3 GB buffers up front, so all of it was resident even though little is touched. With
  a raw `new uint8_t[]` (lazy pages) the OpenCL host RSS is **~30 MB** — the compute is
  on the GPU, so the CPU barely touches the arena, and on M1 the GPU buffers live in
  unified memory not counted in process RSS. On Android the GPU weight/activation
  buffers *do* count, so device RSS scales with model size (F32 2.4 GB, Q8_0 2.2 GB,
  Q4_K 1.9 GB). (The size-checked buffer cache also fixed an earlier host-F32 quirk
  where the 775 MB F32 weights were re-uploaded every step — host F32 dropped 26→2.9 s.)
- Reference vs XNNPACK F32 is bit-identical (the two F32 samples are the same file).
  Across precisions/backends/machines the *image composition* varies — tiny logit
  differences flip a few discrete token samples — but every sample is a clean,
  class-correct golden retriever.

### Sample gallery (Pixel 9)

XNNPACK CPU:

| F32 (15.4 s) | int8 (4.2 s) | int4 (4.0 s) |
|---|---|---|
| ![](samples/dog207_xnnpack_f32_device.png) | ![](samples/dog207_xnnpack_int8_device.png) | ![](samples/dog207_xnnpack_int4_device.png) |

OpenCL on the Mali-G715 GPU (full pipeline on-device, tiled GEMM):

| F32 (33 s) | ggml Q8_0 (7.1 s) | ggml Q4_K (25 s) |
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

Transformer forward (one full 24-layer pass), naive → **tiled** → **2D-micro-tiled**:

Transformer forward (one full 24-layer pass), single forward (cool device):

| Transformer forward | M1 Max GPU | Mali-G715 GPU | cosine |
|---|---|---|---|
| F32 (1-output tiled) | 2.81 → **0.38 s** | 18.7 → **6.7 s** | 1.0000000 |
| **ggml Q8_0** (2D-tiled fp16) | 0.54 → **0.28 s** | 18.7 → **3.5 s** | 0.99999979 |
| **ggml Q8_0** (int8-dot, Mali) | — | **2.9 s** | 0.99999929 |

The int8-dot single-forward gain is modest (~14%), but over the sustained 8-step decode
loop it is **2.5×** (transformer 16.1 → 6.4 s): the int8 datapath draws less power and
avoids the thermal throttling that throttles the fp16 path. The same int8-dot lever was
then applied to the **VQGAN conv** (per-(pixel,32-block) gather-time activation quant,
cosine 0.99997): Conv2D 4.4 → 1.25 s, VQGAN 6.2 → 3.1 s. End-to-end gq8 on Mali
**22 → 9.7 s**.
| **ggml Q4_K** (2D-tiled) | 0.55 → **0.30 s** | — → **3.9 s** | 0.99995951 |

VQGAN decode (F32, runs once) uses the tiled implicit-GEMM conv (§ below): **1.29 s**
host (was 1.77 s naive), cosine 1.0.

**Tiled local-memory matmul.** The naive kernel was one work-item per output, so
each weight row was re-read M times and each activation column N times (~2·N·K·M
global reads — pure memory-bound). The tiled kernels stage tiles of both operands in
local (shared) memory so each operand is read far fewer times (the classic GEMM
blocking). Two levels: the F32 attention matmul (`k_mul_mat_t`, K-stride and head
batch folded into the cooperative load) uses 16×16 tiles, 1 output/work-item; the
quantized FC (`k_mul_mat_q8_t2`/`k_mul_mat_q4k_t2`) adds **2D register micro-tiling**
— a 64×64 output tile per 16×16 workgroup, each work-item computing a 4×4 micro-tile
(loads 4 A + 4 B values into registers per step, does 16 FMAs → ~4× the arithmetic
intensity). Tiling helps desktop *and* mobile, so it replaced the earlier per-device
scalar/register-blocked selection. Largest win on bandwidth-bound Mali.

The VQGAN **conv** uses the same idea — a **tiled implicit-GEMM conv** (`k_conv2d_t`):
`out[oc,p] = Σ_k ker[oc,k]·col[k,p]` with the im2col column *gathered on the fly* into
local memory (no ~300 MB im2col buffer at 256×256). It cut the conv from 30→18% of
device time (9.1→4.4 s) and 23→12% on host. Net across all tiling stages: **end-to-end
Mali Q8_0 dropped 111→22 s**, host Q8_0 to **2.4 s — faster than the XNNPACK CPU
(3.9 s)**. Per the M5 profiler the device cost is now dominated by the quantized FC
matmul (52%), register-pressure-limited on Mali. **fp16** was tried for the FC matmul
(`cl_khr_fp16`-gated, Mali only): half local slabs + fp16 micro-tile multiply, fp32
accumulate. It speeds the isolated **Q8_0** FC forward ~14 % (4.3→3.7 s, cosine
identical) but *hurts* Q4_K (dequant-bound) and doesn't move end-to-end — confirming
the FC is no longer the bottleneck. The Tensor G4 CPU (KleidiAI i8mm int8) is still
~7× faster on device for this small-M (257-token) workload — see the note below on why
a phone GPU trails its CPU here. Next real lever: fp16 + tiling the **F32 attention
path** and the VQGAN conv. (The earlier scalar/`*_b4`/1-output `*_t` kernels remain in
the source for reference; the 2D `*_t2` kernels supersede them for quantized FC.)

**On-device (Pixel 9 Mali-G715):** the same OpenCL backend cross-compiles and runs
on the phone's GPU (`scripts/build_android_opencl.sh` for the component verifier;
`scripts/build_android_generate_opencl.sh` for the full `mg-generate -m … --backend
opencl`), linking the device's `libOpenCL.so`. Per-component forwards are cosine
bit-identical to the host GPU, and the **entire class-id → image pipeline (8
transformer steps + VQGAN decode) runs end-to-end on Mali** in 22–33 s (Q8_0 22 s,
Q4_K 25 s, F32 33 s) — see the OpenCL rows in the results table above; every output a
clean golden retriever.

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
dequant-fused matmul, **tiled local-memory GEMM (FC + attention)**, full on-device
pipeline. Next levers: register micro-tiling and fp16 compute for the matmul, a
tiled VQGAN conv (still direct/one-thread-per-pixel), and a memory-tuned arena.

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
