# MaskGIT On-Device — Custom C/C++ Runtime

Bringing [MaskGIT](https://github.com/google-research/maskgit) (class-conditional
ImageNet image generation) to a from-scratch **C++17** on-device inference runtime
for Android (ARM CPU + GPU), using ggml / stable-diffusion.cpp as architectural
references. Full design and milestone roadmap: [`CLAUDE.md`](CLAUDE.md).

## Status

| Milestone | State |
|---|---|
| **M1 — Reference oracle** | ✅ PyTorch, signed off ([`reference/`](reference/)) |
| **M2 — C/C++ runtime & kernels** | ✅ host: tensor lib, F32 kernels, GGUF, transformer + VQGAN, decode loop, CLI, **XNNPACK backend** · ⬜ Android, GPU, quantization |
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

## Performance (host, Apple M1 Max, class 207, 8 steps)

| Stage | Reference kernels | XNNPACK | Speedup |
|---|---|---|---|
| Transformer (×8 decode steps) | ~432 s | ~11 s | ~40× |
| VQGAN decode | ~326 s | 2.8 s | ~115× |
| **Full generation** | **762 s** | **13.6 s** | **~56×** |

Reference and XNNPACK outputs agree to within ±1 of 255 per pixel (rounding).

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
