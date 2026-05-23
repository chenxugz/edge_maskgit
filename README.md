# MaskGIT On-Device — Custom C/C++ Runtime

Bringing [MaskGIT](https://github.com/google-research/maskgit) (class-conditional
ImageNet image generation) to a from-scratch C/C++ on-device inference runtime
for Android (ARM CPU + Vulkan/OpenCL GPU), using ggml / stable-diffusion.cpp as
architectural references. Full design and milestone roadmap: [`CLAUDE.md`](CLAUDE.md).

## Status

| Milestone | State |
|---|---|
| **M1 — Reference oracle** | ✅ implemented (see [`reference/`](reference/)) — PyTorch, pending user sign-off |
| M2 — C/C++ runtime & kernels | 🔲 not started |
| M3 — Numerical verification | 🔲 not started |
| M4 — Evaluation framework | 🔲 not started |
| M5 — Benchmark tool | 🔲 not started |

### Note on M1

The roadmap specifies a JAX/Flax reference, but Google's `gs://maskgit-public`
checkpoint bucket has been decommissioned (returns `AccessDenied`). M1 therefore
uses the genuine **official weights converted to PyTorch**
([hmorimitsu/maskgit-torch](https://github.com/hmorimitsu/maskgit-torch)). The
model is identical; only the framework differs. See [`reference/README.md`](reference/README.md).

## Quick start (M1)

```bash
conda create -n maskgit-ref python=3.11 -y && conda activate maskgit-ref
pip install -r reference/requirements.txt
cd reference && ./download_checkpoints.sh
python run_reference.py --class-id 207 --seed 42 --steps 8 -o output.png
```

## Repository layout

```
CLAUDE.md      design & milestone roadmap
reference/     M1 — PyTorch reference oracle, weight export, reference images, tests
include/       M2 — public C headers (mg-tensor.h, mg-backend.h, mg-model.h)
src/           M2 — tensor library, CPU/Vulkan/OpenCL backends, graph builders
tools/         M2 — GGUF converter, CLI
verification/  M3 — layer-by-layer numerical comparison harness
evaluation/    M4 — FID/IS evaluation framework
benchmark/     M5 — latency / memory / profiling
```
