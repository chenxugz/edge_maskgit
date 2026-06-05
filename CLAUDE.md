# MaskGIT On-Device: Custom C/C++ Runtime — Design & Milestone Roadmap

## 1. Project Overview

This project brings [MaskGIT](https://github.com/google-research/maskgit) (Masked Generative Image Transformer) from its original JAX research implementation to a custom-built C/C++ on-device inference runtime targeting **Android devices**, with native ARM CPU and mobile GPU (Vulkan/OpenCL) kernel support. The runtime and compute kernels are built from scratch, using [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) as the architectural reference for the tensor library, backend abstraction, and quantization strategy, and [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) as the reference for CNN / Conv2D / VQGAN decoder patterns within the ggml ecosystem.

### Target Platform

| | Details |
|---|---|
| **Primary target** | Android devices (arm64-v8a) |
| **CPU** | ARM Cortex-A (NEON, dotprod, i8mm, SVE where available) |
| **GPU** | Qualcomm Adreno (via Vulkan / OpenCL), ARM Mali (via Vulkan / OpenCL) |
| **Build toolchain** | Android NDK (r26+) cross-compilation from Linux/macOS host |
| **Min API level** | Android 28 (Android 9.0) — Vulkan 1.1 guaranteed |
| **RAM budget** | 4–8 GB typical device, target < 1 GB runtime working set |
| **Development host** | Linux/macOS x86_64 for JAX reference + cross-compilation |

### Why Build From Scratch (Not ONNX)

- **Full kernel control.** Custom ARM NEON/SVE SIMD and Vulkan/OpenCL compute shader kernels tuned specifically for MaskGIT's operator mix — bidirectional self-attention over short sequences (256–1024 tokens) + CNN decoder. No dependency on a generic runtime's operator coverage or mobile GPU quirks.
- **Quantization flexibility.** Block-based quantization (Q4_K, Q8_0, etc.) applied at the weight level with custom dequantization fused into matmul kernels, following the ggml approach. Critical for fitting the ~250M parameter model into Android's tight RAM budget — Q4_K brings the model down to ~140 MB.
- **Single-binary deployment.** One self-contained `.so` library (or standalone binary for adb testing) with zero external dependencies, shipping a GGUF model file. Integrates into Android apps via JNI. No Python, no protobuf, no runtime installation.
- **Memory efficiency.** mmap-based weight loading, arena allocators for scratch buffers, and explicit control over memory layout — critical on Android where apps typically get 256–512 MB heap and must coexist with other processes.
- **Iterative decoding as first-class citizen.** The 8–16 step masked decoding loop is implemented in C++ with zero-copy tensor reuse between iterations, not fighting a graph-based runtime's session semantics.
- **Android GPU maturity control.** Mobile GPU drivers (especially Adreno and Mali) have known quirks with certain Vulkan features and FP16 precision. A custom runtime lets us work around driver-specific issues directly rather than hoping a generic runtime has the right fallbacks.

### Model Summary

| Component | Architecture | Parameters | Notes |
|---|---|---|---|
| VQGAN Tokenizer (Decoder only) | CNN: residual blocks + ConvTranspose2d upsampling | ~72M | Codebook: 1024 entries × 256 dim, 16× spatial upsample |
| Bidirectional Transformer | 24 layers, 768 hidden, 3072 FFN, 8 heads | ~174M (256×256) / ~176M (512×512) | Learnable pos embed, LayerNorm, GELU |
| **Total** | | **~246M / ~248M** | FP16 ≈ 500 MB, Q8_0 ≈ 262 MB, Q4_K ≈ 140 MB |

### End-to-End Pipeline

```
ImageNet Class ID (0–999)
        │
        ▼
┌─────────────────────────────────────────┐
│  Class Embedding + [MASK] Token Init    │
│  (256 or 1024 masked positions)         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
        ┌─── Iterative Loop (8–16 steps) ───┐
        │                                    │
        │  ┌──────────────────────────────┐  │
        │  │ Bidirectional Transformer    │  │
        │  │ 24L × Self-Attn + FFN       │  │
        │  │ → logits over 1024 codebook  │  │
        │  └──────────────┬───────────────┘  │
        │                 │                  │
        │  ┌──────────────▼───────────────┐  │
        │  │ Sampling + Masking Logic     │  │
        │  │ Gumbel noise, confidence     │  │
        │  │ top-k keep, re-mask rest     │  │
        │  │ (cosine schedule)            │  │
        │  └──────────────┬───────────────┘  │
        │                 │                  │
        └─────────────────┘                  │
                  │ final token grid         │
                  ▼                          │
┌─────────────────────────────────────────┐  │
│  VQGAN Decoder (CNN)                    │  │
│  Codebook lookup → ResBlocks → Upsample │  │
│  → 3-channel RGB output                 │  │
└─────────────────┬───────────────────────┘  │
                  │                          │
                  ▼                          │
           Output Image (256×256 or 512×512) │
```

---

## 2. Architecture Deep Dive

### 2.1 Tensor Library Layer (ggml-inspired)

Following the ggml design, the tensor library provides:

**Core abstractions:**
- `mg_tensor` — N-dimensional tensor (up to 4D) with stride-based layout, type tag, and backend pointer. Row-major storage. Supports non-contiguous views via stride manipulation (permute, transpose, reshape as zero-copy).
- `mg_context` — Arena allocator for tensor metadata. Lightweight, stack-allocated per computation graph.
- `mg_cgraph` — Computation graph (DAG of tensor operations). Built once, executed repeatedly across decoding iterations. Nodes represent operations, edges represent data dependencies.
- `mg_backend` — Plugin interface for hardware-specific kernel dispatch.

**Supported types:**
- `MG_TYPE_F32` — 32-bit float (reference precision)
- `MG_TYPE_F16` — 16-bit float (GPU-friendly, storage)
- `MG_TYPE_BF16` — bfloat16 (optional, for specific accelerators)
- `MG_TYPE_Q8_0` — 8-bit quantized, block size 32, 1 float scale per block
- `MG_TYPE_Q4_K` — 4-bit K-quant, super-blocks of 256 with FP16 scale/min
- `MG_TYPE_Q4_0` — 4-bit basic quantization, block size 32

**Operator set (what MaskGIT needs):**

The full operator inventory, split by which component needs them:

| Category | Operators | Used By |
|---|---|---|
| Embedding | `MG_OP_GET_ROWS` (codebook/token/pos lookup) | Transformer, VQGAN |
| Linear algebra | `MG_OP_MUL_MAT` (matmul), `MG_OP_MUL` (element-wise), `MG_OP_ADD` | Transformer (QKV proj, FFN, output proj) |
| Normalization | `MG_OP_NORM` (LayerNorm) | Transformer (pre-attention, pre-FFN) |
| Attention | `MG_OP_SOFT_MAX`, `MG_OP_SCALE`, `MG_OP_PERMUTE`, `MG_OP_CONT` | Transformer (self-attention) |
| Activation | `MG_OP_GELU` | Transformer (FFN) |
| Convolution | `MG_OP_CONV_2D` (via im2col + matmul), `MG_OP_CONV_TRANSPOSE_2D` (via col2im + matmul) | VQGAN decoder |
| Spatial | `MG_OP_UPSCALE` (nearest-neighbor upsample) | VQGAN decoder |
| Elementwise | `MG_OP_SILU`, `MG_OP_GROUP_NORM` | VQGAN decoder (ResBlocks) |
| Reshape | `MG_OP_RESHAPE`, `MG_OP_VIEW`, `MG_OP_TRANSPOSE` | Everywhere (zero-copy) |

**Key difference from llama.cpp:** MaskGIT does not use KV-cache (no autoregressive decoding). Each forward pass processes all token positions in parallel with full bidirectional attention. This simplifies memory management but means the attention computation is always over the full sequence.

### 2.2 Bidirectional Transformer

Architecture per layer (×24):

```
Input tokens [B, S, 768]        (S = 256 or 1024)
    │
    ├── LayerNorm
    │       │
    │   ┌───▼────────────────────────────────────┐
    │   │  Multi-Head Self-Attention (8 heads)   │
    │   │  Q = x·W_q  [B, S, 768]               │
    │   │  K = x·W_k  [B, S, 768]               │
    │   │  V = x·W_v  [B, S, 768]               │
    │   │  → reshape to [B, 8, S, 96]            │
    │   │  → attn = softmax(Q·K^T / √96) · V    │
    │   │  → reshape to [B, S, 768]              │
    │   │  → out·W_o  [B, S, 768]               │
    │   └───┬────────────────────────────────────┘
    │       │
    └── + ──┤  (residual connection)
            │
    ┌───────┤
    │       │
    │   LayerNorm
    │       │
    │   ┌───▼────────────────────────┐
    │   │  FFN                       │
    │   │  x·W1 [B, S, 3072] → GELU │
    │   │  → x·W2 [B, S, 768]       │
    │   └───┬────────────────────────┘
    │       │
    └── + ──┘  (residual connection)
            │
         Output [B, S, 768]
```

**Compute profile per forward pass (256×256, S=256, FP16):**
- Self-attention: 24 × (3 QKV projections + 1 output projection = 4 matmuls of [256, 768] × [768, 768]) + 24 attention score computations ([256, 96] × [96, 256] per head)
- FFN: 24 × (1 matmul [256, 768] × [768, 3072] + 1 matmul [256, 3072] × [3072, 768])
- Total dominant FLOPs: ~4.6 GFLOPs per forward pass, ×8 iterations ≈ 37 GFLOPs

### 2.3 VQGAN Decoder

The VQGAN decoder converts the predicted token grid back to pixel space. The architecture (from the official checkpoint and stable-diffusion.cpp VQGAN patterns):

```
Token indices [B, 16, 16]  (or [B, 32, 32] for 512×512)
    │
    ├── Codebook lookup → [B, 16, 16, 256]
    │
    ├── Conv2d (256 → 512, 3×3, pad 1) — post-quant conv
    │
    ├── ResBlock(512) × N
    │   └── GroupNorm → SiLU → Conv2d(3×3) → GroupNorm → SiLU → Conv2d(3×3) + skip
    │
    ├── Upsample (×2) + Conv2d(512 → 512, 3×3)  — stage 1: 16→32
    ├── ResBlock(512) × N
    │
    ├── Upsample (×2) + Conv2d(512 → 256, 3×3)  — stage 2: 32→64
    ├── ResBlock(256) × N
    │
    ├── Upsample (×2) + Conv2d(256 → 128, 3×3)  — stage 3: 64→128
    ├── ResBlock(128) × N
    │
    ├── Upsample (×2) + Conv2d(128 → 128, 3×3)  — stage 4: 128→256
    ├── ResBlock(128) × N
    │
    ├── GroupNorm → SiLU → Conv2d(128 → 3, 3×3)  — final output
    │
    └── Output: [B, 256, 256, 3] RGB image
```

**Conv2d implementation strategy** (referencing stable-diffusion.cpp and ggml):
- Standard Conv2d: im2col transform → matmul (leverage optimized matmul kernels)
- Upsample: nearest-neighbor interpolation (simple, no learned parameters)
- GroupNorm: custom kernel (mean/variance per group, affine transform)
- The VQGAN decoder runs once (not iteratively), so its latency is amortized

### 2.4 Iterative Masked Decoding (Host-Side C++)

The decoding loop is implemented in C++ host code, not inside the compute graph:

```cpp
// Pseudocode for the decoding loop
void maskgit_generate(mg_model* model, int class_id, int n_steps, int seed) {
    mg_rng rng(seed);
    
    // Initialize: all positions masked
    int S = model->n_tokens;  // 256 or 1024
    std::vector<int> tokens(S, MASK_TOKEN_ID);
    std::vector<bool> is_masked(S, true);
    
    for (int t = 0; t < n_steps; t++) {
        // 1. Build input tensor from current tokens + class embedding
        // 2. Run transformer forward pass (reuse same compute graph)
        mg_tensor* logits = transformer_forward(model, tokens, class_id);
        
        // 3. Sample from logits (only at masked positions)
        for (int i = 0; i < S; i++) {
            if (!is_masked[i]) continue;
            tokens[i] = sample_with_gumbel(logits, i, &rng, temperature);
        }
        
        // 4. Compute confidence scores
        // 5. Determine how many tokens to keep (cosine schedule)
        int n_mask = cosine_schedule(t, n_steps, S);
        
        // 6. Keep top-(S - n_mask) most confident, re-mask the rest
        // ... (sort by confidence, mask lowest)
    }
    
    // All tokens are now unmasked — run VQGAN decoder
    mg_tensor* image = vqgan_decode(model, tokens);
    save_image(image, "output.png");
}
```

**Key property:** The transformer compute graph is built once and executed `n_steps` times. Between iterations, only the input token buffer and mask are updated — the graph structure and weight tensors are reused with zero-copy.

### 2.5 Model File Format (GGUF)

Following the GGUF standard used by llama.cpp and stable-diffusion.cpp:

```
maskgit-256-q4_k.gguf
├── Header: magic "GGUF", version 3
├── Metadata (key-value):
│   ├── general.architecture: "maskgit"
│   ├── general.name: "MaskGIT ImageNet 256x256"
│   ├── maskgit.resolution: 256
│   ├── maskgit.n_layer: 24
│   ├── maskgit.n_head: 8
│   ├── maskgit.n_embd: 768
│   ├── maskgit.n_ffn: 3072
│   ├── maskgit.n_codebook: 1024
│   ├── maskgit.codebook_dim: 256
│   ├── maskgit.n_tokens: 256
│   ├── maskgit.vqgan.n_channels: [512, 512, 256, 128, 128]
│   ├── maskgit.vqgan.n_res_blocks: 2
│   └── general.file_type: Q4_K
├── Tensor info: name, shape, type, offset for each tensor
└── Tensor data: quantized weights, mmap-aligned
```

**Tensor naming convention** (standardized for loader):
- Transformer: `blk.{i}.attn_q.weight`, `blk.{i}.attn_k.weight`, `blk.{i}.attn_v.weight`, `blk.{i}.attn_o.weight`, `blk.{i}.ffn_up.weight`, `blk.{i}.ffn_down.weight`, `blk.{i}.attn_norm.weight`, `blk.{i}.ffn_norm.weight`
- Embeddings: `token_embd.weight`, `pos_embd.weight`, `class_embd.weight`, `output_norm.weight`, `output.weight`
- VQGAN: `vqgan.codebook.weight`, `vqgan.post_quant_conv.weight`, `vqgan.decoder.blk.{i}.conv1.weight`, `vqgan.decoder.up.{i}.conv.weight`, etc.

---

## 3. Backend Architecture

### 3.1 Backend Interface

Following the ggml pluggable backend pattern:

```c
struct mg_backend_interface {
    const char* (*name)(void);
    
    // Buffer management
    mg_buffer* (*alloc_buffer)(size_t size);
    void       (*free_buffer)(mg_buffer* buf);
    
    // Tensor operations
    void (*compute_graph)(mg_backend* be, mg_cgraph* graph);
    
    // Op-level support query
    bool (*supports_op)(mg_backend* be, const mg_tensor* op);
};
```

### 3.2 CPU Backend (ARM)

The primary backend, targeting Android ARM64 processors:

**SIMD dispatch (runtime detection, following llama.cpp's pattern):**
- Baseline: scalar C (always works)
- NEON: 128-bit SIMD, universal on arm64-v8a (Android 28+)
- dotprod: `UDOT`/`SDOT` instructions for int8 dot products (ARMv8.2-A+, most 2019+ SoCs)
- i8mm: int8 matrix multiply (ARMv8.6-A+, Cortex-A520/A720+, recent flagship SoCs)
- SVE/SVE2: scalable vector extension (ARMv9+, Cortex-X3/A715+, high-end 2023+ SoCs)

Runtime detection via `getauxval(AT_HWCAP)` / `getauxval(AT_HWCAP2)` on Android, selecting the best available ISA variant at startup.

**Critical kernel implementations:**
- `mg_compute_mul_mat` — Quantized matmul (Q4_K × F16 → F32). The hottest kernel. Block-based dequantization fused into the NEON dot product inner loop. For dotprod-capable SoCs, the Q8_0 kernel uses `SDOT` for 4× throughput vs scalar. Reference: llama.cpp's `ggml-cpu/` ARM kernels and KleidiAI micro-kernels.
- `mg_compute_conv_2d` — im2col transform + `mg_compute_mul_mat`. The im2col produces NHWC-layout columns that feed directly into the optimized matmul.
- `mg_compute_soft_max` — NEON vectorized with `vfmaq_f32` for online max/sum.
- `mg_compute_layer_norm` — Two-pass (mean, variance) with NEON reduction + fused affine.
- `mg_compute_gelu` — Tanh approximation with NEON `vrsqrteq_f32`.
- `mg_compute_group_norm` — Per-group mean/variance for VQGAN ResBlocks.

**Thread pool:** pthread-based work-stealing pool. Thread count defaults to `sysconf(_SC_NPROCESSORS_ONLN) / 2` (use big cores only on big.LITTLE). Matmul and conv parallelized across rows. Thread affinity pinning to big cores via `sched_setaffinity` for consistent performance.

**Android-specific considerations:**
- No OpenMP (`-DGGML_OPENMP=OFF`) — Android NDK ships it but CMake dependency management is unreliable. Use custom thread pool instead (same approach as llama.cpp on Android).
- `-march=armv8.4a+dotprod` as the default compile target for broad compatibility, with runtime dispatch to higher ISA levels.
- Stack size limits: Android's default thread stack is 1 MB — ensure scratch buffers use heap allocation, not large stack arrays.

### 3.3 Vulkan Backend (Primary Android GPU)

The primary GPU backend for Android, targeting Vulkan 1.1+ (guaranteed on Android 28+):

**Target GPUs:**
- Qualcomm Adreno 600/700/800 series (Snapdragon 8xx, 7xx, 6xx)
- ARM Mali-G710/G715/G720, Immortalis-G720 (Samsung Exynos, MediaTek Dimensity, Google Tensor)

**Implementation approach (referencing llama.cpp and stable-diffusion.cpp Vulkan backends):**
- GLSL compute shaders compiled to SPIR-V at build time (via `glslc` from NDK shader-tools)
- Pre-compiled SPIR-V shader binaries embedded in the library — no runtime shader compilation
- Shader variants for different workgroup sizes tuned per GPU family (Adreno prefers 64, Mali prefers 64–128)
- Custom compute shaders for: matmul (tiled, quantized), softmax, layer_norm, GELU, group_norm, im2col, conv2d, silu, upscale
- `VK_KHR_16bit_storage` and `VK_KHR_shader_float16_int8` for FP16 compute where supported

**Vulkan-specific design:**
- Descriptor sets for weight buffers (persistent) and activation buffers (per-invocation)
- Push constants for per-dispatch parameters (matrix dimensions, strides)
- Pipeline cache for shader pipeline objects (warm up on first run, persist to app cache dir)
- Memory allocation via a custom sub-allocator over `VkDeviceMemory` — avoid per-tensor `vkAllocateMemory` overhead
- Double-buffered command submission: prepare iteration N+1's command buffer while GPU executes iteration N

**Known driver issues and mitigations:**
- Adreno FP16 precision: some Adreno drivers produce NaN in certain FP16 reduction patterns. VQGAN decoder may need FP32 fallback (same issue stable-diffusion.cpp encountered with SDXL VAE)
- Mali warp divergence: Mali GPUs have warp size 16 (vs 32 on Adreno), requiring different reduction tree implementations
- Driver timeout: Android enforces a GPU watchdog timer (~2s default). Long kernel launches must be split into multiple dispatches
- Memory limits: mobile GPUs share system RAM, no dedicated VRAM. Peak allocation must stay within Android's per-process GPU memory budget

**Host-GPU synchronization:**
- The iterative decoding loop runs on CPU. Each iteration submits the transformer forward pass to GPU, waits for completion, reads back logits, performs masking/sampling on CPU, then submits the next iteration
- For GPU-only execution: optional path that keeps tokens on GPU between iterations via Vulkan buffer reuse, avoiding readback until the final VQGAN decode

### 3.4 OpenCL Backend (Alternative Android GPU)

Secondary GPU backend for broader Android device compatibility:

- OpenCL 2.0/3.0 compute kernels for Adreno and Mali GPUs
- Relevant for older devices where Vulkan drivers are unstable but OpenCL is mature
- Kernel source compiled at runtime via `clBuildProgram` (with binary caching to app storage)
- Same architectural pattern as Vulkan backend: persistent weight buffers, per-iteration dispatch
- Reference: MLC-LLM and TFLite GPU delegate both use OpenCL as a primary Android GPU backend

### 3.5 Host Development Backend (x86 CPU)

For development and verification on the host machine (not deployed to Android):
- x86 CPU backend with SSE4.1/AVX2 SIMD
- Used only for: running the numerical verification suite (Milestone 3), debugging, and rapid iteration during development
- Same kernel interfaces as the ARM CPU backend, different SIMD intrinsics
- This is NOT a deployment target — it exists to make development faster without needing an Android device for every test

---

## 4. Milestone Plan

### Legend

- 🔲 Not started
- 🔄 In progress
- ✅ Complete
- 🚧 Blocked / waiting for validation

---

### Milestone 1 — Stable Reference Implementation ✅ (implemented, awaiting sign-off)

**Goal:** A clean, reproducible pipeline that takes an ImageNet class ID and produces a generated image, using official pretrained checkpoints. This becomes the ground-truth oracle for all subsequent work.

> **Implemented as PyTorch, not JAX.** Google's `gs://maskgit-public` checkpoint bucket is decommissioned (all objects return `AccessDenied`); the original flax/msgpack checkpoints are unobtainable. The genuine official weights survive only as a faithful PyTorch conversion ([hmorimitsu/maskgit-torch](https://github.com/hmorimitsu/maskgit-torch)). The model math is unchanged. See [`reference/README.md`](reference/README.md). Also note the actual config differs from the prose below: **16 attention heads** (not 8), post-norm LayerNorm, vocab 2025, choice temperature 4.5.

**Scope:**

1. **Environment setup**
   - Python 3.10+, JAX (CPU backend sufficient), Flax, TensorFlow (checkpoint loading only)
   - Pin exact dependency versions in `requirements.txt`
   - Dockerfile for reproducibility

2. **Checkpoint loading and validation**
   - Download official checkpoints from GCS:
     - `tokenizer_imagenet256_checkpoint` (VQGAN, ~72M params)
     - `maskgit_imagenet256_checkpoint` (Transformer, ~174M params)
     - Optionally: 512×512 variants
   - Parse and print full parameter tree with shapes and dtypes
   - Validate total parameter counts match expected values

3. **Inference script** (`run_reference.py`)
   - Input: `--class-id` (0–999), `--steps` (default 8), `--seed`, `--temperature`
   - Output: generated PNG image
   - Implements complete iterative masked decoding:
     - Cosine masking schedule
     - Gumbel noise sampling for diversity
     - Confidence-based token selection
   - Single-image and batch generation modes

4. **Weight export for C++ ingestion** (`export_weights.py`)
   - Dumps all model weights to flat numpy `.npz` files with standardized naming
   - Tensor names follow the GGUF convention defined in Section 2.5
   - Includes metadata JSON: architecture hyperparameters, tensor shapes, dtypes
   - This is the input to the GGUF converter in Milestone 2

5. **Intermediate tensor dumping** (`--dump-intermediates` flag)
   - Saves per-layer activations to `.npz` at every point needed for Milestone 3:
     - Per transformer layer: post-norm input, Q/K/V projections, attention scores, attention output, post-attention residual, FFN intermediate (post-GELU), FFN output, post-FFN residual
     - Per VQGAN decoder stage: post-conv feature maps, post-resblock outputs, post-upsample outputs
     - Per decoding iteration: logits, sampled tokens, confidence scores, mask decisions
   - Controlled by `--dump-dir` and `--dump-layers` (specific layers or "all")

6. **Reproducibility validation**
   - Fixed seed → deterministic output (bit-identical across runs)
   - Visual quality check: generated images should match demo Colab outputs
   - Generate reference outputs for 10 fixed (class_id, seed) pairs → `reference_outputs/`

**Deliverables:**

| Artifact | Description |
|---|---|
| `reference/run_reference.py` | End-to-end inference script |
| `reference/export_weights.py` | Weight extractor with standardized naming |
| `reference/requirements.txt` | Pinned dependencies |
| `reference/Dockerfile` | Reproducible environment |
| `reference/reference_outputs/` | 10 reference images with metadata |
| `reference/tests/test_reproducibility.py` | Determinism + parameter count tests |

**Exit criteria:**
- `python run_reference.py --class-id 207 --seed 42 --steps 8` produces a clear, recognizable golden retriever
- `export_weights.py` produces a complete weight dump that round-trips (load back into JAX → identical inference)
- Intermediate dumps verified for 3 (class_id, seed) pairs
- **User validates generated image quality and approves before proceeding to Milestone 2**

---

### Milestone 2 — Custom C/C++ Runtime & Kernels 🔲

**Goal:** A from-scratch C/C++ inference engine that loads a GGUF model file and runs the full MaskGIT generation pipeline on CPU and GPU, producing images matching the JAX reference.

**Scope:**

#### Phase 2A: Tensor Library & CPU Kernels

1. **Core tensor library** (`src/mg-tensor.h`, `src/mg-tensor.c`)
   - `mg_tensor` struct: data pointer, shape (4D), strides, type, op, src pointers
   - `mg_context`: arena allocator for tensor objects
   - `mg_cgraph`: DAG with topological sort for execution ordering
   - Zero-copy view operations: reshape, permute, transpose, slice

2. **CPU kernel implementations** (`src/mg-cpu/`)
   - Start with F32 reference kernels for correctness, then optimize:
   - `mg_compute_mul_mat_f32` — naive triple-loop, then SIMD-optimized
   - `mg_compute_mul_mat_q4k_f32` — quantized matmul with fused dequantization
   - `mg_compute_soft_max` — numerically stable, SIMD-vectorized
   - `mg_compute_layer_norm` — two-pass with fused affine
   - `mg_compute_gelu` — tanh approximation
   - `mg_compute_conv_2d` — im2col + mul_mat
   - `mg_compute_conv_transpose_2d` — matmul + col2im
   - `mg_compute_group_norm` — per-group statistics
   - `mg_compute_silu`, `mg_compute_upscale`, `mg_compute_get_rows`
   - Thread pool with configurable thread count

3. **Unit tests per kernel** (`tests/test_kernels.c`)
   - Each kernel tested against numpy reference values
   - Edge cases: zero-size, non-contiguous inputs, various quantization types
   - Numerical tolerance tests (F32 vs F32: exact; quantized: bounded error)

#### Phase 2B: Model Loading & Graph Construction

4. **GGUF converter** (`tools/convert_to_gguf.py`)
   - Reads the `.npz` weight dump from Milestone 1
   - Applies quantization (F32 / F16 / Q8_0 / Q4_K) per tensor
   - Writes GGUF file with proper metadata, alignment, and tensor layout
   - Mixed-precision support: keep embeddings and norm weights in F32, quantize attention/FFN weights

5. **GGUF loader** (`src/mg-model.c`)
   - mmap-based loading (no copy, instant startup)
   - Parse metadata → populate `mg_model_params` struct
   - Map tensor data pointers directly into mmap'd region
   - Memory-lock option to prevent page faults during inference

6. **Transformer graph builder** (`src/mg-transformer.c`)
   - Builds `mg_cgraph` for one forward pass:
     - Token embedding + positional embedding + class embedding
     - 24× (LayerNorm → Self-Attention → Residual → LayerNorm → FFN → Residual)
     - Final LayerNorm → output projection to codebook logits
   - Graph is built once, executed per iteration with updated input buffer

7. **VQGAN decoder graph builder** (`src/mg-vqgan.c`)
   - Builds `mg_cgraph` for the decoder:
     - Codebook lookup (get_rows)
     - Post-quantization conv
     - ResBlock stages with GroupNorm + SiLU + Conv2d
     - Upsample stages (nearest-neighbor + Conv2d)
     - Final conv to RGB

#### Phase 2C: Iterative Decoding & CLI

8. **Decoding loop** (`src/mg-generate.c`)
   - Cosine masking schedule
   - Gumbel noise sampling with configurable temperature
   - Confidence-based top-k selection
   - Token buffer management between iterations (zero-copy)
   - Random seed for reproducibility

9. **CLI tool** (`tools/mg-generate`)
   - `./mg-generate -m maskgit-256-q4k.gguf --class-id 207 --steps 8 --seed 42 -o output.png`
   - Flags: `--threads N`, `--device cpu|vulkan|opencl`, `--precision f32|f16|q8|q4k`
   - Outputs: PNG image, optional timing breakdown
   - Built for both host (x86, for development) and Android (arm64, via NDK cross-compile)

#### Phase 2D: Android GPU Backends

10. **Vulkan backend** (`src/mg-vulkan/`)
    - GLSL compute shaders → SPIR-V (compiled at build time via `glslc`)
    - Tiled matmul shader with workgroup-size variants for Adreno/Mali
    - Custom shaders for: softmax, layernorm, GELU, group_norm, im2col, silu, upscale
    - FP16 compute path with FP32 fallback for numerically sensitive ops (VQGAN decoder norms)
    - VkPipelineCache persistence to Android app cache directory
    - Reference: llama.cpp Vulkan backend + stable-diffusion.cpp Vulkan Conv2D

11. **OpenCL backend** (`src/mg-opencl/`)
    - OpenCL C kernel sources embedded as strings
    - Runtime compilation with binary caching (`clGetProgramInfo` → app private storage)
    - Same kernel set as Vulkan, implemented as OpenCL kernels
    - Fallback for devices with unstable Vulkan drivers

#### Phase 2E: Android Build & Deployment

12. **NDK cross-compilation**
    - CMake toolchain file for Android NDK (r26+)
    - Target: `arm64-v8a`, Android platform 28+
    - Compile flags: `-march=armv8.4a+dotprod` (broad compatibility)
    - Static linking (single `.so` or standalone binary)
    - Vulkan headers from NDK or Vulkan-Headers repo

13. **On-device execution**
    - `adb push` workflow for standalone binary testing
    - JNI wrapper (`libmaskgit_jni.so`) for integration into Android apps
    - Model file loading from Android `assets/` or app private storage
    - Minimal JNI API: `maskgit_load_model()`, `maskgit_generate()`, `maskgit_free()`

**Deliverables:**

| Artifact | Description |
|---|---|
| `src/mg-tensor.{h,c}` | Tensor library core |
| `src/mg-cpu/` | ARM CPU kernels (F32, F16, Q8_0, Q4_K) with NEON/dotprod |
| `src/mg-vulkan/` | Vulkan backend + GLSL shaders |
| `src/mg-opencl/` | OpenCL backend |
| `src/mg-host-cpu/` | x86 host CPU backend (development only) |
| `src/mg-model.c` | GGUF loader |
| `src/mg-transformer.c` | Transformer graph builder |
| `src/mg-vqgan.c` | VQGAN decoder graph builder |
| `src/mg-generate.c` | Iterative decoding + CLI |
| `src/mg-jni.c` | JNI wrapper for Android apps |
| `tools/convert_to_gguf.py` | Weight converter |
| `tools/mg-generate` | CLI binary (host + Android) |
| `tests/` | Kernel unit tests |
| `CMakeLists.txt` | Build system with NDK cross-compile support |

**Exit criteria:**
- On **host x86**: `./mg-generate -m maskgit-256-f32.gguf --class-id 207 --seed 42 --steps 8` produces a visually correct golden retriever
- On **Android device** (via adb): same command produces matching output on ARM CPU
- At least one mobile GPU backend (Vulkan or OpenCL) functional on a reference device (e.g., Pixel 7/8 or Samsung S23/S24)
- Images are visually indistinguishable from JAX reference for the same class/seed

---

### Milestone 3 — Numerical Verification (Layer-by-Layer) ✅ (2026-06-04)

**Goal:** Automated, layer-by-layer numerical comparison between the PyTorch reference (M1) and the C++ runtime, verifying correctness at every intermediate computation.

> **Status: ✅ done (2026-06-04).** Per-layer numerical verification scaffold for **step 1 of the iterative-decoding loop** (the canonical class-token + all-MASK input already shared with `verify-opencl-transformer`). The transformer forward is deterministic given input tokens, so RNG control is not needed for the math verification — we sidestep Gumbel-noise alignment entirely by checking a single fixed forward. Loop-trajectory verification (teacher-forcing PyTorch's per-step inputs into the C++ side) is documented in `verification/README.md` as the next-step option if regression escapes the step-1 gate.
>
> **Dumper**: `reference/dump_intermediates.py` registers forward hooks on PyTorch's `emb_ln`, per-block `AttentionLN`/`MlpLN`, and the final LN — writes 51 tensors (embd_post_norm + 24·2 layer outputs + output_norm + output_logits) as raw float32 to `reference/export/intermediates_step1/` plus a `meta.json` with shapes. **C++ side**: `src/mg-transformer.cpp` calls `->named(...)` on the 51 matching tensors (free — naming is metadata only); `src/mg-opencl/mg-opencl.cpp` reads back named non-view tensors when `MG_DUMP_NAMED=1` (env-gated so production pays nothing); `tools/verify-opencl-transformer.cpp` adds `--dump-dir DIR` to write each named tensor as `<name>.bin`. **Compare**: `verification/compare_intermediates.py` walks both directories, computes per-tensor (max_abs_diff, mean_abs_diff, cosine), auto-detects the precision tier from the logit-level diff magnitude, and prints a per-layer table + PASS/FAIL at tiered tolerances (f32: cos ≥ 0.99999; q8: cos ≥ 0.9999; q4: cos ≥ 0.99 per-layer — the int4 weight precision budget drifts mid-network to ~0.995 before the output projection smooths the logits back to ≥0.9999).
>
> **Scorecard** (M=257, step 1, 51 tensors per backend):
>
> | run | worst cosine | worst max_abs | result |
> |---|---:|---:|---|
> | Host OpenCL F32 | 1.00000000 | 2.86e-5 | 51/51 PASS (bit-perfect, float-add reordering only) |
> | Host OpenCL Q8_0 | 0.99997516 | 6.87e-2 | 51/51 PASS |
> | Mali OpenCL Q8_0 (fp16 FA) | 0.99991820 | 1.99e-1 | 51/51 PASS |
> | Mali OpenCL Q4_K (int8-dot) | 0.99427133 | 1.38e0  | 51/51 PASS |
>
> See `verification/README.md` for methodology + the trajectory-replay extension plan.

**Original scope (kept below for reference):**

**Scope:**

1. **C++ intermediate dumping**
   - `--dump-intermediates` flag in `mg-generate`
   - Hooks into graph execution to capture tensor values after each operation
   - Dumps to `.npz` format (via a minimal C npz writer, or binary + Python reader)
   - Same tensor naming convention as the JAX dumper

2. **Verification harness** (`verification/verify.py`)
   - For a given (class_id, seed, steps) triple:
     - Load JAX intermediates from `reference/dumps/`
     - Load C++ intermediates from `build/dumps/`
     - Match tensors by name, compare layer-by-layer
   - Per-tensor metrics:
     - Max absolute difference
     - Mean absolute difference
     - Cosine similarity
     - Relative error distribution (histogram)
   - Aggregate pass/fail per tolerance tier

3. **Layer mapping and verification matrix**

   | Layer | Tensor Name | F32 Tolerance | Q8_0 Tolerance | Q4_K Tolerance |
   |---|---|---|---|---|
   | Token embedding | `blk.0.input` | max < 1e-5 | max < 1e-3 | max < 5e-2 |
   | Attn Q proj (layer 0) | `blk.0.attn_q.output` | max < 1e-4 | max < 1e-2 | max < 1e-1 |
   | Attn scores (layer 0) | `blk.0.attn_scores` | max < 1e-4 | max < 1e-2 | max < 1e-1 |
   | FFN output (layer 23) | `blk.23.ffn.output` | max < 5e-4 | max < 5e-2 | max < 2e-1 |
   | Final logits | `output_logits` | max < 1e-3 | max < 1e-1 | max < 5e-1 |
   | VQGAN stage 0 | `vqgan.stage0.output` | max < 1e-4 | max < 1e-2 | max < 1e-1 |
   | Final image pixels | `output_image` | max < 1e-2 | max < 5e-1 | max < 2.0 |

   Note: Tolerances widen at deeper layers due to error accumulation. Quantized tolerances are measured against the F32 C++ result, not the JAX result (isolating quantization error from implementation error).

4. **Regression test suite**
   - CI-runnable: `python verify.py --test-suite standard`
   - 5 fixed (class_id, seed) pairs
   - F32 must pass all layers; Q8_0 and Q4_K must pass with relaxed tolerances
   - Generates HTML report with per-layer comparison plots

5. **Error attribution**
   - When a layer fails tolerance, the report identifies:
     - Is the error in this layer's kernel, or accumulated from prior layers?
     - Which specific operation (matmul, norm, softmax) has the largest discrepancy?
     - Is the error systematic (bias) or random (numerical noise)?

**Deliverables:**

| Artifact | Description |
|---|---|
| `verification/verify.py` | Layer-by-layer comparison harness |
| `verification/tolerance_spec.yaml` | Per-layer tolerance thresholds |
| `verification/reports/` | HTML comparison reports |
| `verification/tests/test_numerical.py` | CI regression tests |

**Exit criteria:** All 24 transformer layers + all VQGAN decoder stages pass F32 tolerance for 5 test cases. Q8_0 and Q4_K reports generated and reviewed.

---

### Milestone 4 — Evaluation Framework ✅ (2026-06-02)

**Goal:** Standardized evaluation pipeline measuring generative quality of the C++ runtime against established benchmarks.

> **Status: ✅ done (2026-06-02).** `evaluation/eval_runner.py` runs a single InceptionV3 pass per backend folder and emits three metrics — **IS (mean ± std, 10 splits)**, **top-1 / top-5 ImageNet classifier accuracy** (target class encoded in filename `c{cls}_s{seed}.png`), and optional **paired PSNR** vs an oracle folder. The PyTorch oracle is wired via `--backend reference`, which pipes a job list into a new `reference/batch_reference.py` (model loads once for the whole batch — 40 images in 75 s on M1 MPS vs. the ~17 min the per-subprocess flow would take). All three runtimes scored on **Quick-5** (40 images): IS within ~0.1 of oracle and top-5 ≥ 95% for both **xnnpack-q8** (host C++ int8) and **opencl-gq8** (Pixel 9 Mali-G715 int8) — no detectable quality regression from int8 quantization. Results in `evaluation/results/quick-5/{reference,xnnpack,opencl}/`. **Deviation from original plan:** dropped clean-fid + ImageNet val FID — clean-fid's hosted stats mirror does not include any imagenet*.npz, and IS+accuracy+PSNR give us a closed-loop scorecard without downloading ~50k val images. Caveat documented in `evaluation/README.md`: PSNR-vs-PyTorch-oracle is dominated by RNG-path divergence (Mersenne Twister vs our C++ PRNG), not runtime error — PSNR's real use is F32-vs-quantized on the same backend.

**Scope:**

1. **Evaluation datasets**

   | Dataset | Size | Resolution | Purpose |
   |---|---|---|---|
   | ImageNet-256 (val) | 50k reference images | 256×256 | Full FID/IS benchmark |
   | Quick-20 | 20 classes × 50 images = 1k | 256×256 | Fast sanity check (~5 min) |
   | Single-class-1k | 1 class × 1k images | 256×256 | Per-class quality analysis |
   | ImageNet-512 (val) | 50k reference images | 512×512 | High-res benchmark |

2. **Metrics**
   - FID (Fréchet Inception Distance) — primary quality metric
   - IS (Inception Score) — diversity metric
   - Precision / Recall — mode coverage
   - Computed using `clean-fid` (Python, wraps the C++ runtime for generation)

3. **Evaluation runner** (`evaluation/eval_runner.py`)
   - Calls `mg-generate` binary in batch mode to produce images
   - Computes metrics against ImageNet validation set statistics
   - Supports runtime variants: jax-reference / cpp-f32 / cpp-q8 / cpp-q4k / cpp-gpu
   - Resumable: checkpoints partial generation progress
   - Parallelized: spawns multiple `mg-generate` processes

4. **Quality regression gates**

   | Runtime | vs. JAX Reference FID Delta | Threshold |
   |---|---|---|
   | C++ F32 (CPU) | | < 0.3 |
   | C++ F16 (GPU) | | < 0.5 |
   | C++ Q8_0 | | < 1.0 |
   | C++ Q4_K | | < 2.0 |

5. **Baseline documentation**
   - Paper-reported FID: 6.06 (256×256), 7.32 (512×512)
   - JAX reference measured FID (our reproduction)
   - All C++ variants measured FID
   - Analysis of any deviations

**Deliverables:**

| Artifact | Description |
|---|---|
| `evaluation/eval_runner.py` | Evaluation orchestrator |
| `evaluation/configs/` | Dataset configs |
| `evaluation/results/` | JSON metrics + sample grids |
| `evaluation/tests/test_quality.py` | Quality regression tests |

**Exit criteria:** C++ F32 runtime FID matches JAX reference within 0.3 on Quick-20 dataset (1k images). Full ImageNet results documented.

---

### Milestone 5 — Benchmark Tool (Latency & Performance) 🔲

**Goal:** Comprehensive benchmarking tool measuring latency, throughput, memory, and per-op profiling across devices and quantization levels.

**Scope:**

1. **Benchmark runner** (`benchmark/bench.cpp` or `tools/mg-bench`)
   - Built into the C++ binary as a benchmark mode:
   - `./mg-generate --bench -m model.gguf --steps 8 --n-runs 50 --warmup 5`
   - Measures and reports:
     - **End-to-end latency** (class ID → output image): p50, p90, p99, mean, std
     - **Per-component breakdown:**
       - Transformer forward pass (per iteration)
       - Masking / sampling logic (host-side, typically negligible)
       - VQGAN decoding
     - **Per-op profiling:** time spent in each kernel type (matmul, softmax, norm, conv2d, etc.)

2. **Configuration matrix**

   | Dimension | Values |
   |---|---|
   | Backend | ARM CPU, Vulkan, OpenCL |
   | Precision | F32, F16, Q8_0, Q4_K |
   | Resolution | 256×256, 512×512 |
   | Decoding steps | 4, 8, 12, 16 |
   | Threads (CPU) | 1, 2, 4, big-cores-only |
   | Batch size | 1 |
   | Target devices | See reference device list below |

   **Reference devices for benchmarking:**

   | Device | SoC | CPU | GPU | RAM |
   |---|---|---|---|---|
   | Samsung Galaxy S24 | Snapdragon 8 Gen 3 | Cortex-X4 + A720 + A520 | Adreno 750 | 8 GB |
   | Google Pixel 8 | Tensor G3 | Cortex-X3 + A715 + A510 | Mali-G715 | 8 GB |
   | Samsung Galaxy S23 | Snapdragon 8 Gen 2 | Cortex-X3 + A715 + A510 | Adreno 740 | 8 GB |
   | Google Pixel 7 | Tensor G2 | Cortex-X1 + A78 + A55 | Mali-G710 | 8 GB |
   | Mid-range (e.g., Pixel 7a) | Tensor G2 | same | Mali-G710 | 8 GB |

3. **Resource monitoring**
   - Peak RSS memory via `/proc/self/status` (VmHWM)
   - GPU memory via Vulkan `vkGetPhysicalDeviceMemoryProperties` + allocation tracking
   - Model load time (cold start, mmap vs eager)
   - Per-iteration memory (scratch buffer size)
   - Battery/thermal: monitor `BatteryManager` temperature to detect thermal throttling during sustained generation

4. **Profiling**
   - Built-in timer instrumentation around each graph node
   - Output as JSON timeline (Chrome trace format — viewable in `chrome://tracing` or Perfetto)
   - Identifies kernel-level bottlenecks:
     - Which matmul shapes are slowest?
     - Is im2col memory-bound or compute-bound?
     - Attention vs FFN time split
   - On Adreno: Snapdragon Profiler integration (optional) for GPU occupancy and memory bandwidth
   - On Mali: ARM Streamline integration (optional) for shader core utilization

5. **Performance targets** (indicative, hardware-dependent)

   | Config | Target (256×256, 8 steps) | Model Size |
   |---|---|---|
   | Snapdragon 8 Gen 3 CPU (big cores, Q4_K) | < 5 seconds | ~140 MB |
   | Snapdragon 8 Gen 3 CPU (big cores, Q8_0) | < 8 seconds | ~262 MB |
   | Snapdragon 8 Gen 3 Adreno 750 (Vulkan, F16) | < 2 seconds | ~500 MB |
   | Tensor G3 CPU (big cores, Q4_K) | < 6 seconds | ~140 MB |
   | Tensor G3 Mali-G715 (Vulkan, F16) | < 3 seconds | ~500 MB |
   | Snapdragon 8 Gen 2 CPU (big cores, Q4_K) | < 7 seconds | ~140 MB |
   | Snapdragon 8 Gen 2 Adreno 740 (Vulkan, F16) | < 3 seconds | ~500 MB |
   | Host x86 CPU (dev machine, 8 threads, F32) | < 5 seconds | ~985 MB |

6. **Comparison output**
   - Auto-generated Markdown table comparing all configurations
   - Speedup ratios: Q4_K vs Q8_0, Vulkan vs CPU, big-cores vs all-cores
   - CPU vs GPU crossover analysis: at what model/batch size does GPU become faster?
   - Roofline analysis: memory-bandwidth-bound vs compute-bound per kernel

**Deliverables:**

| Artifact | Description |
|---|---|
| `tools/mg-bench` | Benchmark binary (or mode in mg-generate) |
| `benchmark/configs/` | Hardware profile configs |
| `benchmark/results/` | Raw JSON + Markdown summary |
| `benchmark/profile/` | Chrome trace files |
| `benchmark/analysis.md` | Written analysis with roofline discussion |

**Exit criteria:** Benchmark results collected for CPU-F32, CPU-Q4K, and at least one GPU backend on a reference machine. Per-op profiling identifies top-3 bottleneck operations. Results documented in summary table.

> **Status: ✅ done (2026-05-25).** `mg-generate --bench --n-runs N --warmup K` reports end-to-end latency percentiles (p50/p90/p99/mean±sd/min), a per-component breakdown (transformer / sampling / VQGAN), peak RSS, model-load time, and — for OpenCL — a per-op-type GPU profile (`MulMat` split into quantized-FC vs F32). Deliverables in `benchmark/` (run_bench.sh, results/{host,device}.md, analysis.md). The profiler immediately corrected a mis-ranked bottleneck (the F32 attention path was assumed dominant; it is only ~5% on Mali — see analysis.md).

---

### Milestone 6 — Performance Hill-Climbing ✅ (2026-06-04)

**Goal:** Iteratively drive down on-device latency, **guided by the M5 profiler** — measure, optimize the top-ranked op, re-profile, repeat. Each step is data-driven: never optimize an op without a profile showing it dominates, and always re-profile after a win because the bottleneck moves.

**Method (the loop):**
1. Profile the current build (`--bench`) on the target device → ranked per-op breakdown.
2. Optimize the #1 op (kernel rewrite / tiling / precision / tuning).
3. Validate correctness against the PyTorch oracle (cosine; the verify-opencl tools).
4. Re-benchmark host + device; record the delta. Re-profile → new #1.

**Baseline (Pixel 9 / Mali-G715, gq8, from M5):** end-to-end ~26 s; per-op `MulMat(q)` 44%, **`Conv2D` 30%**, GroupNorm 6%, Add 5%, `MulMat(f32)` attention 5%.

**Backlog (ranked by the profiler, highest first):**
- ✅ **Tiled VQGAN `Conv2D`** (#1): implicit-GEMM tiling, im2col gathered on-the-fly. Conv2D 30→18%, device gq8 26→21.5 s.
- ✅ **Mali tile-autotune for the quantized FC** (#2): swept 7 micro-tile configs — 4×4 already optimal (no win). Tunable via `-DGEMM_WPTM/N`.
- ✅ **Matmul-epilogue fusion** (#3): bias + GELU/SiLU + residual fused into the matmul (`mul_mat_ex`); correct, ~wash (bottleneck is matmul compute, not launches).
- ✅ **int8-dot matmul** (#4): `arm_dot_acc` for Q8_0 (quantize activation to int8). **device gq8 22→13 s** (2.5× transformer over the throttling-bound loop).
- ✅ **int8 VQGAN conv** (#5): `k_conv2d_i8` (arm_dot, per-(pixel,32-block) gather-time activation quant → cosine 0.99997, accuracy-safe). On by default. Conv2D 4.4→1.25 s, VQGAN 6.2→3.1 s, end-to-end gq8 12.8→9.7 s.
- ✅ **Workgroup-parallel reductions** (#6): one workgroup per row with local-memory tree reduction. GroupNorm 9.1×, Norm 4.2×, SoftMax 2.7×, end-to-end 9.79→7.08 s.
- ❌ **fp16 / TSK probes on F32 attention** (#7): NEGATIVE — attention overhead-bound at M=257, fusion is the only lever. *Documented as such.*
- ✅ **Flash-attention v2 + fp16 K/V tiles + strided-input** (#8a-c): single fused kernel; online softmax, M² scores never in DRAM. End-to-end 7.1 → 6.5 s. **Crossover with CPU shows up at M=1025; GPU 2.86× faster at M=4097.**
- ✅ **LN-affine fusion** (#8d): `k_norm_affine` replaces the Norm+Mul+Add chain in the transformer LayerNorm. 100 ops collapsed; end-to-end 6.5 → 6.5 s (M=257), M=1025 −9%.
- ❌ **`cl_arm_import_memory` zero-copy weight upload** (#8e): NEGATIVE — Mali driver rejects file-backed mmap (e=-6, accepts only CL_MEM_ALLOC_HOST_PTR or dma_buf). Gated OFF by default for future drivers.
- ✅ **VQGAN GroupNorm + affine + SiLU fusion** (#8f): 4 ops → 1 fused kernel. VQGAN decode 1671 → 1366 ms (−18%); end-to-end **6.466 → 6.119 s (−5.4%)**.
- ✅ **Q4_K int8-dot matmul** (#8g): generalized the M6 #4 path to Q4_K (extract 4-bit nibbles to int8, cache per-sub-block (scale, min)). Mali gq4 **16 s → 6.7 s (−58%)**, Q4_K now ≈ Q8_0 latency at 73% the file size.

**Final state (Pixel 9 Tensor G4 / Mali-G715, end-to-end Q8_0):** **111 s → 6.1 s = 18.2× speedup**, cosine ≥ 0.99997 (int8 quant noise budget); host M1 cosine bit-perfect (1.00000000) through every fusion. Mali Q4_K 16 → 6.7 s same path.

**Seq-len cross-domain finding:** the original "for this chip the CPU is the right tool" claim holds at M=257 (CPU 1.5× ahead). At M=1025 GPU pulls even; at M=4097 GPU is **2.86× faster (Q8) / 4.14× faster (Q4)** than the CPU. The crossover sits inside the prefill regime real workloads will hit (longer-context LLMs, larger-resolution image models).

See **`docs/DEEP_DIVE.md` §13.3 (step-by-step), §13.6 (flash-attention algorithm), §13.7 (closeout + lessons)** and **`benchmark/seqlen-sweep/`** for the full sweep data.

---

## 5. Project Structure

```
maskgit.cpp/
├── CMakeLists.txt
├── README.md
├── docs/
│   └── design.md                         ← this document
│
├── reference/                            ← Milestone 1
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── run_reference.py
│   ├── export_weights.py
│   ├── reference_outputs/
│   └── tests/
│
├── include/                              ← Public headers
│   ├── mg-tensor.h                       ← Tensor library API
│   ├── mg-backend.h                      ← Backend interface
│   └── mg-model.h                        ← Model loading API
│
├── src/                                  ← Milestone 2
│   ├── mg-tensor.c                       ← Core tensor ops, graph, allocator
│   ├── mg-model.c                        ← GGUF loader
│   ├── mg-transformer.c                  ← Transformer graph builder
│   ├── mg-vqgan.c                        ← VQGAN decoder graph builder
│   ├── mg-generate.c                     ← Iterative decoding + main CLI
│   ├── mg-cpu/                           ← ARM CPU backend (+ x86 host for dev)
│   │   ├── mg-cpu.c                      ← Backend registration + dispatch
│   │   ├── mg-cpu-ops.c                  ← Kernel implementations
│   │   ├── mg-cpu-quant.c                ← Quantization/dequantization
│   │   ├── mg-cpu-neon.h                 ← ARM NEON intrinsics
│   │   └── mg-cpu-x86.h                  ← x86 AVX2 intrinsics (host dev only)
│   ├── mg-vulkan/                        ← Vulkan backend (Android GPU)
│   │   ├── mg-vulkan.c                   ← Backend setup, command buffer mgmt
│   │   ├── mg-vulkan-pipeline.c          ← Pipeline/descriptor management
│   │   └── shaders/                      ← GLSL compute shaders → SPIR-V
│   │       ├── mul_mat.comp
│   │       ├── mul_mat_q4k.comp
│   │       ├── softmax.comp
│   │       ├── layer_norm.comp
│   │       ├── gelu.comp
│   │       ├── group_norm.comp
│   │       ├── im2col.comp
│   │       └── ...
│   ├── mg-opencl/                        ← OpenCL backend (Android GPU fallback)
│   │   ├── mg-opencl.c
│   │   └── kernels/                      ← OpenCL C kernel sources
│   └── mg-jni.c                          ← JNI bindings for Android apps
│
├── tools/                                ← Milestone 2
│   ├── convert_to_gguf.py                ← Weight conversion
│   └── imagenet_classes.json             ← Class ID → name mapping
│
├── verification/                         ← Milestone 3
│   ├── verify.py
│   ├── tolerance_spec.yaml
│   ├── reports/
│   └── tests/
│
├── evaluation/                           ← Milestone 4
│   ├── eval_runner.py
│   ├── configs/
│   ├── results/
│   └── tests/
│
├── benchmark/                            ← Milestone 5
│   ├── configs/
│   ├── results/
│   └── analysis.md
│
└── tests/                                ← Shared
    ├── test_kernels.c
    ├── test_model_load.c
    └── fixtures/
```

---

## 6. Build System

```cmake
# CMakeLists.txt (simplified)
cmake_minimum_required(VERSION 3.16)
project(maskgit_cpp C CXX)

# Detect if we're cross-compiling for Android
if(ANDROID)
    message(STATUS "Building for Android: ABI=${ANDROID_ABI} API=${ANDROID_PLATFORM}")
    # ARM CPU backend with NEON (always available on arm64-v8a)
    add_library(mg-cpu src/mg-cpu/mg-cpu.c src/mg-cpu/mg-cpu-ops.c src/mg-cpu/mg-cpu-quant.c)
    # dotprod is our baseline for Android — available on most 2019+ SoCs
    target_compile_options(mg-cpu PRIVATE -march=armv8.4a+dotprod)
else()
    # Host x86 backend for development
    add_library(mg-cpu src/mg-cpu/mg-cpu.c src/mg-cpu/mg-cpu-ops.c src/mg-cpu/mg-cpu-quant.c)
    target_compile_options(mg-cpu PRIVATE -mavx2 -mfma)
endif()

# GPU backends (optional, primarily for Android)
option(MG_VULKAN "Build Vulkan backend" OFF)
option(MG_OPENCL "Build OpenCL backend" OFF)

if(MG_VULKAN)
    if(ANDROID)
        # Android NDK provides Vulkan headers; link against libvulkan.so at runtime
        add_library(mg-vulkan src/mg-vulkan/mg-vulkan.c src/mg-vulkan/mg-vulkan-pipeline.c)
        target_link_libraries(mg-vulkan vulkan)
        # Compile GLSL shaders to SPIR-V at build time
        # glslc is in $ANDROID_NDK/shader-tools/<host-platform>/
        file(GLOB SHADER_SOURCES src/mg-vulkan/shaders/*.comp)
        foreach(SHADER ${SHADER_SOURCES})
            get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
            add_custom_command(
                OUTPUT ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv
                COMMAND glslc -fshader-stage=compute ${SHADER}
                        -o ${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv
                DEPENDS ${SHADER})
        endforeach()
    else()
        find_package(Vulkan REQUIRED)
        add_library(mg-vulkan src/mg-vulkan/mg-vulkan.c src/mg-vulkan/mg-vulkan-pipeline.c)
        target_link_libraries(mg-vulkan Vulkan::Vulkan)
    endif()
endif()

if(MG_OPENCL)
    add_library(mg-opencl src/mg-opencl/mg-opencl.c)
    if(ANDROID)
        target_link_libraries(mg-opencl OpenCL)  # dlopen at runtime
    else()
        find_package(OpenCL REQUIRED)
        target_link_libraries(mg-opencl OpenCL::OpenCL)
    endif()
endif()

# Core library
add_library(mg-core
    src/mg-tensor.c src/mg-model.c
    src/mg-transformer.c src/mg-vqgan.c src/mg-generate.c
)
target_link_libraries(mg-core mg-cpu)

# CLI binary (for adb push testing on Android, or host development)
add_executable(mg-generate tools/main.c)
target_link_libraries(mg-generate mg-core)

# JNI shared library (Android app integration)
if(ANDROID)
    add_library(maskgit_jni SHARED src/mg-jni.c)
    target_link_libraries(maskgit_jni mg-core log android)
endif()
```

**Build commands:**

```bash
# Host build (development, x86)
cmake -B build-host && cmake --build build-host -j$(nproc)

# Android build (cross-compile)
cmake \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_C_FLAGS="-march=armv8.4a+dotprod" \
  -DCMAKE_CXX_FLAGS="-march=armv8.4a+dotprod" \
  -DGGML_OPENMP=OFF \
  -DMG_VULKAN=ON \
  -B build-android
cmake --build build-android -j$(nproc)

# Deploy and test on device
adb push build-android/mg-generate /data/local/tmp/
adb push models/maskgit-256-q4k.gguf /data/local/tmp/
adb shell "cd /data/local/tmp && ./mg-generate -m maskgit-256-q4k.gguf --class-id 207 --steps 8 -o output.png"
adb pull /data/local/tmp/output.png .
```

---

## 7. Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| ggml-style tensor library, not linking ggml directly | MaskGIT's op set is smaller and more specialized than llama.cpp's. A purpose-built library avoids carrying unused complexity while allowing MaskGIT-specific optimizations (e.g., bidirectional attention without KV-cache, Conv2d with known small spatial dims). Can always integrate upstream ggml later if warranted. |
| GGUF file format for model weights | Proven format with excellent mmap support, standardized quantization types, and single-file self-containment. Compatible with the broader llama.cpp ecosystem tooling. |
| Host-side iterative decoding loop | The masking/sampling logic is lightweight, sequential, and involves dynamic control flow (variable mask counts per iteration). Keeping it on the host avoids graph compilation overhead and allows zero-copy tensor reuse between iterations. |
| im2col + matmul for Conv2d | Reuses optimized matmul kernels (the same ones used for transformer attention/FFN). The spatial dimensions are small (16×16 to 256×256), so im2col memory overhead is bounded. This is the same strategy used by stable-diffusion.cpp. |
| F32 reference kernels first, then SIMD/quantized | Correctness before performance. The F32 kernels serve as the ground truth for the numerical verification in Milestone 3. Optimized kernels are validated against them. |
| No KV-cache | MaskGIT uses bidirectional attention — every token attends to every other token at every iteration. There is no causal masking and no opportunity for KV-cache reuse across positions. Each iteration is a fresh, full attention computation over S tokens. |
| CNN ops as first-class citizens | Unlike llama.cpp (which rarely needs Conv2d), MaskGIT's VQGAN decoder is a full CNN. The Conv2d and ConvTranspose2d kernels are on the critical path and must be well-optimized. We reference stable-diffusion.cpp's approach here. |

---

## 8. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| JAX checkpoint format is non-trivial to parse | Blocks M1 | Reference the demo Colab which successfully loads these; community PyTorch conversions exist as fallback |
| Conv2d / ConvTranspose2d kernel performance on ARM CPU | Degrades M5 | im2col + matmul leverages optimized NEON matmul; reference ncnn's conv2d ARM optimizations |
| Quantization (Q4_K) causes unacceptable image quality loss | Degrades M4 | MaskGIT's ~250M params are relatively small — Q8_0 (262 MB) still fits in mobile RAM; Q4_K is optional |
| Vulkan driver instability on specific Android devices | Blocks M2D | OpenCL fallback backend; CPU-only mode always available; test on 3+ reference devices |
| Adreno FP16 produces NaN in reduction ops (known issue) | Breaks GPU path | FP32 fallback for VQGAN decoder norms; same workaround as stable-diffusion.cpp uses |
| Mali GPU slower than CPU for certain workloads | Degrades M5 | Profile early; GPU path may only be beneficial for transformer matmuls, not conv2d; hybrid CPU+GPU scheduling |
| Android GPU watchdog kills long-running shaders (~2s timeout) | Crashes GPU path | Split large dispatches into multiple smaller ones; each dispatch must complete within watchdog window |
| Android per-app memory limits (typically 256–512 MB heap) | Blocks deployment | Q4_K (140 MB model) + streaming scratch buffers; mmap avoids loading entire model into heap |
| NDK cross-compilation breaks on specific NDK versions | Blocks M2 | Pin NDK r26; test CI on multiple NDK versions; reference llama.cpp's proven Android build setup |
| Error accumulation through 24 transformer layers in quantized mode | Blocks M3 | Mixed precision: keep LayerNorm and embeddings in F32, quantize only attention/FFN weights |
| big.LITTLE CPU scheduling: OS may schedule on little cores | Degrades M5 | Thread affinity pinning to big cores via `sched_setaffinity`; runtime core topology detection |
| MaskGIT is class-conditional only (no free-text prompts) | Limits usability | Document clearly; Muse-style text conditioning is a future extension |
| GGUF ecosystem assumes LLM tensor naming conventions | Confuses tooling | Define a maskgit-specific naming convention in metadata; document thoroughly |

---

## 9. Milestone Progression Protocol

1. **Implementation** — build the deliverables for the current milestone.
2. **Self-test** — run all automated tests, verify exit criteria are met.
3. **Notification** — notify the user with:
   - Sample outputs (images, metrics, benchmark tables)
   - Test results summary (pass/fail counts, any known issues)
   - Clear statement of what was delivered vs. what was planned
4. **User validation** — user reviews, tests independently, provides feedback.
5. **Sign-off** — user confirms the milestone is complete.
6. **Proceed** — begin next milestone only after sign-off.

**Current status: Milestone 1 implemented (PyTorch oracle) — awaiting user validation & sign-off before Milestone 2.**

---

## 10. Important Note on "Prompts"

MaskGIT is a **class-conditional** image generation model. It takes an **ImageNet class ID** (integer 0–999) as input, not free-text prompts. Examples: 207 = golden retriever, 933 = cheeseburger, 88 = macaw, 972 = cliff. For text-to-image generation, Google extended MaskGIT into [Muse](https://arxiv.org/abs/2301.00704) with a text encoder — that extension is out of scope but could be added as a future milestone.
