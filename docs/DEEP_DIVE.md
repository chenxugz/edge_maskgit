# MaskGIT On-Device C++ Runtime — Technical Deep Dive

This document explains how the MaskGIT class-conditional image generator is
implemented as a from-scratch, ggml-inspired C++17 inference runtime. It covers
the tensor library and graph, the correctness-first reference CPU backend, the
accelerated XNNPACK backend, int8/int4 quantization, the verification
methodology, and the measured performance / memory results (host macOS arm64 +
Pixel 9). It is written so an engineer can understand the implementation from
this document plus the cited source.

All file references are `path:line`. Code snippets are quoted verbatim and kept
short; the load-bearing detail is the conventions and the *why*, not the bulk of
the code.

---

## 1. Overview & pipeline

MaskGIT turns an **ImageNet class id (0–999)** into a 256×256 RGB image in three
host-orchestrated phases:

1. **Iterative masked decoding** — a bidirectional transformer predicts a 16×16
   grid of discrete VQGAN codebook tokens over a handful of steps (default 8),
   refining the grid from "all masked" to fully filled in via a cosine schedule
   and confidence-based re-masking.
2. **VQGAN decode** — the final token grid is gathered into codebook embeddings
   and run through a convolutional decoder (residual blocks + 4× nearest 2×
   upsamples = 16× spatial) to produce a 256×256×3 image.
3. **PNG write** — clip to `[0,1]`, scale to bytes, write with `stb_image_write`.

The pipeline lives in `src/mg-generate.cpp` (`generate()`) and the CLI in
`tools/main.cpp`. The high-level flow is mirrored from the upstream JAX/PyTorch
reference (`reference/maskgit/libml/parallel_decode.py`), documented at
`src/mg-generate.cpp:1`.

### Two backends, and why both exist

The runtime ships **two interchangeable backends**, selected by the decoding
loop via optional callbacks (`mg::TransformerFwd` / `mg::VqganFwd`, see
`include/mg-generate.hpp:15`):

- **Reference CPU backend** (`src/mg-cpu/mg-cpu.cpp`) — scalar, stride-aware F32
  kernels driving a graph the runtime builds itself. This is the **correctness
  oracle**: it is written for clarity and exact numerical agreement with
  PyTorch, not speed. It is the default `--backend reference`.
- **XNNPACK backend** (`src/mg-xnn.cpp`, `src/mg-xnn-vqgan.cpp`) — the transformer
  and VQGAN are each expressed as a full XNNPACK subgraph, built once and invoked
  per decode step, with int8/int4 quantization support. Selected with
  `--backend xnnpack`.

Both exist because they validate each other: the reference path defines "correct"
to ~1e-5 against the PyTorch oracle, and the XNNPACK path is verified against the
*same* fixed-input oracle. Reference vs XNNPACK F32 is bit-identical for the
transformer (README.md:51). The reference path is also where new ops are
prototyped before being mapped onto XNNPACK primitives. The cost is latency: the
scalar reference takes ~731 s vs XNNPACK's ~14 s for the same F32 run
(README.md:36-37).

The decode loop itself (`generate()`) is backend-agnostic. When a callback is
supplied it is used; otherwise the loop builds and runs the reference graph
inline (`src/mg-generate.cpp:64-76`).

---

## 2. Tensor library & graph

The tensor library (`include/mg-tensor.hpp`, `src/mg-tensor.cpp`) is a deliberately
small ggml clone. The conventions are stated once at the top of the header and
relied on everywhere.

### Layout conventions

- **Row-major, `ne[0]` is the innermost (contiguous) dimension**, like ggml. A
  PyTorch tensor `[d0, d1, …, dn]` maps to `ne[]` *reversed*: e.g. activations
  `[batch, seq, hidden]` → `ne = {hidden, seq, batch, 1}` (`mg-tensor.hpp:4-8`).
- Up to `MAX_DIMS = 4`; unused trailing dims are 1.
- **`nb[]` are byte strides.** Contiguous strides are `nb[0] = type_size`,
  `nb[i] = nb[i-1] * ne[i-1]` (`mg-tensor.cpp:44-47`). Views may carry arbitrary
  strides, which is what makes permute/transpose zero-copy.

`is_contiguous()` walks the dims confirming each stride equals the running
product (`mg-tensor.cpp:34-42`).

### `mul_mat` semantics

The most important convention is the matmul:

```cpp
//  - mul_mat(w, x): w is [K, N] (ne={K,N}), x is [K, M] (ne={K,M}) -> [N, M]
//    (ne={N,M}); result[n,m] = sum_k w[k,n]*x[k,m]. This matches a PyTorch
//    Linear weight stored [out=N, in=K]: pass weight as `w`, input as `x`.
```
(`mg-tensor.hpp:10-12`)

Both operands share the **contraction dimension `ne[0]`**. A PyTorch
`nn.Linear` weight (logically `[out, in]`, stored row-major) lands as `ne={in,
out}` and is passed directly as `w`; the activation `[…, in]` is `x`. The builder
also propagates batch dims `ne[2]`/`ne[3]` from `x` and allows `w` to broadcast
over them (`mg-tensor.cpp:103-108`), which is exactly what batched attention
needs.

### Context arena allocator

`Context` owns every `Tensor` object (stable pointers via
`vector<unique_ptr<Tensor>>`) and a single contiguous **bump arena** for tensor
data (`mg-tensor.hpp:82-107`). Allocation is a pointer bump, 32-byte aligned,
that throws on overflow (`mg-tensor.cpp:54-61`):

```cpp
void* Context::alloc(size_t bytes) {
    off_ = (off_ + ALIGN - 1) & ~(ALIGN - 1);
    if (off_ + bytes > arena_.size())
        throw std::runtime_error("mg::Context arena out of memory (increase arena_bytes)");
    void* p = arena_.data() + off_;
    off_ += bytes;
    return p;
}
```

`reset()` rewinds `off_` to 0 and drops all node objects, so the same arena is
reused across decode iterations with zero re-allocation
(`mg-tensor.cpp:52`; used at `mg-generate.cpp:71`). The reference path's arenas
are sized generously (1.5 GB transformer, 3 GB VQGAN) and never freed mid-graph —
the source of the 5.4 GB RSS row (see §10).

### Zero-copy views and externals

`reshape`, `permute`, and `cont` are graph ops, but `reshape`/`permute` allocate
no data: `make_view()` sets `t->data = a->data` and `is_view = true`
(`mg-tensor.cpp:155-178`). `permute` just reorders `ne[]`/`nb[]` entries — the
strided result is consumed directly by stride-aware kernels, and only `cont`
materializes a contiguous copy when a kernel needs one. `external()` wraps
caller-owned memory (e.g. mmap'd GGUF weights) with no copy
(`mg-tensor.cpp:86-91`).

### Graph build (topo sort)

`Graph::build_forward(out)` does a post-order DFS from the output, pushing each
node after its sources, producing a topologically sorted node list. `Op::None`
nodes are leaves (weights / inputs) and are skipped; a linear `seen()` check
dedupes shared subgraphs (`mg-tensor.cpp:184-195`). Execution is just iterating
that list (`mg-cpu.cpp:223-225`).

---

## 3. Reference CPU kernels

`src/mg-cpu/mg-cpu.cpp` is the F32 backend: a graph executor plus one scalar
kernel per op. The design philosophy is stated at the top
(`mg-cpu.cpp:1-5`): **correctness-first, stride-aware scalar code that matches
PyTorch numerically**; SIMD, threading, and im2col are deferred. `n_threads` is
accepted but ignored (`mg-cpu.cpp:223`).

### Stride-aware addressing

Every read goes through a single strided address helper, so kernels operate
directly on permuted/reshaped views without first materializing them:

```cpp
inline const float* fptr(const Tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(t->data) +
        i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
}
```
(`mg-cpu.cpp:17-21`)

A companion `bidx()` collapses a source dim of size 1 to index 0, giving free
NumPy-style broadcasting for `Add`/`Mul` (`mg-cpu.cpp:23`). Destinations are
always freshly allocated and contiguous, so they are written with a flat index.

### The kernels

- **`k_mul_mat`** (`mg-cpu.cpp:48-63`) — the batched, stride-aware GEMM.
  `w[K,N]·x[K,M] → dst[N,M]` with `w` broadcasting over the two batch dims
  (`bidx(w,2,…)`, `bidx(w,3,…)`). This single kernel serves linear layers *and*
  the per-head attention matmuls.
- **`k_soft_max`** (`mg-cpu.cpp:79-93`) — over `ne[0]`, numerically stabilized
  (subtract row max), with an optional fused scale applied before the max so the
  `1/√d` attention scaling is exact.
- **`k_norm`** (`mg-cpu.cpp:96-112`) — LayerNorm over `ne[0]`, no affine
  (affine is composed separately, see §4). Mean/variance accumulate in `double`
  for stability, matching PyTorch's reduction precision.
- **`k_group_norm`** (`mg-cpu.cpp:115-139`) — over layout `{W,H,C,N}`, normalizing
  per `(sample, group)` across `W·H·(C/groups)`, again with `double`
  accumulators.
- **`k_gelu`** (`mg-cpu.cpp:141-147`) — **exact erf GELU**, not the tanh
  approximation, so it matches `torch.nn.GELU()`:
  ```cpp
  o[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * inv_sqrt2));
  ```
- **`k_silu`** (`mg-cpu.cpp:149-154`) — swish `x·sigmoid(x)`, used as VQGAN's
  activation.
- **`k_conv_2d`** (`mg-cpu.cpp:157-179`) — a **direct** (not im2col) convolution
  with explicit padding skip. Kernel `{KW,KH,IC,OC}` × input `{IW,IH,IC,N}`. The
  straightforward 6-deep loop is chosen for obvious correctness; im2col+matmul is
  the planned optimization (CLAUDE.md:189, 305).
- **`k_get_rows`** (`mg-cpu.cpp:65-77`) — gather rows of `a` by int32 indices,
  with **negative-index wrap** (`if (r < 0) r += R`) to reproduce PyTorch's
  `-1`/mask-token addressing.
- **`k_upscale`** (`mg-cpu.cpp:181-187`) — nearest neighbour: output `(ow,oh)`
  reads source `(ow/f, oh/f)`.
- **`k_cont`** (`mg-cpu.cpp:189-197`) — materialize a contiguous copy from a
  strided view (the only place `permute` results get realized).

The exact-vs-approx choices (erf GELU, double-accumulate norms, direct conv,
stabilized softmax) are precisely what let the reference match the PyTorch oracle
to ~1e-5 (see §9). View ops (`Reshape`/`View`/`Permute`) are no-ops at exec time
since their data already aliases the source (`mg-cpu.cpp:214-215`).

---

## 4. Building the transformer

`src/mg-transformer.cpp` builds the bidirectional transformer forward graph,
mirroring `reference/maskgit/nets/bidirectional_transformer.py`
(`mg-transformer.cpp:1-6`). Config for the 256 model:
`n_layer=24, n_head=16, head_dim=48, n_embd=768, n_ffn=3072, vocab=2025,
mask_token_id=2024` (defaults at `mg-model.hpp:17-22`; head dims read at
`mg-transformer.cpp:33`).

> Note: `head_dim·n_head = 48·16 = 768 = n_embd`. CLAUDE.md's prose mentions "8
> heads × 96"; the actual checkpoint metadata loaded here is **16 heads × 48**.

### Embeddings + position + LN

```cpp
Tensor* tok = get_rows(c, m.require("token_embd.weight"), token_ids);  // {768,S,B}
Tensor* pe  = m.require("pos_embd.weight");                            // {768,257}
... Tensor* pos = c.external(Type::F32, 2, pne, pe->data);             // prefix view {768,S}
Tensor* x = add(c, tok, pos);
x = layer_norm_affine(c, x, "token_embd_norm.weight", "token_embd_norm.bias", eps);
```
(`mg-transformer.cpp:38-44`)

Token embeddings are gathered, the learned position table is added (a zero-copy
prefix view of the `[768,257]` table is taken so only the first `S` rows are
used), and a LayerNorm is applied. `layer_norm_affine` composes the affine from
primitives: `add_bias(mul(norm(x), w), b)` (`mg-transformer.cpp:15-17`). The
no-affine `Norm` op plus separate `mul`/`add` keeps the kernel set minimal.

### 16-head attention as batched strided matmuls

Per layer (post-norm), Q/K/V come from fused-named projections, then heads are
split by reshape + permute (zero-copy):

```cpp
q = permute(c, reshape(c, q, {D, H, S, B}), 0, 2, 1, 3);   // {D,S,H,B}
... scores = mul_mat(c, k, q);                              // {S(keys),S(query),H,B}
scores = soft_max(c, scores, attn_scale);
Tensor* vp = permute(c, v, 1, 0, 2, 3);                     // {S(t),D,H,B}
Tensor* attn = mul_mat(c, vp, scores);                      // {D,S(query),H,B}
attn = cont(c, permute(c, attn, 0, 2, 1, 3));               // {D,H,S,B} contiguous
attn = reshape(c, attn, {n_embd, S, B});
```
(`mg-transformer.cpp:56-70`)

The trick is mapping multi-head attention onto the single `mul_mat` op by putting
the head dim into a batch axis (`ne[2]`) and relying on the strided GEMM. `mul_mat`
contracts over `ne[0]`:

- **Scores:** `K{D,S,H,B}` and `Q{D,S,H,B}` share `ne[0]=D`, so `mul_mat(k,q)`
  gives `scores[t,s,head,b] = Σ_d K[d,t]·Q[d,s]` → `{S_keys, S_query, H, B}`. The
  `1/√D` scale is fused into `soft_max` over keys (`ne[0]=t`).
- **Context:** permuting `V` to `{S_t, D, H, B}` makes `ne[0]=S_t`, so
  `mul_mat(vp, scores)` contracts over keys → `attn[d,s,head,b]`.

No data is copied for the reshapes/permutes; only the final merge does one `cont`
to get a contiguous `{768,S,B}` for the output projection. This is exactly why
the stride-aware `mul_mat` (§3) matters — it consumes permuted views directly.

### Fused-QKV split, residuals, post-norm

The converter stores Q/K/V already split into separate weights
(`tools/convert_to_gguf.py:9`), so `linear()` just calls `mul_mat` + `add_bias`
per projection (`mg-transformer.cpp:20-25, 51-53`). Each sublayer is **post-norm**:
`x = LN(x + sublayer(x))` (`mg-transformer.cpp:73-83`), matching the upstream
architecture comment (`mg-transformer.cpp:5`). The FFN is `Linear → GELU →
Linear` (`mg-transformer.cpp:78-80`).

### Tied MLM head

The output head re-uses the input token-embedding matrix (weight tying):

```cpp
Tensor* hd = linear(c, m, "output_proj.weight", "output_proj.bias", x);
hd = gelu(c, hd);
hd = layer_norm_affine(c, hd, "output_norm.weight", "output_norm.bias", eps);
Tensor* logits = mul_mat(c, m.require("token_embd.weight"), hd);   // {vocab,S,B}
logits = add_bias(c, logits, m.require("output.bias"));
```
(`mg-transformer.cpp:87-93`)

`token_embd.weight` is `ne={E, V}`, so passing it as `w` to `mul_mat` against
`hd{E,S,B}` contracts over `E` and yields `{vocab, S, B}` — i.e. `hd @
token_embd.weightᵀ`. Only a separate `output.bias` is needed.

---

## 5. Building the VQGAN decoder

`src/mg-vqgan.cpp` builds the decoder graph, mirroring
`reference/maskgit/nets/vqgan_tokenizer.py` (`mg-vqgan.cpp:1-6`). Activations use
layout **`ne = {W, H, C, N}`**. Config: `filters=128`,
`channel_mult=[1,1,2,2,4]`, `num_res_blocks=2`, `GroupNorm` 32 groups eps 1e-5,
swish activation, `embedding_dim=256`.

### Stages

1. **Codebook gather → spatial layout** (`mg-vqgan.cpp:69-71`): `get_rows` from
   `vqgan.quantizer.codebook.weight`, reshape to `{C,W,H,1}`, then `cont(permute)`
   to `{W,H,C,1}`.
2. **`conv_in`** 256→512, 3×3 pad 1, biased (`mg-vqgan.cpp:74`).
3. **Mid block**: `nrb` ResBlocks at the top channel count (512)
   (`mg-vqgan.cpp:77-78`).
4. **Upsample stages** iterating `channel_mult` in reverse
   (`mg-vqgan.cpp:82-96`): `nrb` ResBlocks at the current width, then (except the
   last) a nearest 2× `upscale` followed by a 3×3 post-upsample conv. Four
   upsamples take 16×16 → 256×256.
5. **Output**: GroupNorm → swish → `conv_out` 128→3 (`mg-vqgan.cpp:98-100`).

GroupNorm affine is composed by reshaping the 1-D weight/bias to `{1,1,C,1}` and
applying `mul`/`add` so they broadcast over `W,H` (`mg-vqgan.cpp:21-27`).

### The ResBlock shortcut "quirk"

`norm0 → swish → conv0 → norm1 → swish → conv1 → (+ shortcut)`. The upstream
implementation has an unusual shortcut: when in/out channels differ, the 1×1
shortcut conv is applied to the **processed** tensor `h`, *not* the input
residual. The runtime faithfully reproduces this (`mg-vqgan.cpp:50-53`):

```cpp
if (in_dim != out_dim) {
    // QUIRK: shortcut conv applies to the processed h, not the input residual.
    residual = conv(c, m, pfx + "conv_res", h, 1, 0, /*bias=*/false);  // 1x1
}
return add(c, h, residual);
```

This is a numerical-fidelity decision: it does not "fix" the architecture, it
reproduces the checkpoint's actual behaviour so outputs match the oracle.

---

## 6. GGUF format & loader

Weights ship as a single **GGUF v3** file. `tools/convert_to_gguf.py` merges the
PyTorch export (`maskgit_transformer_f32.npz` + `maskgit_vqgan_f32.npz`) into one
file (transformer GGUF-named tensors with QKV pre-split, plus `vqgan.*`), writing
hyperparameters as GGUF metadata KVs (`convert_to_gguf.py:4-19, 121-148`). Dims
are stored innermost-first (reversed NumPy shape) so `ne[0]` is the last NumPy
axis, and data is the raw C-order buffer (`convert_to_gguf.py:18-19, 170-180`).
The VQGAN **encoder is dropped** (decode-only) to shrink the file
(`convert_to_gguf.py:164-165`).

The loader (`src/mg-model.cpp`) is an **mmap, zero-copy** reader:

- `mmap(PROT_READ, MAP_PRIVATE)` the whole file (`mg-model.cpp:145`); the `Model`
  destructor `munmap`s (`mg-model.cpp:198-201`).
- A `Cursor` parses magic/version (must be 3), tensor count, KV count, then a
  first pass reads `general.alignment` (default 32, `mg-model.cpp:109-126`), the
  metadata pass fills `HParams` (`mg-model.cpp:52-104`), and a tensor-info pass
  records name/dims/type/offset (`mg-model.cpp:170-182`).
- Each tensor becomes a **zero-copy external** pointing straight into the mmap'd
  region — no weight bytes are copied:
  ```cpp
  void* dptr = const_cast<uint8_t*>(b) + data_start + ti.off;
  Type mt = ti.type == GGML_I8 ? Type::I8 : (ti.type == MG_I4 ? Type::I4 : Type::F32);
  Tensor* t = ctx.external(mt, ti.n_dims, ti.ne, dptr);
  ```
  (`mg-model.cpp:188-194`)

Three tensor type codes are accepted: `GGML_F32=0`, `GGML_I8=24`, and the project's
own packed-int4 code `MG_I4=100` (`mg-model.cpp:22`; written at
`convert_to_gguf.py:34-36`). Config is fully metadata-driven — `HParams`
(`mg-model.hpp:17-32`) carries transformer + VQGAN + sampling settings, and the
graph builders read everything from it (e.g. `vq_channel_mult`,
`choice_temperature`, `mask_token_id`), so the runtime has no hardcoded
architecture.

---

## 7. XNNPACK backend

The XNNPACK backend (`src/mg-xnn.cpp`, `src/mg-xnn-vqgan.cpp`) takes a
**full-subgraph** approach: the entire transformer (or VQGAN decoder) is defined
once as an `xnn_subgraph_t`, an `xnn_runtime_t` is created from it in the
constructor, and each decode step just rebinds external buffers and invokes the
runtime. This **build-once / invoke-per-step** structure is the key latency
property — operator fusion, microkernel selection, and weight packing happen once
at construction, then every step is pure compute.

### Transformer subgraph (`src/mg-xnn.cpp`)

The constructor builds the graph with two external values (input embeddings,
output logits) and a set of lambda helpers for each primitive
(`mg-xnn.cpp:32-193`). Op mapping vs the reference:

| Runtime concept | XNNPACK primitive |
|---|---|
| Linear / `mul_mat` w/ weight | `xnn_define_fully_connected` (`mg-xnn.cpp:159,161`) |
| Attention `Q·Kᵀ`, `attn·V` | `xnn_define_batch_matrix_multiply` (`mg-xnn.cpp:223,229`) — QK uses `XNN_FLAG_TRANSPOSE_B` |
| softmax | `xnn_define_softmax` (`mg-xnn.cpp:226`) |
| GELU | native `xnn_unary_gelu` (`mg-xnn.cpp:239,247`) |
| LayerNorm | **composed** (see below) |
| reshape / permute | `xnn_define_static_reshape` / `xnn_define_static_transpose` (`mg-xnn.cpp:165-174`) |

XNNPACK has no native gather, so the **embedding lookup is done host-side**: each
step `forward()` copies `token_embd` rows into the external input buffer (with
negative-index wrap) before invoking (`mg-xnn.cpp:262-273`). XNNPACK dims are the
*reverse* of `ne[]` (`mg-xnn.cpp:44-52`), and activations are kept as `{B,S,E}`
with attention reshaped to `{B,S,H,D}` → transpose → `{B,H,S,D}`
(`mg-xnn.cpp:206-231`) — the NCHW-style analogue of the reference's permute dance.

**Composed LayerNorm.** XNNPACK has no LayerNorm op, so it is built from
primitives over the last dim (`mg-xnn.cpp:178-193`):

```cpp
check(xnn_define_static_reduce(sg, xnn_reduce_mean, 1, &last, x, mean, XNN_FLAG_KEEP_DIMS), "mean");
uint32_t xc  = binary(xnn_binary_subtract, x, mean, full);
uint32_t sq  = unary(xnn_unary_square, xc, full);
... reduce_mean -> var; binary_add eps; reciprocal_square_root -> inv;
uint32_t xn  = binary(xnn_binary_multiply, xc, inv, full);  // normalized
... multiply by weight; add bias                            // affine
```

The tied MLM head is one final `fully_connected` against `token_embd.weight` into
the external output (`mg-xnn.cpp:251-255`).

### VQGAN subgraph (`src/mg-xnn-vqgan.cpp`)

Built entirely in **NHWC** (XNNPACK's conv layout), with the codebook gather again
host-side (`mg-xnn-vqgan.cpp:248-255`). Notable mappings:

- **Conv weight repack OIHW→OHWI.** PyTorch conv weights (`ne={KW,KH,IC,OC}`) are
  repacked into XNNPACK's `[OC,KH,KW,IC]` filter layout into owned buffers
  (`mg-xnn-vqgan.cpp:81-93`).
- **Composed GroupNorm over `[N,H,W,G,Cg]`.** The NHWC tensor is reshaped to a
  5-D `{1,H,W,GROUPS,Cg}`, mean/var reduced over axes `{1,2,4}`, normalized, then
  reshaped back and given per-channel affine (`mg-xnn-vqgan.cpp:155-173`).
- **Nearest 2× upsample as a depthwise all-ones deconvolution.** XNNPACK has no
  nearest-upsample op, so it is expressed as a stride-2 grouped (one group per
  channel) transposed convolution with a `2×2` all-ones filter and zero bias —
  each input pixel is replicated into a 2×2 block:
  ```cpp
  packed_.emplace_back((size_t)C*2*2*1, 1.0f);           // ones filter [C,2,2,1]
  ... xnn_define_deconvolution_2d(sg, 0,0,0,0, 0,0, 2,2, 2,2, 1,1,
        /*groups=*/C, /*gic=*/1, /*goc=*/1, ..., x, fid, bid, out, 0)
  ```
  (`mg-xnn-vqgan.cpp:176-185`)
- **swish = sigmoid·x** built from `xnn_unary_sigmoid` + `xnn_binary_multiply`
  (`mg-xnn-vqgan.cpp:76-78`).
- The same ResBlock shortcut quirk is reproduced (`mg-xnn-vqgan.cpp:196-197`).

### Why it is so much faster

Measured (README.md:36-37, 74-75): XNNPACK F32 is **~40× faster on the
transformer** and **~115× on VQGAN** than the scalar reference (731 s → 13.8 s
end-to-end). The reasons:

1. **Optimized microkernels** — XNNPACK selects NEON / dotprod / i8mm
   GEMM/conv microkernels; the reference is plain scalar triple loops.
2. **Build-once subgraph** — graph construction, fusion, and weight packing are
   amortized across all decode steps; per-step cost is just `setup` + `invoke`.
3. **Direct conv microkernels** vs the reference's naive direct conv.
4. **Threadpool-capable** runtime (created via `xnn_create_runtime_v2`,
   `mg-xnn.cpp:258`).

---

## 8. Quantization

Quantization targets the **transformer FC weights** and, on-load, the **VQGAN
conv weights**. Modes are `F32 | Q8 | Q4` (`mg-model.hpp:15`). The strategy is
ggml-style **per-output-channel symmetric** quantization with F32 activations
dynamically quantized at runtime.

### Per-channel int8 (qc8w)

Each output channel gets its own scale `amax/127`, weights clamped to `[-127,127]`
(`convert_to_gguf.py:39-46`; on-load mirror at `mg-xnn.cpp:86-101`). XNNPACK
consumes these as `xnn_datatype_qcint8` channelwise tensors with `channel_dim=0`
(`mg-xnn.cpp:104-107`).

### Per-channel int4 (qc4w), packed

Int4 uses scale `amax/7` and **clamps to `[-7,7]`** (not the full `[-8,7]` of
two's complement) — an XNNPACK GEMM requirement — packed **two nibbles per byte**,
low nibble = even input-channel index (`convert_to_gguf.py:49-58`;
`mg-xnn.cpp:128-139`):

```cpp
int v0 = qz(w[oc*IC + 2*k]   * inv);
int v1 = qz(w[oc*IC + 2*k+1] * inv);
q[oc*(IC/2) + k] = (uint8_t)((v0 & 0xF) | ((v1 & 0xF) << 4));
```

Fed to XNNPACK as `xnn_datatype_qcint4` with **`zero_point = 0`** (symmetric)
(`mg-xnn.cpp:143-145`). The packed type halves storage (`I4` reports
`(nelements()+1)/2` bytes, `mg-tensor.cpp:30`).

### Dynamic int8 activations (qd8)

For any quantized FC the *activation* is quantized per-row to int8 at runtime via
a `qdint8` dynamically-quantized value plus an `xnn_unary_convert`, then the int8
GEMM (`qd8-f32-qcXw`) produces F32 directly (`mg-xnn.cpp:150-159`). Activations
stay F32 in the graph; only the matmul input is transiently int8. This is what
makes the int8/int4 GEMMs use ARM dotprod/i8mm.

### Quantize-on-load vs pre-quantized GGUF

Two paths exist:

- **Pre-quantized GGUF** — the converter stores int8/int4 bytes + a `.scale`
  tensor; the backend detects the stored weight type and points XNNPACK straight
  at the mmap'd bytes, **zero-copy** (`mg-xnn.cpp:27-30, 77-84, 114-121`).
- **Quantize-on-load** — when the GGUF stores F32, the backend quantizes into
  owned buffers at subgraph-build time (`mg-xnn.cpp:86-107, 123-146`).

This yields **small model files**: int8 GGUF is 288 MB and int4 is 207 MB vs 775
MB F32 (README.md:38-39). Conv stays F32 in the quantized files and is quantized
on-load (the reason int4 is 207 MB rather than the ~125 MB ideal — see §11).

### Mixed precision

Only FC weights (and, on-load, conv) are quantized. **Embeddings, position table,
LayerNorm/GroupNorm affines, and all biases stay F32** — `is_fc()` explicitly
excludes `token_embd` and biases (`convert_to_gguf.py:151-154`). These tensors are
small but precision-sensitive, so keeping them F32 costs little and preserves
accuracy. The verification table shows int8 stays near-lossless (cosine ≈
0.99999) and int4 keeps cosine ≈ 0.9998 (README.md:72-73).

---

## 9. Verification methodology

Verification compares the C++ runtime to the PyTorch oracle (M1) at **component
boundaries**, on **fixed inputs**, isolating kernel/quantization error from the
stochastic sampling RNG (which differs between PyTorch and C++).

### Fixed-input dumps

`reference/dump_verify.py` runs the genuine PyTorch model on two fixed inputs and
dumps the references (`dump_verify.py:4-15, 31-53`):

| File | Shape | What |
|---|---|---|
| `verify_tokens.bin` | int32 `[S=257]` | class-207 transformer input (label + all-mask) |
| `verify_logits.bin` | f32 `[S, 2025]` | PyTorch logits for it |
| `verify_grid.bin` | int32 `[16,16]` | fixed VQGAN token grid |
| `verify_image.bin` | f32 `[256,256,3]` | PyTorch decode of that grid |

Crucially the grid is a *fixed* random grid (`RandomState(0)`,
`dump_verify.py:46-47`), not a sampled one — so the VQGAN comparison never depends
on the transformer's sampling.

### Component comparison

`tools/verify-transformer.cpp` feeds `verify_tokens.bin` through
`build_transformer` and compares logits to `verify_logits.bin` via
**max-abs / mean-abs diff and cosine similarity** (`verify-transformer.cpp:47-65`),
passing if `max_abs < 2e-2 && cosine > 0.99999` (`verify-transformer.cpp:66`).
`verify-xnn-transformer.cpp`, `verify-vqgan.cpp`, and `verify-xnn-vqgan.cpp` do
the same for the other backend/component (`verify-xnn-vqgan.cpp:47-62`), with the
int8-conv tolerance widened to `cosine > 0.999` (`verify-xnn-vqgan.cpp:60-61`).

Because errors at any internal layer propagate to the final logits / image, a
boundary match pins down whole-graph correctness even though per-layer dumping is
not yet implemented (README.md:241-258). Kernel unit tests
(`tests/test_kernels.cpp`) and a GGUF round-trip (`mg-model-info`) catch
stride/indexing and serialization bugs independently.

### Why this isolates kernel error

The sampling RNG (gumbel + categorical, §1) makes end-to-end pixels diverge
between PyTorch and C++ even when every kernel is correct. By feeding the same
fixed tensors and comparing *pre-sampling* outputs (logits) and a *fixed-grid*
decode (image), the metrics measure only kernel/graph/quantization error. Observed
results (README.md:68-76): F32 (both backends) matches to ~2.6e-6 / cosine
1.0000000; int8 logits cosine 0.9999951; int4 logits cosine 0.9998129.

---

## 10. Performance & memory insights

Measured end-to-end (class 207, seed 42, 8 steps, 256×256), from README.md:34-42:

| Kernel | Precision | Model file | Machine | Latency | Peak RSS |
|---|---|---|---|---|---|
| reference (scalar) | F32 | 775 MB | M1 Max | 731 s | 5451 MB |
| XNNPACK | F32 | 775 MB | M1 Max | 13.8 s | 2398 MB |
| XNNPACK | int8 | 288 MB | M1 Max | 3.9 s | 928 MB |
| XNNPACK | int4 | 207 MB | M1 Max | 4.1 s | 767 MB |
| XNNPACK | F32 | 775 MB | Pixel 9 | 15.4 s | 1977 MB |
| XNNPACK | int8 | 288 MB | Pixel 9 | 4.2 s | 839 MB |
| XNNPACK | int4 | 207 MB | Pixel 9 | 4.0 s | 676 MB |

**Why these numbers, mechanistically:**

- **Reference → XNNPACK F32 (~53×):** optimized NEON GEMM/conv microkernels vs
  scalar triple loops, plus build-once subgraph fusion (§7). The transformer
  alone is ~40× and the VQGAN ~115×.
- **F32 → int8 (~3.5× more):** int8 `qd8-f32-qc8w` GEMM lights up ARM **dotprod /
  i8mm** instructions (4× throughput vs scalar per the design notes,
  CLAUDE.md:304), and a 288 MB working set has far better cache behaviour than 775
  MB.
- **int8 ≈ int4 in latency** but smaller file: the int4 GEMM is dominated by the
  same dynamic-activation quantize + dot-product cost; the win is **file size and
  RSS** (207 MB / 767 MB vs 288 MB / 928 MB), important for small devices.
- **Model size ↔ RSS:** RSS tracks the model file because **mmap weights are
  zero-copy** (§6) — the resident set is essentially the touched weight pages plus
  XNNPACK's quantized/packed buffers and activation arenas, not a second copy.
- **Arena reuse:** the decode loop reuses one transformer context across all 8
  steps (`mg-generate.cpp:71`), so step count does not multiply memory.

**The reference path's 5.4 GB artifact.** The reference row's RSS is almost
entirely its **unoptimized bump arenas** — `1536 MB` for the transformer
(`mg-generate.cpp:62`) and `3 GB` for the VQGAN (`mg-generate.cpp:132`) — that are
sized for worst case and never freed mid-graph. This is a property of the
correctness-first reference path (every intermediate is materialized into the
arena), not a fundamental requirement; the README calls it out explicitly
(README.md:46-48). The XNNPACK path skips the transformer arena entirely (the
`ctx` is only created when no callback is set, `mg-generate.cpp:61-62`).

**On-device.** The runtime cross-compiles to Android arm64-v8a with NDK r27
clang, `-O3 -march=armv8.4a+dotprod -static-libstdc++`, linking the Android
XNNPACK static libs into a self-contained PIE binary
(`scripts/build_android.sh:25-36`). Output is bit-identical to the host
(README.md:124).

---

## 11. Known issue: VQGAN conv pre-quant NaN

There is one open issue, documented fully in `docs/KNOWN_ISSUES.md`. When the
VQGAN decoder conv weights are stored **pre-quantized** as per-channel int8 in the
GGUF and fed to XNNPACK's `xnn_define_convolution_2d` as a `qd8-f32-qc8w`
convolution, the decoded image comes out **all NaN** from the first conv — even
though the stored int8 bytes and scales are verified **byte-identical** to the
working **on-load** quantization path (which produces a correct image, cosine
0.99939). The transformer FC pre-quant path (`qc8w`/`qc4w`) works fine; only conv
pre-quant fails.

**Workaround (current behaviour):** quantized GGUFs keep VQGAN conv weights **F32**
and quantize them **on load**; only transformer FC weights are stored
pre-quantized. The pre-quant conv code path throws if reached
(`src/mg-xnn-vqgan.cpp:101-102`; converter no-op note at
`convert_to_gguf.py:175-178`). This is why `q4` is ~207 MB rather than the ~125 MB
ideal — the F32 conv weights are the remaining bulk. The leading hypothesis is a
conv-specific weight packing/alignment expectation in XNNPACK's `qd8-f32-qc8w`
microkernel that the fresh on-load buffers satisfy but the GGUF-resident layout
does not. See `docs/KNOWN_ISSUES.md` for the full investigation and repro.
