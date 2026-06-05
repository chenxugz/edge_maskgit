# MaskGIT On-Device C++ Runtime — Technical Deep Dive

This document explains how the MaskGIT class-conditional image generator is
implemented as a from-scratch, ggml-inspired C++17 inference runtime. It covers
the tensor library and graph, the correctness-first reference CPU backend, the
accelerated XNNPACK backend, the from-scratch OpenCL GPU backend (§12),
int8/int4 quantization, the verification methodology, and the measured
performance / memory results (host macOS arm64 + Pixel 9). It is written so an
engineer can understand the implementation from this document plus the cited
source.

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

A **third backend** — a from-scratch **OpenCL GPU backend** (`src/mg-opencl/`) —
executes the *same* `mg::Graph` the reference path builds, one kernel per node, on
a mobile/desktop GPU. It is documented in full in §12, including its matmul
optimization journey (naive → register-blocked → tiled local-memory GEMM).

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

---

## 12. OpenCL GPU backend

The third backend (`src/mg-opencl/mg-opencl.cpp`, public API
`include/mg-opencl.hpp`) runs the **same `mg::Graph`** the reference and
transformer/VQGAN builders produce — but on a GPU, one OpenCL kernel per node.
It is a from-scratch backend: the kernels are hand-written OpenCL C in a single
raw-string `kKernels` (`mg-opencl.cpp:29-340`), compiled at startup, and dispatched
by a graph walker (`OpenCLRuntime::compute`, `mg-opencl.cpp:424-566`). The whole
class-id → image pipeline (8 transformer decode steps + VQGAN decode) runs on the
GPU; it is validated on the M1 Max GPU (host) and a Pixel 9 Mali-G715 (Android).

This section is the **optimization journey** for the matmul — the kernel that
dominates the runtime — from a naive one-thread-per-output kernel to a tiled
local-memory GEMM, with the measured numbers at each step. It is self-contained:
the GPU concepts it needs are introduced inline. For the *methodology* behind these
wins (profiling loop, roofline, the M6 hill-climb and its negative results), see §13.

### A few GPU concepts (just enough)

- A **work-item** is one thread; OpenCL launches an N-D grid of them
  (`clEnqueueNDRangeKernel`). `get_global_id(d)` is the work-item's index along
  dimension `d`.
- Work-items are grouped into **workgroups** (`get_group_id(d)` is the group
  index, `get_local_id(d)` is the index *within* the group). A workgroup is the
  unit of cooperation.
- `__global` memory is the large, slow device DRAM. `__local` memory is small,
  fast on-chip memory **shared by all work-items in a workgroup** — the lever for
  cutting global-memory traffic.
- `barrier(CLK_LOCAL_MEM_FENCE)` synchronizes a workgroup: every work-item waits
  until all of them reach the barrier, so a value one work-item wrote to `__local`
  is visible to the others.
- **Arithmetic intensity** = useful FLOPs per byte read from global memory. A
  kernel that re-reads the same operand many times has low intensity and is
  **memory-bound** — its speed is set by DRAM bandwidth, not the ALUs. Raising
  intensity (reading each byte fewer times) is the whole game here.

### 12.1 Backend architecture

The backend mirrors the reference `mg::compute()` exactly: `compute(Graph&)` walks
`g.nodes` (the topologically sorted node list from §2) in order and `switch`es on
`t->op`, dispatching the matching kernel (`mg-opencl.cpp:429-545`). The dispatch
structure is identical to the reference executor — only the per-op body differs
(enqueue a kernel instead of running a scalar loop).

**Per-tensor device-buffer cache (`buf()`).** Every `Tensor*` that needs storage
on the GPU gets one `cl_mem` buffer, cached in an `unordered_map<Tensor*, cl_mem>`
keyed by the host tensor pointer (`mg-opencl.cpp:352, 363-374`):

```cpp
cl_mem buf(Tensor* t) {
    if (t->op == Op::Reshape || t->op == Op::Permute || t->op == Op::View)
        return buf(t->src[0]);          // views share the source's buffer
    auto it = bufs.find(t);
    if (it != bufs.end()) return it->second;
    cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE, t->nbytes(), nullptr, &e);
    if (t->op == Op::None && t->data)   // leaf: upload host data once
        clEnqueueWriteBuffer(q, m, CL_TRUE, 0, t->nbytes(), t->data, 0, ...);
    bufs[t] = m; return m;
}
```
(`mg-opencl.cpp:363-374`)

Two things fall out of this:

- **Views resolve to their source's buffer.** `reshape`/`permute`/`view` allocate
  no data in the tensor library (§2 — they alias the source via strides), so `buf()`
  recurses to `src[0]`. The strided *addressing* is reapplied inside the kernels,
  not by materializing the view (see §12.4 on how attention's permuted views feed
  the matmul). The `compute()` loop therefore treats `Reshape`/`Permute`/`View`
  nodes as no-ops (`mg-opencl.cpp:431-432`), exactly like the reference.
- **Leaf tensors (weights / inputs) are uploaded once.** `Op::None` tensors with
  host data — the mmap'd GGUF weights and the input token buffer — are
  `clEnqueueWriteBuffer`'d the first time `buf()` is asked for them, then stay
  cached across calls.

**Keep intermediates on the GPU; read back only the final node.** This is the key
architectural decision. A naive executor would read each node's result back to
host after computing it; with hundreds of nodes per graph that is hundreds of
**blocking** device→host transfers (each forces a `clFinish`-style sync, stalling
the pipeline). Instead, every intermediate stays in its `cl_mem`, and only the
graph's output node is copied back (`mg-opencl.cpp:546-554`):

```cpp
// Keep all intermediates on the GPU; read back only the final output node.
if (!g.nodes.empty()) {
    Tensor* out = g.nodes.back();
    clEnqueueReadBuffer(I.q, I.buf(out), CL_TRUE, 0, out->nbytes(), out->data, ...);
}
clFinish(I.q);
```

The whole graph is enqueued as a stream of kernels with no intermediate
synchronization; the GPU runs them back-to-back and the single readback at the end
is the only host↔device round-trip. This alone is what makes the executor usable —
the commit history calls the readback-per-node version "the naive executor that was
dominated by syncs."

**Layout mapping.** mg's row-major convention (§2 — `ne[0]` is innermost /
contiguous, `nb[]` are byte strides) carries straight into the kernels. Kernels
that take strides receive them in *elements* (`Impl::es` divides `nb[d]` by
`sizeof(float)`, `mg-opencl.cpp:375`) and index with `ne[0]` as the fastest-varying
axis — e.g. a contiguous `out[N,M,...]` is written at `o[(((p3*B2+p2)*M + m)*N + n]`
(`mg-opencl.cpp:44`). This is what lets the GPU kernels consume the same tensors,
strides, and `mul_mat` contract (`ne[0]` is the contraction dim) as the reference.

### 12.2 Iterative-decoding buffer management (a stale-buffer bug)

The decode loop (§1) runs the transformer forward **8 times**, each time rebuilding
the graph into a `Context` that is `reset()` between steps (§2 — the arena rewinds
to offset 0 and drops node objects). Because the arena reuses the same memory, the
scratch tensors of step *N+1* land at the **same host addresses** as step *N*'s.
Since `buf()` keys its cache on `Tensor*` (the host address), step 2 would find a
*stale* cached buffer from step 1 sitting at that address — wrong contents, and in
practice a `CL_INVALID_VALUE`.

The fix has two parts:

1. **Release computed-node buffers after readback, keep leaves cached.** At the end
   of `compute()`, every non-leaf, non-view node's buffer is freed and dropped from
   the cache; weight/input leaf buffers (and views, which own nothing) are kept
   (`mg-opencl.cpp:556-565`):

   ```cpp
   for (Tensor* t : g.nodes) {
       if (t->op == Op::None || t->op == Op::Reshape ||
           t->op == Op::Permute || t->op == Op::View)
           continue;   // leaves have no own buffer to free; views alias their src
       auto it = I.bufs.find(t);
       if (it != I.bufs.end()) { clReleaseMemObject(it->second); I.bufs.erase(it); }
   }
   ```

   So scratch buffers never outlive the step that created them, and the expensive
   one-time weight uploads survive across all 8 steps.

2. **`invalidate()` the changed token leaf.** The input token buffer *is* a leaf
   (so it would stay cached), but its contents change every step as the decoder
   fills in tokens — while its host address is stable across the arena reset. The
   loop calls `invalidate(token_tensor)` to drop just that one cached buffer so the
   next `compute()` re-uploads the new tokens (`mg-opencl.cpp:419-422`,
   `mg-opencl.hpp:25-29`):

   ```cpp
   void OpenCLRuntime::invalidate(Tensor* t) {
       auto it = p_->bufs.find(t);
       if (it != p_->bufs.end()) { clReleaseMemObject(it->second); p_->bufs.erase(it); }
   }
   ```

Together these give the right invariant for iterative decoding: **weights uploaded
once, the token leaf re-uploaded each step, all scratch re-created fresh each step.**

### 12.3 The matmul optimization journey

The matmul is the hot kernel — transformer FC layers (QKV/output projections,
FFN), the tied MLM head, the attention score/context matmuls, and the VQGAN
conv-as-matmul all route through it. The journey below is the heart of the GPU
work; each stage is a real kernel in the source.

#### Stage 1 — naive dequant-fused matmul (one work-item per output)

The first quantized kernel computes one output element `o[n,m]` per work-item,
with a scalar register accumulator. For **ggml Q8_0** the weight row is stored as
`K/32` blocks of 34 bytes each (an fp16 scale + 32 int8 quants); the dequant
`d * q[i]` is **fused into the dot product**, so the kernel reads ¼ the weight
bytes of an F32 matmul (`mg-opencl.cpp:77-90`):

```cpp
__kernel void k_mul_mat_q8(__global const uchar* w, __global const float* x,
                           __global float* o, int K, int N, int M) {
    int n = get_global_id(0), m = get_global_id(1);
    if (n >= N || m >= M) return;
    int nb = K / 32; float acc = 0.0f;
    for (int b = 0; b < nb; b++) {
        __global const uchar* blk = w + (long)(n*nb + b) * 34;
        float d = vload_half(0, (__global const half*)blk);      // fp16 scale
        __global const char* qs = (__global const char*)(blk + 2);
        int kk = b*32;
        for (int i = 0; i < 32; i++) acc += d * (float)qs[i] * x[(long)m*K + kk + i];
    }
    o[(long)m*N + n] = acc;
}
```

The **Q4_K** variant (`k_mul_mat_q4k`, `mg-opencl.cpp:125-150`) is the same shape
over 256-weight super-blocks: an fp16 `d` + fp16 `dmin` + 12 bytes of packed 6-bit
(scale, min) pairs + 128 bytes of 4-bit quants (144 bytes total). A helper
`q4k_sm()` unpacks the 6-bit scale/min for a sub-block (`mg-opencl.cpp:116-122`),
and each nibble dequantizes as `y = d·scale·q − dmin·min`.

Reading ¼ the weight bytes makes this ~4× faster than F32-on-GPU (M1 Max
transformer forward: F32 2.81 s → Q8_0 0.73 s in the first cut). **But it is still
memory-bound.** Each work-item reads an entire weight row *and* an entire activation
column; across the whole grid, **each weight row is re-read M times and each
activation column N times** — roughly `2·N·K·M` global reads. The arithmetic
intensity is low: a multiply-add per pair of loaded values. The dequant trick cut
the *weight* traffic, not the *redundancy*.

#### Stage 2 — M register-blocking (`_b4`)

The first attack on redundancy: have each work-item compute **MR = 4 output columns
for one weight row**, so a dequantized weight block is computed **once** and reused
across the 4 activation columns — ¼ the weight-byte traffic again, but now via reuse
rather than via the quant format (`mg-opencl.cpp:96-114`):

```cpp
#define MRB 4
__kernel void k_mul_mat_q8_b4(__global const uchar* w, __global const float* x,
                              __global float* o, int K, int N, int M) {
    int n = get_global_id(0), m0 = get_global_id(1) * MRB;
    if (n >= N || m0 >= M) return;
    int mr = min(MRB, M - m0), nb = K / 32;
    float acc[MRB]; for (int r = 0; r < MRB; r++) acc[r] = 0.0f;
    for (int b = 0; b < nb; b++) {
        ... float d = ...; const char* qs = ...;
        for (int i = 0; i < 32; i++) {
            float wv = d * (float)qs[i];                 // dequant once...
            for (int r = 0; r < mr; r++)
                acc[r] += wv * x[(long)(m0+r)*K + kk + i]; // ...reused across 4 cols
        }
    }
    for (int r = 0; r < mr; r++) o[(long)(m0+r)*N + n] = acc[r];
}
```

**Critical gotcha — `MR` must be a compile-time constant.** `MRB` is a `#define`, so
`float acc[MRB]` is a fixed-size array the compiler can **register-allocate** and the
inner `for (r ...)` loops **unroll**. If `MR` were a runtime value (or `acc[1]` with a
dynamic count), the accumulator spills to **private memory** (off-chip), and the
kernel runs **~5× slower** — the register file is what makes the reuse free. The
commit measured this directly.

This was **device-adaptive**: register-blocking is a clear win on bandwidth-bound
mobile GPUs (Mali-G715 Q8_0 18.7 → 14.3 s), but it *regressed* the high-thread
desktop M1 Max (0.54 → 0.83 s), which prefers raw parallelism — fewer, fatter
work-items means a less-occupied GPU. So the host **selected the variant per device**
by name: `_b4` only on Mali/Adreno, the scalar kernel elsewhere
(`mg-opencl.cpp:393-397`):

```cpp
p_->mr = (p_->dev_name.find("Mali")   != std::string::npos ||
          p_->dev_name.find("Adreno") != std::string::npos) ? 4 : 1;
```

#### Stage 3 — tiled local-memory GEMM (the current best)

The register-blocked kernel still re-reads operands many times from global memory;
it just amortized the *weight* reads a bit. The textbook fix attacks the redundancy
head-on with **`__local` memory tiling** (the classic blocked GEMM), and it wins on
*both* desktop and mobile — so it **replaced the per-device selection** entirely (the
`_b4` kernels remain in the source for reference but are no longer dispatched).

A **16×16 workgroup** (`TS = 16`) computes a 16×16 tile of the output. It marches
along `K` in slabs of 16: at each step the 256 work-items **cooperatively** load one
16×16 slab of the activation `x` and one 16×16 slab of the (dequantized) weight `w`
into `__local` arrays `As`/`Bs`, barrier, then each work-item accumulates its dot
product over that slab from local memory. Because the slab is read from global memory
**once per workgroup** but used by all 16 work-items along each axis, **each operand
is read ~16× fewer times** from DRAM — arithmetic intensity goes up ~16×.

Here is the Q8_0 tiled kernel in full (`mg-opencl.cpp:201-225`):

```cpp
#define TS 16
__kernel void k_mul_mat_q8_t(__global const uchar* w, __global const float* x,
                             __global float* o, int K, int N, int M) {
    __local float As[TS][TS];   // x slab:  As[k_local][m_local]
    __local float Bs[TS][TS];   // w slab:  Bs[n_local][k_local] (dequantized)
    int tx = get_local_id(0), ty = get_local_id(1);
    int m = get_group_id(0)*TS + tx;        // activation column
    int n = get_group_id(1)*TS + ty;        // weight row
    int nb = K / 32;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        int ml = get_group_id(0)*TS + tx, kA = k0 + ty;       // As[ty][tx] = x[ml, kA]
        As[ty][tx] = (ml < M && kA < K) ? x[(long)ml*K + kA] : 0.0f;
        int nl = get_group_id(1)*TS + ty, kB = k0 + tx;       // Bs[ty][tx] = dequant w[nl, kB]
        float wv = 0.0f;
        if (nl < N && kB < K) {
            __global const uchar* blk = w + (long)(nl*nb + (kB>>5)) * 34;
            wv = vload_half(0, (__global const half*)blk)
               * (float)((__global const char*)(blk+2))[kB & 31];
        }
        Bs[ty][tx] = wv;
        barrier(CLK_LOCAL_MEM_FENCE);                          // slab fully loaded
        for (int kk = 0; kk < TS; kk++) acc += Bs[ty][kk] * As[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);                          // done reading before reload
    }
    if (n < N && m < M) o[(long)m*N + n] = acc;
}
```

Reading it carefully:

- **Cooperative load.** Each of the 256 work-items loads exactly one element of `As`
  and one of `Bs` per step; together they fill both 16×16 slabs. The weight slab is
  **dequantized as it is loaded** — `(kB>>5)` is the block index, `kB & 31` the offset
  within the 32-int8 block — so the dequant happens once per loaded element, not once
  per use.
- **The two barriers.** The first ensures the whole slab is in `__local` before any
  work-item reads it; the second ensures every work-item has finished reading the
  current slab before the next iteration overwrites it. Both are mandatory for
  correctness.
- **Bounds checks.** The transformer sequence length is **M = 257** (256 grid tokens
  + 1 class label), which is **not a multiple of 16**. The grid is rounded up to the
  next multiple of TS, and out-of-range loads write `0.0f` into the slab (a harmless
  zero contribution) while the final store is gated by `if (n < N && m < M)`. The
  small, non-aligned M is also *why* parallelism is limited and tiling matters: there
  just aren't many output columns to spread across the GPU.

The **Q4_K** tiled kernel (`k_mul_mat_q4k_t`, `mg-opencl.cpp:226-245`) is identical
in structure; the only change is the cooperative load calls a single-element
dequant helper `deq_q4k_elem()` (`mg-opencl.cpp:190-200`) that unpacks one Q4_K
weight `(n, k)` from the super-block storage — locating the super-block, sub-block,
nibble, and applying `d·scale·q − dmin·min`.

**The same tiling, generalized to the F32 attention matmul.** Multi-head attention's
`Q·Kᵀ` and `softmax·V` are *batched, strided* matmuls over **permuted views** (§4 —
the head dim lives in a batch axis and the operands are non-contiguous). The tiled
F32 kernel `k_mul_mat_t` (`mg-opencl.cpp:49-72`) is the *same* 16×16 local-memory
blocking, but it folds the per-operand strides and the head batch into the
cooperative load address instead of assuming contiguous rows:

```cpp
__kernel void k_mul_mat_t(__global const float* w, __global const float* x,
        __global float* o, int K, int N, int M, int B2,
        int ws0,int ws1,int ws2,int ws3, int xs0,int xs1,int xs2,int xs3,
        int wb2,int wb3) {
    __local float As[16][16];   // x slab
    __local float Bs[16][16];   // w slab
    int tx = get_local_id(0), ty = get_local_id(1), pq = get_global_id(2);
    int p2 = pq % B2, p3 = pq / B2;                       // unflatten the batch (head) index
    int wp2 = (wb2==1?0:p2), wp3 = (wb3==1?0:p3);         // w broadcasts over size-1 batch dims
    long wbase = (long)wp2*ws2 + (long)wp3*ws3;
    long xbase = (long)p2*xs2 + (long)p3*xs3;
    ...
    for (int k0 = 0; k0 < K; k0 += 16) {
        ... As[ty][tx] = (ml < M && kA < K) ? x[xbase + (long)ml*xs1 + (long)kA*xs0] : 0.0f;
        ... Bs[ty][tx] = (nl < N && kB < K) ? w[wbase + (long)nl*ws1 + (long)kB*ws0] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < 16; kk++) acc += Bs[ty][kk] * As[kk][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    ...
}
```

The third grid dimension `get_global_id(2)` is the flattened batch (= head × outer
batch), unpacked into `p2`/`p3`; the strides `xs0..xs3` / `ws0..ws3` are the view's
element strides (passed via `Impl::es`), so the **K-stride and head offset fold into
the load address** — the permuted Q/K/V views are consumed in place, no `cont`
materialization, exactly as the reference's stride-aware GEMM does (§3, §4). `w`
broadcasts over batch dims of size 1 (`wb2`/`wb3`), which is how a single weight
matrix applies across all heads.

**Dispatch.** `compute()` picks the kernel by weight type: Q8_0/Q4_K weights get the
quantized tiled kernel, everything else the F32 tiled kernel, with the global work
size rounded up to a multiple of TS and a `{TS, TS}` (or `{TS, TS, 1}`) local size
(`mg-opencl.cpp:437-458`):

```cpp
if (w->type == Type::Q8_0 || w->type == Type::Q4_K) {
    const int TS = 16;
    cl_kernel kr = I.k(w->type == Type::Q8_0 ? "k_mul_mat_q8_t" : "k_mul_mat_q4k_t");
    ... auto up = [&](int v){ return (size_t)((v + TS - 1) / TS * TS); };
    size_t gws[2] = {up(M), up(N)}, lws[2] = {(size_t)TS, (size_t)TS};
    clEnqueueNDRangeKernel(I.q, kr, 2, nullptr, gws, lws, ...);
    break;
}
cl_kernel kr = I.k("k_mul_mat_t");           // F32 batched/strided (attention, conv-matmul)
... size_t gws[3] = {up(M), up(N), (size_t)(B2*B3)}, lws[3] = {(size_t)TS, (size_t)TS, 1};
```

### 12.4 Measured results

Transformer forward (one full 24-layer pass), naive → tiled, cosine vs the PyTorch
oracle unchanged (README.md:286-290):

| Transformer forward | M1 Max GPU | Mali-G715 GPU | cosine |
|---|---|---|---|
| F32 | 2.8 → **0.38 s** | 18.7 → **6.7 s** | 1.0000000 |
| ggml Q8_0 | 0.54 → **0.31 s** | 14.3 → **4.8 s** | 0.99999979 |
| ggml Q4_K | 0.55 → **0.41 s** | 9.5 → **4.5 s** | 0.99995951 |

End-to-end on the Mali-G715 phone (class-id → PNG, 8 steps + VQGAN) dropped
**~3×** with tiling: **Q8_0 111 → 36 s, F32 347 → 36 s** (README.md:50-52, 304-305).
On device all three precisions converge at ~36 s: once the FC is tiled, the run is
bounded by the F32 attention matmuls and the **still-direct VQGAN conv** kernel
(§12.5), not the FC quant level.

**Honest framing.** On the **host**, the tiled GPU path is now *competitive with the
XNNPACK CPU* — host Q8_0 transformer 0.31 s, end-to-end gq8 4.5 s vs XNNPACK int8
3.9 s. On **device**, the Tensor G4 CPU (XNNPACK + KleidiAI **i8mm** int8
microkernels) is still **~8× faster** than the Mali GPU for this workload: M = 257 is
a *small* matmul dimension, and GPUs favor large batched GEMMs where there is enough
parallelism to hide memory latency. The GPU backend is the right architecture for
larger workloads and a path to offloading the CPU, but for this 257-token model the
optimized CPU int8 microkernels win on the phone today.

### 12.5 Stage 4 — 2D register micro-tiling (quantized FC)

The 1-output tiled kernel (§12.3) does 1 FMA per 2 local-memory loads — low
arithmetic intensity, so the inner loop is local-load-bound. **Stage 4** adds a
second level of blocking on top of the local-memory staging: each work-item computes
a `WPTM × WPTN` (4×4) *micro-tile* of outputs (`k_mul_mat_q8_t2` / `k_mul_mat_q4k_t2`,
generated from one shared `K_GEMM2D_BODY` macro). A 64×64 output tile is owned by a
16×16 workgroup; each step loads a 16-deep K-slab of both operands into local memory
(weights dequantized once on load), then each work-item pulls `WPTM` A-values and
`WPTN` B-values into registers and does `WPTM*WPTN = 16` FMAs:

```c
for (int k = 0; k < TSK; k++) {
    float areg[WPTM], breg[WPTN];
    for (int wm = 0; wm < WPTM; wm++) areg[wm] = Asub[k][tidm + wm*RTSM];
    for (int wn = 0; wn < WPTN; wn++) breg[wn] = Bsub[k][tidn + wn*RTSN];
    for (int wm = 0; wm < WPTM; wm++)            // 16 FMAs from 8 register loads
        for (int wn = 0; wn < WPTN; wn++) acc[wm][wn] += areg[wm]*breg[wn];
}
```

That is 8 loads per 16 FMA (~4× the intensity of the 1-output kernel). Result
(cosine-identical): transformer forward host Q8_0 0.40 → **0.28 s**, Q4_K 0.41 →
**0.30 s**; Mali Q8_0 4.8 → **4.3 s**, Q4_K 4.5 → **3.9 s**; end-to-end host gq8
4.5 → **3.5 s** — *now faster than the XNNPACK CPU (3.9 s)* — and device gq8
36 → **~28 s**. The Mali gain is smaller than the host gain because 16 accumulators +
operand registers per work-item raise register pressure and cap Mali occupancy; the
F32/attention path still uses the 1-output `k_mul_mat_t`, so the F32 model is
unchanged (~36 s device).

### 12.6 Memory: non-zeroing arena & the buffer cache

Two memory facts surfaced while profiling. **(1)** `Context` (§10) allocated its bump
arena with `std::vector::resize`, which *zero-fills* every byte — so the whole
over-provisioned arena (1.5 GB transformer + 3 GB VQGAN) became resident up front and
peak RSS was ~4.5–5 GB regardless of how little was touched. Switching to a raw
`new uint8_t[]` (default-init, no zeroing) lets pages stay unmapped until first
written; because the GPU does the compute, the CPU barely touches the arena and host
RSS fell to **~30 MB**. On Android the GPU weight/activation buffers themselves count
in process RSS, so device RSS scales with model size (F32 2.9 GB, Q8_0 2.1 GB, Q4_K
1.9 GB) rather than the spurious 5 GB. **(2)** The device buffer cache (§12.1) is
*size-checked*: a cached `cl_mem` is reused across `compute()` calls only if its size
still matches the tensor, else recreated — a robust fix for the iterative-decode
stale-buffer crash. Computed-node buffers are freed after each readback so peak RSS
stays at the single-step working set; measuring with vs without that free showed
**no device latency change** (~730 MB RSS difference), confirming Mali here is
compute-bound on the kernels, not buffer-allocation-bound.

### 12.7 fp16 on Mali — tried (Q8_0 only)

Half-precision variants of the 2D-tiled matmul (`k_mul_mat_q8_t2_h` /
`k_mul_mat_q4k_t2_h`) store the local slabs as `half` and run the micro-tile multiply
on Mali's 2×-rate fp16 ALU, accumulating in `float` so the K-length sum stays
accurate. They are gated on `cl_khr_fp16` (`#ifdef cl_khr_fp16` in the kernel source +
a runtime `CL_DEVICE_EXTENSIONS` check) so the program still builds and the host (M1
OpenCL has no fp16) stays on the fp32 kernels. Findings, cosine bit-identical
(`max_abs_diff` 6.37e-3 → 6.40e-3):

- **Q8_0**: isolated FC forward on Mali **4.3 → 3.7 s (~14 %)**. Q8_0's dequant
  (`scale·int8`) is cheap, so the kernel is ALU/local-mem-bound — exactly where fp16
  helps. Kept (Q8_0 + fp16-capable device only).
- **Q4_K**: *slower* with fp16 (3.9 → 4.2 s). Its kernel is **dequant-bound** —
  unpacking the 6-bit super-block scales dominates — so the extra `float → half` cast
  is pure overhead. fp16 is therefore disabled for Q4_K.
- **End-to-end device unchanged (~28 s):** the quantized FC is no longer the
  bottleneck. Once it's tiled + micro-tiled + (for Q8_0) fp16, the wall-clock is
  dominated by the **F32 attention path, the direct VQGAN conv, and thermal behaviour**
  on sustained runs. So the next real end-to-end lever is fp16/tiling *those*, not the
  FC. This is the key lesson: optimize the actual bottleneck, and re-profile after each
  win because the bottleneck moves.

### 12.8 Further optimization (remaining levers)

The matmul/conv kernels here are the *current* state; the M6 hill-climb (§13) layered
the tiled conv, fusion and int8-dot on top. Done since this section was first written:
the **VQGAN conv is now tiled** (implicit-GEMM `k_conv2d_t`, §13 step #1) and the
**quantized FC uses an int8 dot-product path** (`k_mul_mat_q8_i8` via `arm_dot_acc`,
§13 step #6). Remaining levers, ranked by the current device profile (~50% transformer
/ ~48% VQGAN after int8-dot):

- **int8 the VQGAN conv.** The conv is tiled but still F32. Quantizing its weights +
  activations to int8 and using `arm_dot_acc` (as the FC does) is the analogous win and
  the largest remaining *compute* lever (~48% of device time).
- **Cold-start weight upload (~3.8 s, one-time).** A cold single image is ~16 s vs the
  warmed ~13 s; the gap is the ~298 MB Q8_0 weight upload + first-run JIT. `cl_arm_import_memory`
  (which Mali exposes) could **zero-copy** the mmap'd weights into device buffers and
  eliminate most of it; amortized away in a persistent (JNI) process.
- **fp16 / int8 the F32 attention path** (`k_mul_mat_t`) — now only ~6% of device time,
  so low priority.
- **`cl_arm_matrix_multiply`** — Mali exposes a GPU matrix-multiply primitive that could
  go beyond `arm_dot_acc` for the FC.
- **Smaller GPU footprint** — free the mmap'd weights after upload (duplicated on GPU)
  and right-size the reference arenas.

## 13. Performance optimization: methodology & journey

§12 explains *what* the GPU kernels are. This section is the *methodology* and the
*journey*: how we decided which kernel to touch next, what each attempt actually
bought us, and — just as usefully — what didn't work and why. It is the story of the
M6 hill-climb. The full FLOP/byte derivations and the raw profiles live in
[`benchmark/analysis.md`](../benchmark/analysis.md); this section synthesizes them
into a narrative a reader can learn the *process* from.

All numbers below are class 207, seed 42, 8 steps, 256×256, on the two reference
machines: an Apple M1 Max (host) and a Pixel 9 / Mali-G715 (the deployment target).

### 13.1 The optimization loop

Every M6 step ran the same loop, and the order is the point:

1. **Measure.** The benchmark is built into the runtime as a mode:
   `mg-generate … --backend opencl --bench --n-runs N --warmup K`. It reports load
   time, end-to-end latency percentiles (p50/p90/p99, mean±sd), a **per-component**
   breakdown (transformer forwards / host sampling / VQGAN decode), peak RSS, and —
   for OpenCL — a **per-op-type GPU profile**. That last pass is `clFinish`-serialized,
   so its *absolute* total inflates versus the real overlapped run, but the *relative*
   split between ops is reliable; that split is what we steer by.
2. **Roofline to find the ceiling.** Compute the ideal time for each stage from FLOPs
   and bandwidth (§13.2). This tells us *how much room there is* before we spend effort,
   and whether a kernel is compute- or memory-bound — which dictates the right fix.
3. **Per-op profile to find the dominant op.** The roofline says the stage is slow; the
   per-op split says *which kernel* to open. We never optimize a kernel that isn't at
   the top of the profile.
4. **Optimize the one dominant op.**
5. **Validate correctness.** Cosine similarity of the final logits / image against the
   PyTorch oracle (§9). A speedup that moves cosine is a bug, not a win.
6. **Re-measure, then re-profile.** The bottleneck *moves* after every win — so the
   profile from step 3 is stale the moment the optimization lands. Re-rank before
   choosing the next target.

The discipline this enforces is worth stating plainly: **never optimize without a
profile, and re-profile after every win.** Two M6 missteps (an earlier guess that the
attention matmul was the bottleneck; the register-pressure hypothesis for the FC tile)
were *both* corrected by the profiler, not by reasoning — see §13.4.

### 13.2 Roofline as the north star

A roofline bounds achievable speed by the smaller of two ceilings — peak compute
`P` and peak bandwidth `B` — as a function of a kernel's **arithmetic intensity**
`I = FLOP/byte`. Below the ridge point `I* = P/B` a kernel is memory-bound; above it,
compute-bound. For the Mali-G715 (order-of-magnitude: ~2.6 TFLOP/s fp16, ~60 GB/s)
the ridge is `I* ≈ 43 FLOP/byte`. Applied to MaskGIT-256:

| stage | FLOP | ideal | measured (Mali, pre-M6) | efficiency |
|---|--:|--:|--:|--:|
| Transformer ×8 | ~890 GFLOP | **~0.34 s** (fp16) | 15.0 s | **~2%** |
| VQGAN decode | ~188 GFLOP | **~0.14 s** (fp32) | 6.3 s | **~2%** |

Both stages are compute-bound (the Q8_0 FC intensity is ~97 FLOP/byte even with a 5×
weight re-read), and the *whole* pipeline *could* run in well under a second on this
GPU. We achieved ~2%. The headline conclusion — robust to the 2× uncertainty in the
vendor-undocumented specs, since the gap is 60–100× — is that the GPU↔CPU gap on device
is **a kernel-efficiency / software-maturity gap, not a hardware gap.** For contrast,
XNNPACK int8 on the same CPU runs at ~25% of *its* roofline: same silicon, ~10× the
software maturity (fused, native-int8 micro-kernels).

That single number, ~2%, is what kept the work honest. It said the levers were
**fusion** and a **native int8 datapath**, not more tile tuning — and the hill-climb
below bears that out. The full FLOP counts (the transformer is ~111 GFLOP/forward,
~24× the design-doc estimate, dominated by the FC projections; VQGAN's 188 GFLOP is
dominated by the high-res `R=256` tail) and the per-cause attribution of the 2% are
derived in [`benchmark/analysis.md`](../benchmark/analysis.md).

### 13.3 The hill-climb, step by step

The starting point for M6 was the §12 backend: a tiled, micro-tiled, fp16 (Q8_0) FC
matmul, but a **naive direct VQGAN conv** and unfused periphery. The pre-M6 device
profile (Mali gq8) ranked: **MulMat(q) 44%**, **Conv2D 30%**, GroupNorm 6%, Add 5%,
MulMat(f32) 5%, Norm 4%, SoftMax 3%. We worked that ranking top-down.

(Steps 1–2 below were the §12 matmul work that preceded the M6 conv/fusion/int8
sequence; they are summarized here only to make the "the bottleneck moves" arc
complete — §12.3–12.7 has the kernels.)

**Step 1 — naive one-kernel-per-node → tiled local-memory GEMM.** The first
quantized matmul read each weight row and activation column once *per output element* —
memory-bound, low intensity. A 16×16 `__local`-memory tiled GEMM cut each operand's
DRAM reads ~16×. *Lesson: the textbook fix for a memory-bound kernel is data reuse via
on-chip memory; it won on both host and Mali, retiring the per-device kernel
selection.* Mali transformer forward Q8_0 14.3 → 4.8 s.

**Step 2 — 2D register micro-tiling.** Each work-item computes a 4×4 micro-tile (8
register loads → 16 FMAs), raising intensity ~4× on top of the local staging. *Lesson:
register-rich desktop GPUs love this (host Q8_0 0.40 → 0.28 s); Mali, register-poorer,
gains less (4.8 → 4.3 s) — the same optimization is device-dependent.*

**Step 3 (M6 #1) — tiled implicit-GEMM VQGAN conv.** With the FC tiled, the profile's
#2 was the conv at 30%, *still the naive one-thread-per-output-pixel kernel*. We
replaced it with a tiled local-memory conv structured as an implicit GEMM —
`out[oc,p] = Σ_k ker[oc,k]·col[k,p]`, with the im2col column **gathered on the fly**
into local memory (a materialized im2col buffer would be ~300 MB at 256×256). *Lesson:
the highest-value target is the biggest *un-optimized* op, not the biggest op overall.*

| Step 3 | host | device (Mali gq8) | cosine |
|---|---|---|---|
| VQGAN decode | 1.77 → **1.29 s** | 11 → **6.3 s** | 1.0 |
| Conv2D share | 23% → **12%** (885→416 ms) | 30% → **18%** (9.1→4.4 s) | |
| end-to-end gq8 | 2.84 → **2.36 s** | 26.2 → **21.5 s** | |

**Step 4 (M6 #2) — FC tile-autotune (a negative result worth recording).** The
re-ranked profile now had MulMat(q) at **52%**. The hypothesis was that the 4×4
micro-tile (16 accumulators) was register-pressure-limited on Mali, so we made the
tile build-time tunable (`-DGEMM_WPTM/N`, env `MG_GEMM_WPTM/N`, one source of truth
shared by kernel and dispatch) and swept 7 configs on Mali:

| tile | 4×4 | 6×6 | 4×8 | 2×4 | 8×4 | 4×2 | 8×8 |
|---|--:|--:|--:|--:|--:|--:|--:|
| Mali gq8 forward (s) | **3.73** | 3.82 | 3.84 | 3.87 | 3.91 | 4.18 | 5.62 |

4×4 was *already* optimal. *Lesson: the register-pressure hypothesis was wrong — reuse/
intensity wins over occupancy here, and tile geometry is exhausted. The remaining
MulMat(q) gap to the CPU is **structural** (GPU dequant→float-FMA vs the CPU's native
i8mm/SMMLA int8 matmul), not a tuning issue.* The tunable tile stays as reusable
autotuning infra; the profiler turned a plausible idea into a documented dead end
before it cost more than a sweep.

**Step 5 (M6 #3) — matmul-epilogue fusion (correct, but ≈wash).** Fuse bias-add +
GELU/SiLU + residual-add into the matmul kernel (`mul_mat_ex`), removing them as
separate graph nodes (~1550 `Add` + all `Gelu` launches per run). Correct — cosine
unchanged, golden retriever intact — and architecturally cleaner. But:

| Step 5 | device (Mali gq8) |
|---|---|
| `Add` | 1655 → 538 ms |
| `Gelu` | removed |
| `MulMat(q)` | 13.3 → 14.2 s (epilogue now runs *inside* it) |
| net | ≈ −0.4 s (~2%), within thermal noise |

*Lesson: exactly what the roofline predicted. Fusion relocates the cheap ~15%
periphery; it does not touch MulMat(q)'s **dequant compute** (56% of device time), so
it can't move the needle. Kept for the cleaner graph; not a perf win.*

**Step 6 (M6 #4) — int8-dot matmul (the big win).** This is the lever the roofline had
been pointing at all along: stop dequantizing to float. Quantize the activation to
int8 and use Mali's `cl_arm_integer_dot_product_accumulate_int8` (`arm_dot_acc`, 4 int8
MACs/op — the GPU analog of i8mm) in a tiled `k_mul_mat_q8_i8`, fed by a new
`k_quantize_q8`. Cosine 0.99999929; Q8_0 only (Q4_K/F32 keep the dequant path;
off-by-default escape hatch `MG_NO_ARM_DOT=1`).

| Step 6 | device (Mali gq8) |
|---|---|
| transformer (×8 loop) | 16.1 → **6.4 s** (2.5×) |
| end-to-end gq8 | 22.4 → **12.8 s** (1.75×) |
| single cold forward | 3.5 → 2.9 s (only ~14%) |

The single-forward gain is modest; the sustained-loop gain is 2.5×. That gap is the
subject of §13.4. After this step the device cost splits **~50/50** between the
transformer and the (already-tiled) VQGAN conv.

End-to-end, M6 took device gq8 from **26 s → ~13 s** (~2×), with the int8-dot path
doing most of the work.

#### Step 7 — int8 the VQGAN conv (analysis: the 9× gap, and the accuracy wall)

After Step 6 the profile re-ranked to ~50/50 transformer/VQGAN, so the question was:
keep pushing the transformer, or attack the VQGAN? A **per-component CPU comparison**
settled it — the gap to XNNPACK is not uniform:

| component | GPU | CPU (XNNPACK) | gap |
|---|--:|--:|--:|
| transformer | 5.8 s | ~3 s | 2× |
| **VQGAN** | **6.2 s** | **~0.7 s** | **9×** |

The transformer is "only" 2× off the CPU (the small-M=257 + maturity gap — it's at ~5%
of roofline and hard to move). The **VQGAN was the real anomaly (9×)**: the CPU runs its
conv in **int8**, our GPU conv was still **F32**. So the same int8-dot lever was untapped
for the conv. Applied it (`k_conv2d_i8`: pre-quantized int8 weights + implicit-GEMM
gather of the im2col column as int8 + `arm_dot_acc`):

| Step 7 (matched-thermal bench) | int8 conv | F32 conv |
|---|--:|--:|
| Conv2D | 1.56 s | 4.38 s |
| end-to-end gq8 | **10.1 s** | 12.8 s (−21%) |

…but it hit an **accuracy wall**: a *per-tensor* activation scale (one `d_in` for the
whole feature map) drops VQGAN cosine **1.0 → 0.9984** (visible image degradation),
because a 32-wide K-block of the im2col column mixes channels and input pixels with very
different magnitudes, so a single scale clips them. (0.9984 is, notably, ~the same tier
as the shipped XNNPACK *int8* conv at 0.9994 — int8-conv quality, not a bug.) The FC int8
path avoided this because it quantizes the activation *per-32-block*, which fits the GEMM
K-blocking exactly; the conv's implicit im2col makes per-block activation scales much
harder (they must be computed gather-time). So Step 7 is **kept but off by default**
(`MG_ARM_CONV=1` opts in) — a deliberate speed↔quality choice, with per-block/per-column
activation quant logged as the way to make it default-safe.

A second debug note worth recording: a *cold single run* first showed the int8 conv
**slower** end-to-end (14.6 vs 12.8 s); only the matched-thermal bench A/B revealed the
21% win. Same trap as Step 6 (§13.4(b)).

A follow-up tightened the activation quant to **per-(pixel, 32-block) gather-time** (the
conv kernel reads F32 input, gathers the column into local memory, and each pixel's
32-element K-block gets its own scale — matching the FC's per-block scheme). That removed
the separate amax/quantize pre-passes AND recovered cosine to **0.99997**, so the int8
conv is now **default-on**. Matched-thermal: end-to-end 12.8 → **9.7 s** with the FC
already on int8.

#### Step 8 — workgroup-parallel reductions (Norm / SoftMax / GroupNorm)

After Step 7 the per-op re-profile turned up an embarrassing anti-pattern. The
reductions were still **one-thread-per-row** sequential code: `k_norm` had one work-item
sequentially read `D=768` elements three times for mean / variance / affine, and
`k_group_norm` had one thread sequentially handle up to **~262 k elements** at the
largest VQGAN resolution. Together Norm + SoftMax + GroupNorm were **22%** of device
time. Rewriting all three as workgroup-parallel (a 64- or 256-thread workgroup per row,
stride-loop + local-memory tree reduction) was the next, very cheap, big win:

| op | naive (1 thread/row) | parallel | speedup |
|---|--:|--:|--:|
| GroupNorm | 1713 ms | **189 ms** | 9.1× |
| Norm | 1409 ms | **337 ms** | 4.2× |
| SoftMax | 1025 ms | **385 ms** | 2.7× |
| VQGAN decode | 3.1 s | **1.66 s** | −46% |
| **end-to-end gq8** | **9.79 s** | **7.08 s** | **−28%** |

cosine bit-identical (transformer 0.99999979, VQGAN 1.0). The lesson — *re-profile after
every win and look for naive patterns in whichever op is suddenly #1* — is the same one
behind §13.4(a). After Step 8 the FC matmul is back to ~51% and `MulMat(f32)` (the F32
attention matmul) is the new ~18% #2.

#### Step 9 — F32 attention matmul: two negative-result experiments

After Step 8 the new #2 op was the F32 attention matmul (`k_mul_mat_t`, ~18% / 1.9 s).
The attention shape is small (per-head Q·Kᵀ has K=48, M=N=257; A·V has K=257, M=48, N=257),
batched 16 heads in one dispatch. We tried two cheap optimizations:

| attempt | hypothesis | result |
|---|---|---|
| **fp16 attention** (`k_mul_mat_t_h`: F32 → half on load, fp16 multiply, fp32 accumulate) | Mali fp16 ALU is ~2× — should halve compute | **Regressed**: MulMat(f32) 1928 → 2532 ms (+31%), end-to-end 7.08 → 7.42 s |
| **TSK 16 → 32** (~halves barriers; K=48 from 6 to 4, K=257 from 34 to 18) | Attention may be barrier-bound | Within noise: 7.03 → 7.23 s, MulMat(f32) effectively unchanged |

Both kept the cosine bit-identical (transformer 0.99999930). The combined finding is
that the attention matmul is **neither ALU-bound nor barrier-bound** at this scale —
it's overhead-bound (kernel-launch + strided global loads on small per-head shapes),
which neither lever attacks. To go further would need a *structural* rewrite — flash-
attention-style fusion of QKᵀ → softmax → A·V into one kernel that keeps the score
matrix in local memory, or a fundamentally different tile geometry. Both are big.
`k_mul_mat_t_h` is left in the source as documented infrastructure, gated off.

*Lesson*: when a tuning lever that "should" help doesn't, the bottleneck isn't where the
lever attacks. Each negative result is information that re-ranks the cost model — here,
*overhead is the wall, not ALU or barriers* — and tells you to stop tuning and either
restructure or move to a different op.

#### Probe note: `cl_arm_matrix_multiply` is *not* a wider primitive on Mali-G715

Before Step 8 I expected `cl_arm_matrix_multiply` (which Mali advertises) to be the next
big lever — a wider matrix-multiply instruction, analog of the CPU's `SMMLA` doing 64
MACs per call instead of `arm_dot_acc`'s 4. The headers don't declare it (it's a Mali
compiler built-in), so I probed by compiling small kernels with various candidate
signatures and reading the compiler's "no matching function" notes for the expected
overloads. The Mali compiler reported exactly three overloads:

```
int   arm_matrix_multiply(char4,  char4,  int)
uint  arm_matrix_multiply(uchar4, uchar4, uint)
float arm_matrix_multiply(float,  float,  float)
```

…which are **the same 4-MAC int8 dot product as `arm_dot_acc`**. So on this Mali there is
no wider int8 matrix primitive; the matmul is already at the hardware ceiling per-
instruction. *Lesson: probe an extension's actual built-ins before designing around an
assumed signature; ARM publishes the spec but vendors may ship a subset.*

**Step 10 (M6 #8) — Flash-attention: the M=large flip.** The fp16/TSK probes in Step 9
told us attention was overhead-bound at M=257 and that the only way forward was
fusion. The seq-len sweep we built after M6 (random-weight synthetic GGUFs at
M ∈ {65, 257, 1025, 4097}) made the cost of *not* fusing concrete: with the naive
`MulMat(f32)+SoftMax+MulMat` chain the GPU stayed ~1.7× slower than the CPU all the
way through M=1025, and OOM'd at M=4097 because the M²·heads·layers·4 B scores arena
ballooned to ~79 GB. So we wrote tiled flash-attention-v2 — one workgroup per
(Q tile, head, batch), K/V tiled into local memory, online softmax statistics in
registers, scores never touch DRAM. The result, on Pixel 9:

| | M=65 | M=257 | M=1025 | M=4097 |
|---|---:|---:|---:|---:|
| GPU baseline (naive attn) | 1 733 | 5 157 | 38 693 | OOM |
| GPU flash-attn            | 1 697 | 4 882 | 23 405 | 147 628 |
| Speedup (GPU FA / GPU base) | 1.02× | 1.06× | **1.65×** | (infeasible→runs) |
| Device CPU (XNNPACK Q8)   | 773  | 2 985 | 22 198 | 319 657 |
| **GPU FA / CPU**          | 2.20× | 1.64× | **1.05× (tied)** | **0.46× (GPU 2.17× faster)** |

Crossover lands at M ≈ 1025; at M=4097 the GPU decisively beats the CPU. **Numerical
correctness verified** end-to-end via `verify-opencl-transformer` — cosine
0.99999979 vs. PyTorch oracle, identical to the unfused chain. The attention block
at M=1025 went from 32 056 ms (MulMat+SoftMax) to **6 467 ms** (single FlashAttn op)
— a 5× block-level reduction. See **§13.6** for the algorithm + kernel walk-through.

*Lesson*: the M6 #7 negative result — *"attention is overhead-bound, fusion is the
only lever"* — was correct, and Step 10 cashed in on it. **The right kernel
restructuring is worth more than any number of in-kernel tuning passes.** The same
pattern (memory-tiled attention) is what LiteRT-LM uses to claim GPU >> CPU on
similar workloads.

### 13.4 Two cross-cutting lessons

These are the most transferable takeaways — they matter more than any single kernel.

**(a) The bottleneck moves; re-rank after every win.** The per-op ranking re-shuffled
at every step, and chasing a stale ranking would have wasted effort:

| op (Mali gq8 share) | pre-M6 | after Step 3 (tiled conv) | after Step 6 (int8-dot) |
|---|--:|--:|--:|
| MulMat(q) | 44% | **52%** (rose — others shrank) | now ~half of a much smaller transformer |
| Conv2D | 30% | **18%** (4.4 s) | — |
| component split (T/VQGAN) | 57 / 42 | — | **~50 / 50** |

Note how MulMat(q) *grew* as a share after the conv win (it was unchanged in absolute
terms; the denominator shrank), which correctly re-pointed us at the FC — first to the
autotune dead-end (Step 4), then to the int8-dot win (Step 6) once tuning was ruled
out. Each arrow in that table is a decision the profiler made for us.

**(b) Device measurement is thermally noisy; A/B with matched thermal state.** The
int8-dot path (Step 6) first looked *slower*: a single cold-device forward measured
**4.1 s vs 3.6 s** for the fp16 path. Only a matched-thermal, multi-run A/B revealed
the true ranking — int8 was **2.5× faster over the sustained loop**. The mechanism is
power: the int8 datapath draws less, so over the 8-step decode it avoids the **thermal
throttling** that crushes the fp16 path. That is also why the per-forward gain (~14%)
and the sustained-loop gain (2.5×) diverge so far — the win *compounds* as the device
stays cool. *Lesson: on a phone, trust neither a single shot nor a cold run; warm up,
run N times, and compare paths under the same thermal conditions. Single-shot device
timings are unreliable.*

### 13.5 What's left

The device CPU (XNNPACK + KleidiAI **i8mm** int8 micro-kernels) is still ~3× ahead of
the Mali GPU on this 257-token model — small-M GEMMs favor the CPU's big caches and few
strong cores, and KleidiAI is a mature fused int8 library.

**Decomposing a cold single image (gq8, Mali ~16 s)** — by sweeping the step count
(cold(1)=10.7, cold(4)=12.7, cold(8)=15.8 s) we can separate the fixed cost from the
per-forward cost:

| part | time | note |
|---|--:|---|
| one-time weight upload + JIT | ~3.8 s | per process; amortized in a service, hidden by the bench's warmup. `cl_arm_import_memory` could zero-copy the mmap'd weights and remove most of it. |
| transformer compute | ~5.8 s | 8 × **0.73 s/forward** — still ~5% of roofline (small-M) |
| VQGAN decode | ~6.2 s | F32 conv (int8-conv opt-in, §13.3 Step 7) |

So the transformer per-forward is 0.73 s (≈18× its ~40 ms roofline): real headroom, but
it's the small-M + maturity gap, not a bug. Remaining levers, descending value:

- **int8 the VQGAN conv accuracy-safely** — Step 7 gives 21% end-to-end but at cosine
  0.9984; per-block/per-column activation quant (gather-time) would recover the quality
  and let it ship by default.
- **The attention path is still F32** (`k_mul_mat_t`, now only ~6%) — fp16/int8 there is
  a smaller transformer win.
- **Cold-start**: `cl_arm_import_memory` zero-copy weight upload (~3.8 s for single-image
  runs).
- **`cl_arm_matrix_multiply`** (a native matrix extension) for the FC, if Mali support is
  reliable.

The roofline still says most of the wall is on the table; M6 closed the easy half, and
the rest is the same native-int8-and-better-quantization story.

### 13.6 Flash-attention: the algorithm and our OpenCL kernel

This section walks through *why* flash-attention works and *how* the OpenCL kernel
in `src/mg-opencl/mg-opencl.cpp` (`k_flash_attention`) is structured. It's a
self-contained reference for anyone reading this code without prior FA background.

#### The problem with naive attention on a GPU

The textbook scaled-dot-product attention is three ops:

```
scores  = Q · Kᵀ * scale            # shape: [H, S_q, S_k]   (the "M²" tensor)
probs   = softmax(scores, dim=-1)   # shape: [H, S_q, S_k]
out     = probs · V                 # shape: [H, S_q, D]
```

At MaskGIT M=257, scores per head per layer = 257² × 4 B = 264 KB. Across 16 heads,
24 layers, 8 steps that's ~810 MB written, ~1.6 GB read (softmax + matmul both
re-read it) — call it **2.4 GB of DRAM traffic per generate just for the scores
tensor**. At M=1025 that's 16× worse — **38 GB**, which is more than the entire
model's weights. The Mali GPU's compute ALU sits idle waiting on those reads.

Flash-attention's key insight is that **you never need scores in main memory at
all**. The information you carry between V-rows is only two scalars per Q-row
(running max `m` and sum `l`), plus the running output accumulator `O`. Everything
else can stay in the kernel's local memory and registers.

#### The online-softmax trick

Standard softmax requires two passes over the row: one to find the max (for numeric
stability), one to compute `exp(x - max)` and sum. That's why naive attention
materializes scores — you need to look at every score twice.

The online version processes scores in **tiles** of `BC` columns at a time. After
processing tile `j`, it carries:

- `m_i` — max of all scores seen so far in this row (a single scalar)
- `l_i` — sum of `exp(score - m_i)` for all scores seen so far (a single scalar)
- `O_i` — partial output row, shape `[D]` (held in registers)

When the next tile arrives:

```
m_new = max(m_i, max(scores_in_tile))               # update running max
α     = exp(m_i - m_new)                            # rescale factor for OLD accumulator
l_new = α · l_i + Σ exp(scores_in_tile - m_new)
O_new = α · O_i + (P_tile · V_tile)                 # P_tile = exp(scores - m_new)
m_i, l_i, O_i  =  m_new, l_new, O_new
```

The `α · O_i` term is the magic — it retroactively corrects the running output for
the fact that the max changed. After the last tile, `O_i / l_i` is exact softmax-V.

The proof: at every step, `O_i = Σ_t (e^(score_t - m_i)) · V_t`. When `m_i`
increases to `m_new`, every old term needs to be multiplied by `e^(m_i - m_new)`
to stay consistent — that's `α`. Then the new tile's contribution is added with
the new max. The denominator `l_i` is accumulated the same way.

#### Our kernel layout

Inputs are contiguous `{D, S, H, B}` (head-dim innermost, then seq, then head,
then batch — `cont()`'d from the permuted Q/K/V views in the transformer builder).
With MaskGIT's `D = 48`, we hardcode:

```c
#define FA_D  48   // head_dim — fixed for MaskGIT
#define FA_BR 32   // Q tile rows per workgroup
#define FA_BC 32   // K/V tile cols per inner loop iteration
```

**Workgroup grid:** `(ceil(S/BR), H, B)`. One workgroup per (Q tile, head, batch).
At M=257 that's `9 × 16 × 1 = 144` workgroups — comfortable for Mali's 7 shader
cores to fill.

**Per-workgroup state** (`FA_BR = 32` threads, one per Q row in the tile):

| storage | what | size |
|---|---|---|
| registers, per-thread | `q_row[FA_D]` (the thread's Q row) | 48 × 4 = 192 B |
| registers, per-thread | `o_row[FA_D]` (running output) | 192 B |
| registers, per-thread | `m_i, l_i` (running max + sum) | 8 B |
| registers, per-thread | `s_kc[FA_BC]` (scores for current K tile) | 128 B |
| `__local`, workgroup | `K_tile[FA_BC × FA_D]` | 32 × 48 × 4 = 6 KB |
| `__local`, workgroup | `V_tile[FA_BC × FA_D]` | 6 KB |

So ~520 B per-thread registers (fits in Mali's per-lane register file) and **12 KB
of local memory per workgroup** — well under Mali-G715's typical 32-64 KB limit.

**Inner loop** (for each K tile `kt = 0..ceil(S/BC)-1`):

1. **Collaborative load.** All `BR` threads cooperate to load the `BC × D = 1536`
   K and V values into local memory; each thread loads exactly `BC·D / BR = 48`
   elements (one strip per thread). Out-of-range rows get 0. `barrier(CLK_LOCAL_MEM_FENCE)`.
2. **Dot products.** Each active thread computes `BC = 32` scores by dotting its
   Q row (in registers) against each K row in `K_tile` (in local memory). The Q
   row is reused across all 32 dots — that's the locality we get for free by
   holding it in registers.
3. **Online softmax update.** Compute `m_new = max(m_i, max(s_kc))`, then
   `α = exp(m_i - m_new)`, then convert each `s_kc[kc]` to a probability
   `p = exp(s_kc[kc] - m_new)`, then `l_new = α · l_i + Σp`.
4. **Output accumulator update.** For each `d ∈ [0, D)`:
   `o_row[d] = α · o_row[d] + Σ_kc s_kc[kc] · V_tile[kc][d]`.
   This is a `BC = 32`-element dot product per output element — well-balanced
   per-thread work.
5. **Commit.** `m_i ← m_new; l_i ← l_new`. `barrier(CLK_LOCAL_MEM_FENCE)` before
   the next iteration's collaborative load can clobber `K_tile / V_tile`.

After the last tile, each active thread writes its normalized output:
`O[row, d] = o_row[d] / l_i`.

#### What this gets us

| metric | naive (M=1025) | flash-attn (M=1025) | ratio |
|---|--:|--:|--:|
| DRAM bytes per layer per step (attention block) | ~270 MB | ~10 MB | **27×** |
| DRAM bytes per generate run | ~52 GB | ~2 GB | **26×** |
| Mali attention block wall time | 32 056 ms | 6 467 ms | **5×** |
| End-to-end transformer (×8) | 40 416 ms | 23 405 ms | **1.73×** |
| Memory peak (transformer scratch arena) | ~5 GB | ~1 GB | — |

The DRAM-traffic reduction is what unlocked everything else: Mali's ALU was idle
waiting on the scores reads in the naive path; flash-attention turns it
compute-bound again, which is the regime the chip's FP32 throughput advantage
actually matters in.

#### Caveats / things we didn't do

- **FP16 K/V tiles (Step 11, 2026-06-04).** Added a fp16 variant
  `k_flash_attention_h` guarded by `cl_khr_fp16` (Mali yes, M1 OpenCL no).
  Q row + K/V tiles cast to fp16; scores, softmax stats (m_i, l_i), output
  accumulator stay fp32. Result: end-to-end M=257 −2%, M=1025 **−7%**,
  M=4097 **−22%**. Mali cosine vs PyTorch oracle measured directly via
  cross-compiled verify-opencl-transformer: **0.99999929** vs the fp32 naive
  chain's 0.99999930 — 10⁻⁸ difference, bit-equivalent at logit precision.
  Step 9's fp16 attention-matmul cast on M=257 *was* a regression — but
  that was attention as a *separate* matmul where it stayed overhead-bound.
  Inside flash-attention's compute-bound inner loop, the same lever pays off.
  Default on Mali; auto-falls back on devices without `cl_khr_fp16`.

- **Strided-input FA (Step 12, 2026-06-04).** Removed the 3 `cont()` ops per
  layer that materialized the permuted Q/K/V views into contiguous buffers
  before FA. The kernel now takes `(s0, s1, s2, s3)` element-stride params
  for the input layout and reads directly from the matmul → reshape → permute
  view. Graph nodes 610 → 538 (= 72 fewer cont ops). End-to-end win at M=1025
  −3% (FA op slightly slower because strided reads are marginally less
  cache-friendly, but cont elimination wins more); cleaner code (one kernel
  handles both contiguous and strided layouts). Cosine unchanged
  (0.99999929 Mali, 0.99999979 M1) — same logits at 8-decimal precision.
  *Lesson:* eliminating a 5%-of-profile op type can still come out to ~1-3%
  e2e because clFinish-serialized profiles inflate the apparent cost of
  cheap ops that would naturally overlap.

- **LN-affine fusion (Step 13, 2026-06-04).** The transformer's
  `layer_norm_affine` was a 3-op chain `Norm → Mul (×γ) → Add (+β)`. Each
  intermediate spilled the full 257·768·4 ≈ 790 KB activation back to DRAM.
  Added `k_norm_affine` that does the LN reductions + affine in a single
  pass, and a `norm_affine()` builder that sets `src[1]=γ, src[2]=β` on the
  Norm node. Graph nodes 538 → 438 (= 100 fewer ops: 50 Muls + 50 Adds from
  25 LNs × 2 ops). Op-count delta in the profile: Add 451→51, Mul 425→25.
  Per-M wins: M=257 −1.8%, M=1025 **−7.5%**, M=4097 −4.3%. The bigger gains
  at M=1025 reflect the larger DRAM traffic eliminated per LN. Cosine
  unchanged (still 0.99999929 Mali, 0.99999979 M1). *Lesson*: cheap
  element-wise ops that look small individually compound when they sit
  between heavier ops — fusion shifts them off the critical DRAM path
  entirely. With a regular pattern (this is the third "fuse N ops into the
  reduction that already loads the data" win in M6, after mul_mat_ex's
  bias/act/residual and the FA scores/softmax/V matmul).

- **`cl_arm_import_memory` zero-copy weight upload (Step 14, NEGATIVE, 2026-06-04).**
  Tried importing the mmap'd GGUF into a Mali `cl_mem` for zero-copy weight
  upload (saves ~3-5 s of `clEnqueueWriteBuffer` per process). The Mali-G715
  driver returns `CL_OUT_OF_HOST_MEMORY` (e=-6) for both MAP_PRIVATE and
  MAP_SHARED file-backed mmap regions — it only accepts `CL_MEM_ALLOC_HOST_PTR`-
  allocated buffers or dma_buf, both of which would require an upfront copy
  defeating the savings. The plumbing is kept (`OpenCLRuntime::import_host_region`,
  `Model::mmap_base()/mmap_size()`); off by default, opt-in via
  `MG_ARM_IMPORT=1` so future drivers / different devices that accept the
  import can pick it up. *Lesson*: vendor extensions advertised in
  `CL_DEVICE_EXTENSIONS` aren't a guarantee that the obvious use case works;
  probe with the actual memory you intend to import.

- **GroupNorm + affine + SiLU fusion in VQGAN (Step 15, 2026-06-04).** The
  decoder's `gn_affine` was a 3-op chain (`GroupNorm → Mul (×γ) → Add (+β)`)
  ALWAYS immediately followed by `swish` (= SiLU). Four ops per GN site,
  ~25 sites per VQGAN forward. Added `k_group_norm_affine_silu` that does
  the GN reductions + affine + SiLU in a single pass; `group_norm_affine_silu()`
  builder + a `gn_affine_silu()` helper that replaces the `gn_affine(...);
  swish(c, ...);` pattern in the decoder. Saves 75 dispatches per generate
  (25 Mul + 25 Add + 25 SiLU collapse into the reduction's apply phase).
  Result: VQGAN decode 1 671 → **1 366 ms (−18%)** on Pixel 9 Mali; end-to-end
  M=257 6 466 → **6 119 ms (−5.4%)**. Cosine: host M1 1.00000000 (bit-perfect),
  Mali 0.99997564 (unchanged — same as the int8 conv path it consumes).
  *Lesson*: the LN-affine pattern (step 13) transfers directly — same shape
  of "reduction with trailing element-wise affine + activation" appears in
  the VQGAN GN block, and the same fusion works.

- **Q4_K int8-dot matmul (Step 16, 2026-06-04).** Q4_K on Mali was stuck on
  the F32 dequant kernel (`k_mul_mat_q4k_t2`) — the int8-dot path (M6 #4 / §13.3
  Step 6) only applied to Q8_0. End-to-end gq4 on Pixel 9 was **16.0 s** vs.
  Q8_0's 6.1 s — a 2.6× gap from a model that's *smaller* (216 MB vs 298 MB)
  and should be *faster*. The fp16 dequant variant was tested first and
  regressed (−17% e2e at 18.6 s) — confirms the older finding that Q4_K is
  dequant-bound, not multiply-bound.
  Wrote `k_mul_mat_q4k_i8`: mirrors `k_mul_mat_q8_i8` but loads the Q4_K
  format inline — extracts 4-bit nibbles to int8 (range 0–15), decodes the
  per-sub-block (scale, min) from the 12-byte packed scales **once per K
  iteration** (cached in `__local`), then applies the contribution
  `scale · dx · int_dot − min · dx · sum_x` per sub-block. `sum_x` is
  computed inline over the loaded int8 activation slab. Dispatched whenever
  the device has `cl_arm_integer_dot_product_accumulate_int8`, same gate as
  the Q8_0 path. Result, M=257 production on Pixel 9 Mali:
  end-to-end **15 955 → 6 688 ms (−58%)**, transformer **14 459 → 5 170 ms
  (−64%)**, MulMat(q) **14 525 → 6 178 ms (−57%)**. Mali cosine **0.99995885**
  vs the previous F32-dequant path's 0.99995951 — a 6×10⁻⁵ delta from int8
  activation quantization, well within the per-block-scale quant error
  budget that was already accepted for Q8_0 int8-dot. The win brings Q4_K
  within 9% of Q8_0 latency despite the extra dequant complexity, **at 73%
  of the file size**. *Lesson*: when a high-leverage hardware path (here
  arm_dot_acc) was applied to one format and not another, generalizing it
  to the other format is usually worth the kernel work — and Q4_K's bit-
  twiddling overhead per super-block is amortized across the 256 weights
  it dequantizes, not multiplied.
- **No causal masking.** MaskGIT is bidirectional, so we don't need it. For a
  causal LLM you'd skip the upper-triangular K-tile portion.
- **No multi-query / GQA.** MaskGIT has full multi-head attention. For GQA you'd
  share K, V tiles across query heads — easy modification.
- **Head dim hardcoded to 48.** A small refactor (compile-time `-D` define or
  runtime parameter) would generalize. Not done for MaskGIT-only use.
- **Last partial tiles.** Handled by `if (k_row >= S) score = -INFINITY` so the
  softmax sees them as zero-probability, but the K/V load still wastes lanes.
  At M=257 the last tile has 1 active row out of 32 — bench it; not currently a
  bottleneck per the per-op profile.
- **Default-off (`MG_FLASH_ATTN=1`).** Waiting on one more pass of correctness +
  perf review at the standard M=257 production setting before flipping the
  default. The verify pass is clean (cosine 0.99999979) but we want a Quick-5
  IS/top-k regression check on actual generated images first.

#### Why the pattern transfers

Flash-attention is **the canonical pattern for any "M²" intermediate that's
larger than the dot-product accumulators**. Same recipe applies to:

- Top-k decoding (sort scores in local memory)
- Pairwise distance metrics
- Differentiable nearest neighbors
- Any custom kernel where you'd "materialize an outer product, then reduce".

The mental model: *every gradient between Q-row inputs and O-row outputs flows
through a vector of `D` floats. Carry that vector through the K-tile loop in
registers, and the M² intermediate becomes the kernel's private workspace, not
a global tensor*. Once you see this, the GPU's local memory becomes a much more
useful tool than just "for tiled GEMM" — it's the *general* way to fuse
quadratic intermediates into linear-memory kernels.


### 13.7 M6 closeout: the journey, the lessons

**Pixel 9 Tensor G4 / Mali-G715, end-to-end Q8_0:**

| stage | end-to-end | notes |
|---|---:|---|
| **pre-M6** (naive kernels) | **111 s** | each op a separate one-thread-per-output kernel |
| §12 Step 1 (tiled GEMM) | 14.3 s (transformer-only) | local-memory data reuse |
| §12 Step 2 (2D micro-tile) | 4.3 s (transformer-only) | register-level reuse |
| M6 #1 (tiled VQGAN conv) | 21.5 s | implicit GEMM + gather-time im2col |
| M6 #2 (FC autotune) | — | NEGATIVE — 4×4 already optimal |
| M6 #3 (mul_mat_ex fusion) | — | wash, clean refactor |
| M6 #4 (int8-dot for Q8_0) | 13 s | `arm_dot_acc`, the biggest single win |
| M6 #5 (int8 VQGAN conv) | 9.7 s | per-(pixel, 32-block) gather-time activation quant |
| M6 #6 (workgroup reductions) | 7.1 s | norm/softmax/groupnorm 9.1×/4.2×/2.7× |
| M6 #7 (fp16 attn matmul) | — | NEGATIVE — overhead-bound at M=257 |
| M6 #8a (flash-attention v2) | 6.7 s | M² scores never in DRAM |
| M6 #8b (fp16 K/V tiles) | 6.7 s @ M=257; M=1025 −7%; M=4097 −22% | Mali's 2× fp16 ALU rate |
| M6 #8c (strided FA) | 6.5 s | eliminated 3 cont()s/layer |
| M6 #8d (LN-affine fusion) | 6.5 s @ M=257; M=1025 −9% | 3 ops → 1 |
| M6 #8e (cl_arm_import_memory) | — | NEGATIVE — Mali rejects file mmap |
| M6 #8f (GN+affine+SiLU fusion) | **6.1 s** | 4 ops → 1 in VQGAN, decode −18% |
| M6 #8g (Q4_K int8-dot) | — | **gq4: 16 → 6.7 s** (separate quant) |

**Pixel 9 final state:**

| precision | size | latency | vs CPU |
|---|---:|---:|---|
| OpenCL Q8_0 | 298 MB | **6.1 s** | M=257: CPU 1.52× ahead; M=1025: GPU **16% faster**; M=4097: GPU **2.86× faster** |
| OpenCL Q4_K | 216 MB | **6.7 s** | M=257: CPU 1.61× ahead; M=1025: GPU **9% faster vs CPU Q8** / **36% vs CPU Q4**; M=4097: GPU **2.74× / 4.14×** |
| Host M1 Max Q8_0 (cross-check) | 298 MB | 1.3 s | benchmark anchor |

**Net: device Q8_0 end-to-end 111 s → 6.1 s = 18.2× speedup** with cosine ≥ 0.99997
(the int8 activation-quant noise budget already accepted at M6 #4) and **bit-perfect
host fp32 cosine** (1.00000000 on M1 OpenCL through every fusion step). 16 hill-climb
steps over the milestone, 12 wins, 4 documented negative results — the negatives
are where the bottleneck *would* have been if our cost model were wrong, so they
carry as much information as the wins.

#### What we learned

These transfer to any similar profile-guided GPU effort and are worth more than
any single kernel:

1. **The bottleneck moves; re-rank after every win.** At every step the per-op
   share shuffled — `MulMat(q)` was 44% pre-M6, dropped to 51% after the conv
   win (others shrank), then re-emerged as the right target for the int8-dot
   path. Chasing a stale ranking would have wasted weeks. The `--bench` per-op
   profile from M5 is the milestone's most-used artifact.

2. **Negative results redirect effort with as much signal as wins.** M6 #2
   (autotune), M6 #7 (fp16 attn matmul), M6 #8e (cl_arm_import_memory) all
   failed — and each told us something specific: the current tile size is
   already optimal, attention is overhead-bound not ALU-bound (which pointed
   directly at flash-attention), Mali rejects file-backed mmap (so we'd need
   pre-allocated buffers to win). Documented as steps, not silently rolled
   back, because the *next* engineer reading this needs the cost model, not
   just the winning code.

3. **Hardware ceilings have to be respected.** The chip exposes `arm_dot_acc`
   = 4 MACs/instruction for int8. Mali's `cl_arm_matrix_multiply` turned out
   to be the *same* primitive under a different name (M6 §13.3 probe). Beyond
   that there is nothing to tune. We hit that ceiling and stopped — anything
   else trying to widen the dot product would have been wasted code.

4. **Fusion is a recurring pattern, not a one-off.** Three independent fusion
   wins land on the same template — "reduction that's already loading the
   data, append the trailing element-wise ops":
   - `mul_mat_ex` (M6 #3): matmul + bias + activation + residual
   - `k_norm_affine` (M6 #8d): LN reductions + γ + β
   - `k_group_norm_affine_silu` (M6 #8f): GN reductions + γ + β + SiLU
   - `k_flash_attention` (M6 #8a): Q·Kᵀ + softmax + ·V
   Once you see the pattern, finding the next one is faster than the first.
   Each compresses N small dispatches into one, with the per-tensor DRAM
   round-trips collapsed into a single load/store.

5. **Generalize hardware-path wins across formats.** The int8-dot path
   (M6 #4) was Q8_0-only for a year. M6 #8g generalized it to Q4_K with a
   day of kernel work and saved 9 s end-to-end. **When a high-leverage
   hardware path applies to one data format and not another, generalizing
   it is usually worth the work.** Format-specific overhead (here, Q4_K's
   6-bit scale decoding) is amortized across many weights when you decode
   it once and cache; what you can't amortize is the missing int8 datapath.

6. **Mobile GPU memory is unified system RAM — DRAM-traffic optimizations
   dominate.** Mali shares LPDDR with the CPU. Every fusion above wins
   primarily by removing DRAM round-trips, not by saving compute. The
   single largest example: flash-attention removed ~52 GB of DRAM traffic
   at M=1025 across a generate, which is what flipped the GPU vs CPU
   comparison at long prefills. Compute throughput rarely binds first;
   memory traffic does.

7. **Device thermals are a real factor in benchmark interpretation.** The
   M6 #4 int8-dot win first *looked* slower on a single cold-shot run
   (4.1 s vs the fp16 path's 3.6 s) — only matched-thermal, multi-run A/B
   revealed the true 2.5× sustained win. Power matters: the int8 path
   draws less and stays cool, so over a sustained loop it avoids the
   throttling that crushed the fp16 path. *Single-shot device timings are
   unreliable* on phones. Always warmup-then-run.

8. **Mobile GPU vs CPU isn't a fixed truth — it's a function of M.** At
   the production M=257 the Cortex-X4 SMMLA stays 1.5× ahead of Mali via
   KleidiAI. But at M=1025 the GPU pulls even, and at M=4097 the GPU is
   2.86× faster (Q8_0) or 4.14× faster (Q4_K with our int8-dot path).
   The original M6 wrap text claimed "for this model on this chip the
   CPU is the right tool" — that's true *at the model's published
   resolution*. For any larger prefill workload — longer-context LLMs,
   higher-resolution image models, multi-image batching — the GPU is
   the right tool today, with the kernels we have. That conclusion is
   the most transferable artifact of M6.

#### What's left (and why we stopped)

| candidate | est. e2e win | why not pursued in M6 |
|---|---|---|
| fp16 throughout activation pipeline | ~10-15% on M=257 | pipeline-wide rewrite; modest gain; M=257 already CPU-dominated regime |
| Norm + MulMat fusion | 2-3% | high kernel complexity, modest win |
| cl_arm_import_memory (re-attempt with dma_buf) | up to ~3 s of cold start | requires Android Framework HAL path, off-scope for a CLI |
| Q4_K activation-precision tuning (per-sub-block sum_x) | small | the inline sum_x already works; precomputing wouldn't beat memory bandwidth |
| 5-bit quants (Q5_K/Q5_0) | quality at gq4 size with better cosine | format support work, plus the model's tail is already in int8 territory |
| Snapdragon (Adreno) re-profile | unknown | different microarch; M6 was Tensor G4 / Mali-only |

Each is ≤ 5% expected e2e on the production M=257 path, against significant
implementation cost. Past this point, **the right next investment is
architecture, not kernels**: smaller models, fewer steps, longer prefills
where the GPU lead compounds.

M6 done.
