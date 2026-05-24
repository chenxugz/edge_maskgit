// mg-tensor.hpp — minimal ggml-inspired tensor library (C++17) for the MaskGIT runtime.
//
// Conventions (documented once, relied on everywhere):
//  - Row-major. ne[0] is the INNERMOST (contiguous) dimension, like ggml.
//    A PyTorch/numpy tensor [d0, d1, ..., dn] maps to ne[] reversed, e.g.
//    activations [batch, seq, hidden] -> ne = {hidden, seq, batch, 1}.
//  - Up to MAX_DIMS dims; unused trailing dims are 1.
//  - nb[] are strides in BYTES. nb[0] = type size, nb[i] = nb[i-1]*ne[i-1] when
//    contiguous. Views may carry arbitrary strides.
//  - mul_mat(w, x): w is [K, N] (ne={K,N}), x is [K, M] (ne={K,M}) -> [N, M]
//    (ne={N,M}); result[n,m] = sum_k w[k,n]*x[k,m]. This matches a PyTorch
//    Linear weight stored [out=N, in=K]: pass weight as `w`, input as `x`.
//
// F32-first: only F32 (+ I32 indices) is computed. F16/quant types reserved.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mg {

inline constexpr int MAX_DIMS = 4;
inline constexpr int MAX_SRC = 2;
inline constexpr int MAX_OP_PARAMS = 4;

// I8 = 1 byte/elem; I4 = packed 2 elems/byte (XNNPACK weight storage). Q8_0 =
// ggml block-32 quant (fp16 scale + 32 int8 = 34 bytes/block; OpenCL GPU path).
enum class Type { F32 = 0, F16 = 1, I32 = 2, I8 = 3, I4 = 4, Q8_0 = 5 };

inline constexpr int Q8_0_BLOCK = 32;
inline constexpr int Q8_0_BYTES = 34;   // 2 (fp16 scale) + 32 (int8)

enum class Op {
    None = 0,
    Add,         // elementwise; src1 may broadcast over leading dims / be a row-vector bias
    Mul,         // elementwise (Hadamard), broadcast like Add
    Scale,       // out = src0 * fparam[0]
    MulMat,      // see header note
    GetRows,     // gather src0 rows by src1 (I32 indices)
    SoftMax,     // over ne[0], optional scale fparam[0]
    Norm,        // LayerNorm over ne[0], eps fparam[0] (no affine)
    GroupNorm,   // iparam[0]=groups, fparam[1]=eps (no affine)
    Gelu,
    Silu,
    Conv2D,      // src0=kernel {KW,KH,IC,OC}, src1=input {IW,IH,IC,N}; iparam[0]=stride, iparam[1]=pad
    Upscale,     // nearest-neighbor, iparam[0]=factor
    Reshape,     // view
    View,        // view
    Permute,     // view
    Cont,        // materialize contiguous copy
};

size_t type_size(Type t);

struct Tensor {
    Type    type = Type::F32;
    int     n_dims = 1;
    int64_t ne[MAX_DIMS] = {1, 1, 1, 1};
    size_t  nb[MAX_DIMS] = {0, 0, 0, 0};

    Op       op = Op::None;
    Tensor*  src[MAX_SRC] = {nullptr, nullptr};
    int32_t  iparam[MAX_OP_PARAMS] = {0, 0, 0, 0};
    float    fparam[MAX_OP_PARAMS] = {0, 0, 0, 0};

    void*    data = nullptr;   // points into Context arena or external (mmap'd weights)
    bool     is_view = false;
    std::string name;

    int64_t nelements() const;
    size_t  nbytes() const;
    bool    is_contiguous() const;

    // F32 element accessors (flat, contiguous assumption)
    float  f32(int64_t i) const { return static_cast<const float*>(data)[i]; }
    void   set_f32(int64_t i, float v) { static_cast<float*>(data)[i] = v; }

    Tensor* named(const std::string& n) { name = n; return this; }
};

// Owns all Tensor objects (stable addresses) and a bump arena for their data.
class Context {
public:
    explicit Context(size_t arena_bytes);

    // creation
    Tensor* tensor(Type t, int n_dims, const int64_t ne[MAX_DIMS]);
    Tensor* tensor1d(Type t, int64_t ne0);
    Tensor* tensor2d(Type t, int64_t ne0, int64_t ne1);
    Tensor* tensor3d(Type t, int64_t ne0, int64_t ne1, int64_t ne2);
    Tensor* tensor4d(Type t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
    // wrap externally-owned data (no copy), e.g. mmap'd weights
    Tensor* external(Type t, int n_dims, const int64_t ne[MAX_DIMS], void* data);

    size_t used() const { return off_; }
    void   reset();                      // rewind arena + drop all tensors (reuse across iterations)
    void*  alloc(size_t bytes);          // arena bump allocator (32B aligned)

    // raw node factory used by ops
    Tensor* make(Type t, int n_dims, const int64_t ne[MAX_DIMS], Op op,
                 Tensor* s0, Tensor* s1, bool alloc_data);

private:
    std::vector<std::unique_ptr<Tensor>> nodes_;
    std::vector<uint8_t> arena_;
    size_t off_ = 0;
};

// ---- ops (build graph nodes; compute happens in mg_compute) ----
Tensor* add(Context&, Tensor* a, Tensor* b);
Tensor* mul(Context&, Tensor* a, Tensor* b);
Tensor* add_bias(Context&, Tensor* a, Tensor* bias);   // bias is a row vector ne={ne0}
Tensor* scale(Context&, Tensor* a, float s);
Tensor* mul_mat(Context&, Tensor* w, Tensor* x);
Tensor* get_rows(Context&, Tensor* a, Tensor* ids);
Tensor* soft_max(Context&, Tensor* a, float scale = 1.0f);
Tensor* norm(Context&, Tensor* a, float eps);
Tensor* group_norm(Context&, Tensor* a, int groups, float eps);
Tensor* gelu(Context&, Tensor* a);
Tensor* silu(Context&, Tensor* a);
Tensor* conv_2d(Context&, Tensor* kernel, Tensor* input, int stride, int pad);
Tensor* upscale(Context&, Tensor* a, int factor);

// views (zero-copy)
Tensor* reshape(Context&, Tensor* a, std::vector<int64_t> ne);
Tensor* permute(Context&, Tensor* a, int a0, int a1, int a2, int a3);
Tensor* cont(Context&, Tensor* a);

// ---- graph ----
struct Graph {
    std::vector<Tensor*> nodes;   // topologically sorted
    void build_forward(Tensor* out);
};

void compute(Context&, Graph&, int n_threads = 1);

} // namespace mg
