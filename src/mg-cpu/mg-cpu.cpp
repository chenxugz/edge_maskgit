// mg-cpu.cpp — F32 CPU backend: graph executor + scalar kernels.
//
// Correctness-first: stride-aware scalar implementations that match the PyTorch
// reference numerically (exact erf-GELU, swish, direct conv). SIMD/threading and
// im2col come later. n_threads is accepted but currently ignored.
#include "mg-tensor.hpp"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mg {
namespace {

// strided element address for a (possibly view) tensor
inline const float* fptr(const Tensor* t, int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    return reinterpret_cast<const float*>(
        reinterpret_cast<const char*>(t->data) +
        i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3]);
}
// broadcast index: if src dim size is 1, always read index 0
inline int64_t bidx(const Tensor* s, int dim, int64_t i) { return s->ne[dim] == 1 ? 0 : i; }

// ---- kernels (dst is contiguous, freshly allocated) ----

void k_add_mul(Tensor* dst, const Tensor* a, const Tensor* b, bool is_mul) {
    float* o = static_cast<float*>(dst->data);
    int64_t idx = 0;
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < dst->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < dst->ne[1]; i1++)
    for (int64_t i0 = 0; i0 < dst->ne[0]; i0++) {
        float av = *fptr(a, i0, i1, i2, i3);
        float bv = *fptr(b, bidx(b,0,i0), bidx(b,1,i1), bidx(b,2,i2), bidx(b,3,i3));
        o[idx++] = is_mul ? av * bv : av + bv;
    }
}

void k_scale(Tensor* dst, const Tensor* a, float s) {
    float* o = static_cast<float*>(dst->data);
    int64_t n = dst->nelements();
    const float* x = static_cast<const float*>(a->data);
    for (int64_t i = 0; i < n; i++) o[i] = x[i] * s;
}

// w[K,N,wb2,wb3], x[K,M,b2,b3] -> dst[N,M,b2,b3]; w broadcasts over batch if its dim==1
void k_mul_mat(Tensor* dst, const Tensor* w, const Tensor* x) {
    const int64_t K = w->ne[0], N = w->ne[1];
    const int64_t M = x->ne[1], B2 = x->ne[2], B3 = x->ne[3];
    float* o = static_cast<float*>(dst->data);
    // optional fused epilogue (mul_mat_ex): out = act(acc + bias[n]) + residual
    const float* bias  = dst->src[2] ? static_cast<const float*>(dst->src[2]->data) : nullptr;
    const float* resid = dst->src[3] ? static_cast<const float*>(dst->src[3]->data) : nullptr;
    const int act = dst->iparam[0];
    for (int64_t b3 = 0; b3 < B3; b3++)
    for (int64_t b2 = 0; b2 < B2; b2++) {
        int64_t wb2 = bidx(w,2,b2), wb3 = bidx(w,3,b3);
        for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++)
                acc += *fptr(w,k,n,wb2,wb3) * *fptr(x,k,m,b2,b3);
            if (bias)  acc += bias[n];
            if (act == 1) acc = 0.5f * acc * (1.0f + std::erf(acc * 0.70710678f));   // gelu
            else if (act == 2) acc = acc / (1.0f + std::exp(-acc));                  // silu
            int64_t idx = ((b3*B2 + b2)*M + m)*N + n;
            if (resid) acc += resid[idx];
            o[idx] = acc;
        }
    }
}

void k_get_rows(Tensor* dst, const Tensor* a, const Tensor* ids) {
    const int64_t E = a->ne[0], R = a->ne[1];
    const int64_t n = ids->nelements();
    float* o = static_cast<float*>(dst->data);
    const int32_t* id = static_cast<const int32_t*>(ids->data);
    for (int64_t i = 0; i < n; i++) {
        int64_t r = id[i];
        if (r < 0) r += R;                 // mask token id = -1 -> last row (PyTorch semantics)
        assert(r >= 0 && r < R && "get_rows index out of range");
        const float* row = fptr(a, 0, r, 0, 0);
        std::memcpy(o + i * E, row, E * sizeof(float));
    }
}

void k_soft_max(Tensor* dst, const Tensor* a, float scale) {
    const int64_t D = dst->ne[0];
    float* o = static_cast<float*>(dst->data);
    int64_t row = 0;
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < dst->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < dst->ne[1]; i1++, row++) {
        float mx = -INFINITY;
        for (int64_t k = 0; k < D; k++) mx = std::fmax(mx, *fptr(a,k,i1,i2,i3) * scale);
        float sum = 0.0f;
        for (int64_t k = 0; k < D; k++) { float e = std::exp(*fptr(a,k,i1,i2,i3)*scale - mx); o[row*D+k] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int64_t k = 0; k < D; k++) o[row*D+k] *= inv;
    }
}

// LayerNorm over ne[0], no affine
void k_norm(Tensor* dst, const Tensor* a, float eps) {
    const int64_t D = dst->ne[0];
    float* o = static_cast<float*>(dst->data);
    int64_t row = 0;
    for (int64_t i3 = 0; i3 < dst->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < dst->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < dst->ne[1]; i1++, row++) {
        double mean = 0.0;
        for (int64_t k = 0; k < D; k++) mean += *fptr(a,k,i1,i2,i3);
        mean /= D;
        double var = 0.0;
        for (int64_t k = 0; k < D; k++) { double d = *fptr(a,k,i1,i2,i3) - mean; var += d*d; }
        var /= D;
        float inv = 1.0f / std::sqrt((float)var + eps);
        for (int64_t k = 0; k < D; k++) o[row*D+k] = ((float)(*fptr(a,k,i1,i2,i3) - mean)) * inv;
    }
}

// GroupNorm, no affine. Layout ne={W,H,C,N}; normalize per (sample, group) over W*H*(C/groups).
void k_group_norm(Tensor* dst, const Tensor* a, int groups, float eps) {
    const int64_t W = a->ne[0], H = a->ne[1], C = a->ne[2], N = a->ne[3];
    const int64_t cpg = C / groups;
    float* o = static_cast<float*>(dst->data);
    for (int64_t n = 0; n < N; n++)
    for (int g = 0; g < groups; g++) {
        double mean = 0.0; int64_t cnt = W*H*cpg;
        for (int64_t c = g*cpg; c < (g+1)*cpg; c++)
            for (int64_t h = 0; h < H; h++) for (int64_t w = 0; w < W; w++)
                mean += *fptr(a,w,h,c,n);
        mean /= cnt;
        double var = 0.0;
        for (int64_t c = g*cpg; c < (g+1)*cpg; c++)
            for (int64_t h = 0; h < H; h++) for (int64_t w = 0; w < W; w++) {
                double d = *fptr(a,w,h,c,n) - mean; var += d*d;
            }
        var /= cnt;
        float inv = 1.0f / std::sqrt((float)var + eps);
        for (int64_t c = g*cpg; c < (g+1)*cpg; c++)
            for (int64_t h = 0; h < H; h++) for (int64_t w = 0; w < W; w++) {
                int64_t off = ((n*C + c)*H + h)*W + w;
                o[off] = ((float)(*fptr(a,w,h,c,n) - mean)) * inv;
            }
    }
}

void k_gelu(Tensor* dst, const Tensor* a) {   // exact erf GELU (matches PyTorch nn.GELU())
    int64_t n = dst->nelements();
    const float* x = static_cast<const float*>(a->data);
    float* o = static_cast<float*>(dst->data);
    const float inv_sqrt2 = 0.70710678118654752440f;
    for (int64_t i = 0; i < n; i++) o[i] = 0.5f * x[i] * (1.0f + std::erf(x[i] * inv_sqrt2));
}

void k_silu(Tensor* dst, const Tensor* a) {    // swish: x*sigmoid(x)
    int64_t n = dst->nelements();
    const float* x = static_cast<const float*>(a->data);
    float* o = static_cast<float*>(dst->data);
    for (int64_t i = 0; i < n; i++) o[i] = x[i] / (1.0f + std::exp(-x[i]));
}

// direct conv2d. kernel {KW,KH,IC,OC}, input {IW,IH,IC,N} -> {OW,OH,OC,N}
void k_conv_2d(Tensor* dst, const Tensor* ker, const Tensor* in, int stride, int pad) {
    const int64_t KW=ker->ne[0], KH=ker->ne[1], IC=ker->ne[2], OC=ker->ne[3];
    const int64_t IW=in->ne[0], IH=in->ne[1], N=in->ne[3];
    const int64_t OW=dst->ne[0], OH=dst->ne[1];
    float* o = static_cast<float*>(dst->data);
    for (int64_t n=0;n<N;n++)
    for (int64_t oc=0;oc<OC;oc++)
    for (int64_t oh=0;oh<OH;oh++)
    for (int64_t ow=0;ow<OW;ow++) {
        float acc=0.0f;
        for (int64_t ic=0;ic<IC;ic++)
        for (int64_t kh=0;kh<KH;kh++) {
            int64_t ih = oh*stride - pad + kh;
            if (ih<0||ih>=IH) continue;
            for (int64_t kw=0;kw<KW;kw++) {
                int64_t iw = ow*stride - pad + kw;
                if (iw<0||iw>=IW) continue;
                acc += *fptr(ker,kw,kh,ic,oc) * *fptr(in,iw,ih,ic,n);
            }
        }
        o[((n*OC+oc)*OH+oh)*OW+ow] = acc;
    }
}

void k_upscale(Tensor* dst, const Tensor* a, int f) {  // nearest neighbor
    const int64_t OW=dst->ne[0], OH=dst->ne[1], C=dst->ne[2], N=dst->ne[3];
    float* o = static_cast<float*>(dst->data);
    for (int64_t n=0;n<N;n++) for (int64_t c=0;c<C;c++)
    for (int64_t oh=0;oh<OH;oh++) for (int64_t ow=0;ow<OW;ow++)
        o[((n*C+c)*OH+oh)*OW+ow] = *fptr(a, ow/f, oh/f, c, n);
}

void k_cont(Tensor* dst, const Tensor* a) {   // materialize contiguous from strided view
    float* o = static_cast<float*>(dst->data);
    int64_t idx=0;
    for (int64_t i3=0;i3<dst->ne[3];i3++)
    for (int64_t i2=0;i2<dst->ne[2];i2++)
    for (int64_t i1=0;i1<dst->ne[1];i1++)
    for (int64_t i0=0;i0<dst->ne[0];i0++)
        o[idx++] = *fptr(a,i0,i1,i2,i3);
}

void exec(Tensor* t) {
    switch (t->op) {
        case Op::Add:       k_add_mul(t, t->src[0], t->src[1], false); break;
        case Op::Mul:       k_add_mul(t, t->src[0], t->src[1], true);  break;
        case Op::Scale:     k_scale(t, t->src[0], t->fparam[0]); break;
        case Op::MulMat:    k_mul_mat(t, t->src[0], t->src[1]); break;
        case Op::GetRows:   k_get_rows(t, t->src[0], t->src[1]); break;
        case Op::SoftMax:   k_soft_max(t, t->src[0], t->fparam[0]); break;
        case Op::Norm:      k_norm(t, t->src[0], t->fparam[0]); break;
        case Op::GroupNorm: k_group_norm(t, t->src[0], t->iparam[0], t->fparam[1]); break;
        case Op::Gelu:      k_gelu(t, t->src[0]); break;
        case Op::Silu:      k_silu(t, t->src[0]); break;
        case Op::Conv2D:    k_conv_2d(t, t->src[0], t->src[1], t->iparam[0], t->iparam[1]); break;
        case Op::Upscale:   k_upscale(t, t->src[0], t->iparam[0]); break;
        case Op::Cont:      k_cont(t, t->src[0]); break;
        case Op::Reshape: case Op::View: case Op::Permute:
            break;  // views: data already aliases src
        case Op::None:
            break;
    }
}

} // namespace

void compute(Context& /*ctx*/, Graph& g, int /*n_threads*/) {
    for (Tensor* t : g.nodes) exec(t);
}

} // namespace mg
