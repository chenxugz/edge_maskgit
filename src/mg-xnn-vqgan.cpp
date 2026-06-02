// mg-xnn-vqgan.cpp — XNNPACK subgraph for the MaskGIT VQGAN decoder (NHWC).
#include "mg-xnn-vqgan.hpp"

#include "xnnpack.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mg {
namespace {
void check(xnn_status s, const char* what) {
    if (s != xnn_status_success) throw std::runtime_error(std::string("xnn-vqgan: ") + what);
}
using Dims = std::vector<size_t>;
constexpr int GROUPS = 32;
constexpr float GN_EPS = 1e-5f;
} // namespace

XnnVqgan::XnnVqgan(const Model& m, Quant quant) : m_(m), quant_(quant) {
    const auto& h = m.hparams();
    E_ = h.vq_embedding_dim;                 // 256
    const int filters = h.vq_filters;        // 128
    const auto& mult = h.vq_channel_mult;    // [1,1,2,2,4]
    const int nrb = h.vq_num_res_blocks;     // 2
    const int nblocks = (int)mult.size();    // 5
    int64_t L = 1; while (L * L < h.n_tokens) L++;   // latent side 16
    feat_.resize((size_t)L * L * E_);

    // Pre-quantized model? VQGAN conv weights are int8 -> use the quantized path.
    if (Tensor* c0 = m.get("vqgan.decoder.conv_in.weight"))
        if (c0->type == Type::I8) quant_ = Quant::Q8;

    check(xnn_initialize(nullptr), "initialize");
    xnn_subgraph_t sg = nullptr;
    check(xnn_create_subgraph(2, 0, &sg), "create_subgraph");

    auto internal = [&](const Dims& d) -> uint32_t {
        uint32_t id = XNN_INVALID_VALUE_ID;
        check(xnn_define_tensor_value(sg, xnn_datatype_fp32, d.size(), d.data(),
                                      nullptr, XNN_INVALID_VALUE_ID, 0, &id), "internal");
        return id;
    };
    auto static_tensor = [&](const Dims& d, const float* data) -> uint32_t {
        uint32_t id = XNN_INVALID_VALUE_ID;
        check(xnn_define_tensor_value(sg, xnn_datatype_fp32, d.size(), d.data(),
                                      data, XNN_INVALID_VALUE_ID, 0, &id), "static");
        return id;
    };
    auto vec1d = [&](const std::string& name) -> uint32_t {     // 1D affine/bias [C]
        Tensor* t = m_.require(name);
        Dims d = {(size_t)t->ne[0]};
        return static_tensor(d, static_cast<const float*>(t->data));
    };
    auto scalar = [&](float v) -> uint32_t {
        consts_.push_back(v); size_t d1 = 1;
        return static_tensor(Dims{d1}, &consts_.back());
    };
    auto binary = [&](xnn_binary_operator op, uint32_t a, uint32_t b, const Dims& o) -> uint32_t {
        uint32_t r = internal(o);
        check(xnn_define_binary(sg, op, nullptr, a, b, r, 0), "binary");
        return r;
    };
    auto unary = [&](xnn_unary_operator op, uint32_t a, const Dims& o) -> uint32_t {
        uint32_t r = internal(o);
        check(xnn_define_unary(sg, op, nullptr, a, r, 0), "unary");
        return r;
    };
    auto reshape = [&](uint32_t in, const Dims& nd) -> uint32_t {
        uint32_t r = internal(nd);
        check(xnn_define_static_reshape(sg, nd.size(), nd.data(), in, r, 0), "reshape");
        return r;
    };
    auto swish = [&](uint32_t x, const Dims& d) -> uint32_t {
        return binary(xnn_binary_multiply, x, unary(xnn_unary_sigmoid, x, d), d);
    };

    // repack a conv weight OIHW (PyTorch, ne={KW,KH,IC,OC}) -> OHWI [OC,KH,KW,IC]
    auto conv_weight = [&](const std::string& name) -> uint32_t {
        Tensor* t = m_.require(name);
        const int64_t KW = t->ne[0], KH = t->ne[1], IC = t->ne[2], OC = t->ne[3];
        const float* src = static_cast<const float*>(t->data);
        packed_.emplace_back((size_t)OC * KH * KW * IC);
        float* dst = packed_.back().data();
        for (int64_t oc = 0; oc < OC; oc++)
          for (int64_t ic = 0; ic < IC; ic++)
            for (int64_t kh = 0; kh < KH; kh++)
              for (int64_t kw = 0; kw < KW; kw++)
                dst[((oc*KH+kh)*KW+kw)*IC+ic] = src[((oc*IC+ic)*KH+kh)*KW+kw];
        return static_tensor(Dims{(size_t)OC,(size_t)KH,(size_t)KW,(size_t)IC}, dst);
    };

    // repack OHWI + per-output-channel symmetric int8 quant -> qcint8 conv filter
    auto conv_weight_q8 = [&](const std::string& name) -> uint32_t {
        Tensor* t = m_.require(name);
        // NOTE: pre-quantized conv (stored qcint8) is not used — it currently NaNs
        // in XNNPACK's qd8-f32-qc8w conv despite byte-identical weights/scales to the
        // working on-load path. Quantized GGUFs keep conv F32 and quantize here on load.
        if (t->type == Type::I8)
            throw std::runtime_error("pre-quantized conv not supported (see note); store conv as F32");
        const int64_t KW = t->ne[0], KH = t->ne[1], IC = t->ne[2], OC = t->ne[3];
        const float* src = static_cast<const float*>(t->data);
        const int64_t per_oc = KH * KW * IC;
        qw_.emplace_back((size_t)OC * per_oc);
        qs_.emplace_back((size_t)OC);
        int8_t* q = qw_.back().data();
        float*  s = qs_.back().data();
        for (int64_t oc = 0; oc < OC; oc++) {
            float amax = 0.f;
            for (int64_t i = 0; i < per_oc; i++) amax = std::fmax(amax, std::fabs(src[oc*per_oc + i]));
            float sc = amax > 0.f ? amax / 127.f : 1.f;
            s[oc] = sc;
            float inv = 1.f / sc;
            for (int64_t ic = 0; ic < IC; ic++)
              for (int64_t kh = 0; kh < KH; kh++)
                for (int64_t kw = 0; kw < KW; kw++) {
                    int v = (int)lrintf(src[((oc*IC+ic)*KH+kh)*KW+kw] * inv);
                    q[((oc*KH+kh)*KW+kw)*IC+ic] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
                }
        }
        size_t d[4] = {(size_t)OC,(size_t)KH,(size_t)KW,(size_t)IC};
        uint32_t id = XNN_INVALID_VALUE_ID;
        check(xnn_define_channelwise_quantized_tensor_value(
                  sg, xnn_datatype_qcint8, s, 4, /*channel_dim=*/0, d, q,
                  XNN_INVALID_VALUE_ID, 0, &id), "qcint8 conv");
        return id;
    };

    // conv2d NHWC. in [1,H,W,IC] -> [1,H,W,OC] (same spatial; pad keeps size).
    // Quantized path (quant_ != F32): dynamic int8 input + per-channel int8 filter.
    auto conv = [&](uint32_t in, const std::string& base, bool bias,
                    int OC, int IC, int K, int pad, int H, int W) -> uint32_t {
        uint32_t bid = bias ? vec1d(base + ".bias") : XNN_INVALID_VALUE_ID;
        uint32_t out = internal(Dims{1,(size_t)H,(size_t)W,(size_t)OC});
        if (quant_ != Quant::F32) {
            size_t ind[4] = {1,(size_t)H,(size_t)W,(size_t)IC};
            uint32_t qd = XNN_INVALID_VALUE_ID;
            check(xnn_define_dynamically_quantized_tensor_value(
                      sg, xnn_datatype_qdint8, 4, /*num_nonbatch_dims=*/3, ind,
                      XNN_INVALID_VALUE_ID, 0, &qd), "qdint8 conv");
            check(xnn_define_unary(sg, xnn_unary_convert, nullptr, in, qd, 0), "convert");
            check(xnn_define_convolution_2d(sg, pad,pad,pad,pad, K,K, 1,1, 1,1,
                      1, IC, OC, -INFINITY, INFINITY, qd, conv_weight_q8(base + ".weight"), bid, out, 0), "qconv");
        } else {
            check(xnn_define_convolution_2d(sg, pad,pad,pad,pad, K,K, 1,1, 1,1,
                      1, IC, OC, -INFINITY, INFINITY, in, conv_weight(base + ".weight"), bid, out, 0), "conv");
        }
        return out;
    };

    uint32_t eps_id = scalar(GN_EPS);
    // GroupNorm (NHWC) + affine over C (last dim)
    auto groupnorm = [&](uint32_t x, const std::string& base, int C, int H, int W) -> uint32_t {
        int Cg = C / GROUPS;
        Dims full = {1,(size_t)H,(size_t)W,(size_t)C};
        Dims g5   = {1,(size_t)H,(size_t)W,(size_t)GROUPS,(size_t)Cg};
        Dims red  = {1,1,1,(size_t)GROUPS,1};
        size_t axes[3] = {1,2,4};
        uint32_t r = reshape(x, g5);
        uint32_t mean = internal(red);
        check(xnn_define_static_reduce(sg, xnn_reduce_mean, 3, axes, r, mean, XNN_FLAG_KEEP_DIMS), "gn_mean");
        uint32_t xc = binary(xnn_binary_subtract, r, mean, g5);
        uint32_t sq = unary(xnn_unary_square, xc, g5);
        uint32_t var = internal(red);
        check(xnn_define_static_reduce(sg, xnn_reduce_mean, 3, axes, sq, var, XNN_FLAG_KEEP_DIMS), "gn_var");
        uint32_t inv = unary(xnn_unary_reciprocal_square_root, binary(xnn_binary_add, var, eps_id, red), red);
        uint32_t xn = binary(xnn_binary_multiply, xc, inv, g5);
        uint32_t back = reshape(xn, full);
        return binary(xnn_binary_add, binary(xnn_binary_multiply, back, vec1d(base+".weight"), full),
                      vec1d(base+".bias"), full);
    };

    // nearest 2x upsample via depthwise all-ones transposed conv (stride 2)
    auto upsample2x = [&](uint32_t x, int C, int H, int W) -> uint32_t {
        packed_.emplace_back((size_t)C*2*2*1, 1.0f);           // ones filter [C,2,2,1]
        uint32_t fid = static_tensor(Dims{(size_t)C,2,2,1}, packed_.back().data());
        packed_.emplace_back((size_t)C, 0.0f);                 // zero bias [C]
        uint32_t bid = static_tensor(Dims{(size_t)C}, packed_.back().data());
        uint32_t out = internal(Dims{1,(size_t)(2*H),(size_t)(2*W),(size_t)C});
        check(xnn_define_deconvolution_2d(sg, 0,0,0,0, 0,0, 2,2, 2,2, 1,1,
                  /*groups=*/C, /*gic=*/1, /*goc=*/1, -INFINITY, INFINITY, x, fid, bid, out, 0), "deconv");
        return out;
    };

    auto resblock = [&](const std::string& pfx, uint32_t x, int in_dim, int out_dim, int H, int W) -> uint32_t {
        Dims od = {1,(size_t)H,(size_t)W,(size_t)out_dim};
        uint32_t residual = x;
        uint32_t hh = groupnorm(x, pfx+"norm0", in_dim, H, W);
        hh = swish(hh, Dims{1,(size_t)H,(size_t)W,(size_t)in_dim});
        hh = conv(hh, pfx+"conv0", false, out_dim, in_dim, 3, 1, H, W);
        hh = groupnorm(hh, pfx+"norm1", out_dim, H, W);
        hh = swish(hh, od);
        hh = conv(hh, pfx+"conv1", false, out_dim, out_dim, 3, 1, H, W);
        if (in_dim != out_dim)
            residual = conv(hh, pfx+"conv_res", false, out_dim, out_dim, 1, 0, H, W);  // QUIRK: on hh
        return binary(xnn_binary_add, hh, residual, od);
    };

    // ---- external input: codebook features [1,L,L,E] ----
    Dims ind = {1,(size_t)L,(size_t)L,(size_t)E_};
    check(xnn_define_tensor_value(sg, xnn_datatype_fp32, 4, ind.data(), nullptr, 0,
                                  XNN_VALUE_FLAG_EXTERNAL_INPUT, &in_id_), "ext_in");

    int H = (int)L, W = (int)L;
    int curr = filters * mult[nblocks-1];                     // 512
    uint32_t x = conv(in_id_, "vqgan.decoder.conv_in", true, curr, E_, 3, 1, H, W);

    for (int l = 0; l < nrb; l++)                             // mid block (res_blocks.0)
        x = resblock("vqgan.decoder.res_blocks.0." + std::to_string(l) + ".", x, curr, curr, H, W);

    int prev = curr, blockidx = 1;
    for (int i = nblocks - 1; i >= 0; i--) {
        curr = filters * mult[i];
        for (int l = 0; l < nrb; l++) {
            std::string pfx = "vqgan.decoder.res_blocks." + std::to_string(blockidx) + "." + std::to_string(l) + ".";
            x = resblock(pfx, x, prev, curr, H, W);
            prev = curr;
        }
        if (i > 0) {
            x = upsample2x(x, curr, H, W); H *= 2; W *= 2;
            std::string up = "vqgan.decoder.res_blocks." + std::to_string(blockidx) + "." + std::to_string(nrb + 1);
            x = conv(x, up, true, curr, curr, 3, 1, H, W);    // post-upsample conv
        }
        blockidx++;
    }

    x = groupnorm(x, "vqgan.decoder.norm_out", curr, H, W);   // 128 ch
    x = swish(x, Dims{1,(size_t)H,(size_t)W,(size_t)curr});
    WH_ = H;
    // conv_out -> [1,H,W,3] external output
    {
        uint32_t fid = conv_weight("vqgan.decoder.conv_out.weight");
        uint32_t bid = vec1d("vqgan.decoder.conv_out.bias");
        Dims od = {1,(size_t)H,(size_t)W,3};
        check(xnn_define_tensor_value(sg, xnn_datatype_fp32, 4, od.data(), nullptr, 1,
                                      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id_), "ext_out");
        check(xnn_define_convolution_2d(sg, 1,1,1,1, 3,3, 1,1, 1,1, 1, curr, 3,
                  -INFINITY, INFINITY, x, fid, bid, out_id_, 0), "conv_out");
    }

    xnn_runtime_t rt = nullptr;
    check(xnn_create_runtime_v2(sg, nullptr, XNN_FLAG_BASIC_PROFILING, &rt), "create_runtime");
    subgraph_ = sg; runtime_ = rt;
}

// (xnn_op_bucket + xnn_pull_profile are defined in mg-xnn.cpp; declared extern here.)
extern const char* xnn_op_bucket(const char* n);
extern void xnn_pull_profile(xnn_runtime_t rt,
                             std::unordered_map<std::string,double>& tot_ms,
                             std::unordered_map<std::string,int>& tot_n);

void XnnVqgan::profile_enable(bool on) { prof_on_ = on; }
void XnnVqgan::profile_reset() { prof_ms_.clear(); prof_n_.clear(); }
std::vector<XnnVqgan::OpStat> XnnVqgan::profile_report() const {
    std::vector<OpStat> r;
    for (auto& kv : prof_ms_) r.push_back({kv.first, kv.second, prof_n_.at(kv.first)});
    std::sort(r.begin(), r.end(), [](const OpStat& a, const OpStat& b){ return a.ms > b.ms; });
    return r;
}

void XnnVqgan::decode(const int32_t* grid, float* image_out) {
    const Tensor* cb = m_.require("vqgan.quantizer.codebook.weight");   // {E, n_codes}
    const float* cw = static_cast<const float*>(cb->data);
    int n = (int)(feat_.size() / E_);
    for (int i = 0; i < n; i++) {
        int32_t r = grid[i];
        std::memcpy(&feat_[(size_t)i * E_], cw + (size_t)r * E_, E_ * sizeof(float));
    }
    xnn_external_value ext[2] = {{in_id_, feat_.data()}, {out_id_, image_out}};
    check(xnn_setup_runtime(static_cast<xnn_runtime_t>(runtime_), 2, ext), "setup");
    check(xnn_invoke_runtime(static_cast<xnn_runtime_t>(runtime_)), "invoke");
    if (prof_on_) xnn_pull_profile(static_cast<xnn_runtime_t>(runtime_), prof_ms_, prof_n_);
}

XnnVqgan::~XnnVqgan() {
    if (runtime_)  xnn_delete_runtime(static_cast<xnn_runtime_t>(runtime_));
    if (subgraph_) xnn_delete_subgraph(static_cast<xnn_subgraph_t>(subgraph_));
}

} // namespace mg
