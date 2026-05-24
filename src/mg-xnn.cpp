// mg-xnn.cpp — XNNPACK subgraph for the MaskGIT bidirectional transformer.
#include "mg-xnn.hpp"

#include "xnnpack.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mg {
namespace {
void check(xnn_status s, const char* what) {
    if (s != xnn_status_success) throw std::runtime_error(std::string("xnn: ") + what);
}
using Dims = std::vector<size_t>;
} // namespace

XnnTransformer::XnnTransformer(const Model& m, int batch, int seq_len, Quant quant)
    : m_(m), B_(batch), S_(seq_len), quant_(quant) {
    const auto& h = m.hparams();
    V_ = h.vocab_size; H_ = h.n_head; D_ = h.head_dim; E_ = h.n_embd;
    emb_.resize((size_t)B_ * S_ * E_);

    check(xnn_initialize(nullptr), "initialize");
    xnn_subgraph_t sg = nullptr;
    check(xnn_create_subgraph(/*external_value_ids=*/2, 0, &sg), "create_subgraph");

    // ---- helpers ----
    auto internal = [&](const Dims& d) -> uint32_t {
        uint32_t id = XNN_INVALID_VALUE_ID;
        check(xnn_define_tensor_value(sg, xnn_datatype_fp32, d.size(), d.data(),
                                      nullptr, XNN_INVALID_VALUE_ID, 0, &id), "internal");
        return id;
    };
    // static weight from a model tensor; XNNPACK dims = reversed(ne)
    auto weight = [&](const std::string& name) -> uint32_t {
        Tensor* t = m_.require(name);
        Dims d;
        for (int i = t->n_dims - 1; i >= 0; i--) d.push_back((size_t)t->ne[i]);
        if (d.empty()) d.push_back(1);
        uint32_t id = XNN_INVALID_VALUE_ID;
        check(xnn_define_tensor_value(sg, xnn_datatype_fp32, d.size(), d.data(),
                                      t->data, XNN_INVALID_VALUE_ID, 0, &id), "weight");
        return id;
    };
    auto scalar = [&](float v) -> uint32_t {
        consts_.push_back(v);
        size_t d1 = 1;
        uint32_t id = XNN_INVALID_VALUE_ID;
        check(xnn_define_tensor_value(sg, xnn_datatype_fp32, 1, &d1, &consts_.back(),
                                      XNN_INVALID_VALUE_ID, 0, &id), "scalar");
        return id;
    };
    auto binary = [&](xnn_binary_operator op, uint32_t a, uint32_t b, const Dims& outd) -> uint32_t {
        uint32_t o = internal(outd);
        check(xnn_define_binary(sg, op, nullptr, a, b, o, 0), "binary");
        return o;
    };
    auto unary = [&](xnn_unary_operator op, uint32_t a, const Dims& outd) -> uint32_t {
        uint32_t o = internal(outd);
        check(xnn_define_unary(sg, op, nullptr, a, o, 0), "unary");
        return o;
    };
    // Per-output-channel symmetric int8 quantized FC weight (qcint8).
    // weight tensor ne={IC,OC} (ne0=IC inner) -> data is [OC,IC] row-major.
    auto qweight8 = [&](const std::string& name) -> uint32_t {
        Tensor* t = m_.require(name);
        const int64_t IC = t->ne[0], OC = t->ne[1];
        const float* w = static_cast<const float*>(t->data);
        qw_.emplace_back((size_t)OC * IC);
        qs_.emplace_back((size_t)OC);
        int8_t* q = qw_.back().data();
        float*  s = qs_.back().data();
        for (int64_t oc = 0; oc < OC; oc++) {
            float amax = 0.f;
            for (int64_t ic = 0; ic < IC; ic++) amax = std::fmax(amax, std::fabs(w[oc*IC+ic]));
            float sc = amax > 0.f ? amax / 127.f : 1.f;
            s[oc] = sc;
            float inv = 1.f / sc;
            for (int64_t ic = 0; ic < IC; ic++) {
                int v = (int)lrintf(w[oc*IC+ic] * inv);
                q[oc*IC+ic] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
            }
        }
        size_t d[2] = {(size_t)OC, (size_t)IC};
        uint32_t id = XNN_INVALID_VALUE_ID;
        check(xnn_define_channelwise_quantized_tensor_value(
                  sg, xnn_datatype_qcint8, s, 2, /*channel_dim=*/0, d, q,
                  XNN_INVALID_VALUE_ID, 0, &id), "qcint8");
        return id;
    };
    auto fc = [&](uint32_t in, const std::string& w, const std::string& b, const Dims& outd) -> uint32_t {
        uint32_t o = internal(outd);
        if (quant_ == Quant::Q8) {
            // dynamic per-row int8 quantize the activation, then int8 GEMM -> f32
            Dims ind = outd; ind.back() = (size_t)m_.require(w)->ne[0];   // IC (weight ne0)
            uint32_t qd = XNN_INVALID_VALUE_ID;
            check(xnn_define_dynamically_quantized_tensor_value(
                      sg, xnn_datatype_qdint8, ind.size(), /*num_nonbatch_dims=*/1,
                      ind.data(), XNN_INVALID_VALUE_ID, 0, &qd), "qdint8");
            check(xnn_define_unary(sg, xnn_unary_convert, nullptr, in, qd, 0), "convert");
            check(xnn_define_fully_connected(sg, -INFINITY, INFINITY, qd, qweight8(w), weight(b), o, 0), "qfc");
        } else {
            check(xnn_define_fully_connected(sg, -INFINITY, INFINITY, in, weight(w), weight(b), o, 0), "fc");
        }
        return o;
    };
    auto reshape = [&](uint32_t in, const Dims& nd) -> uint32_t {
        uint32_t o = internal(nd);
        check(xnn_define_static_reshape(sg, nd.size(), nd.data(), in, o, 0), "reshape");
        return o;
    };
    auto transpose = [&](uint32_t in, const Dims& perm, const Dims& outd) -> uint32_t {
        uint32_t o = internal(outd);
        check(xnn_define_static_transpose(sg, perm.size(), perm.data(), in, o, 0), "transpose");
        return o;
    };
    const float eps = h.ln_eps;
    uint32_t eps_id = scalar(eps);
    // LayerNorm over last dim (+ affine). x dims = {B,S,E}
    auto layernorm = [&](uint32_t x, const std::string& wn, const std::string& bn) -> uint32_t {
        Dims full = {(size_t)B_, (size_t)S_, (size_t)E_};
        Dims red  = {(size_t)B_, (size_t)S_, 1};
        size_t last = 2;
        uint32_t mean = internal(red);
        check(xnn_define_static_reduce(sg, xnn_reduce_mean, 1, &last, x, mean, XNN_FLAG_KEEP_DIMS), "mean");
        uint32_t xc  = binary(xnn_binary_subtract, x, mean, full);
        uint32_t sq  = unary(xnn_unary_square, xc, full);
        uint32_t var = internal(red);
        check(xnn_define_static_reduce(sg, xnn_reduce_mean, 1, &last, sq, var, XNN_FLAG_KEEP_DIMS), "mean2");
        uint32_t vare = binary(xnn_binary_add, var, eps_id, red);
        uint32_t inv  = unary(xnn_unary_reciprocal_square_root, vare, red);
        uint32_t xn   = binary(xnn_binary_multiply, xc, inv, full);
        uint32_t sc   = binary(xnn_binary_multiply, xn, weight(wn), full);
        return binary(xnn_binary_add, sc, weight(bn), full);
    };

    // ---- external input: embeddings [B,S,E] ----
    Dims embd = {(size_t)B_, (size_t)S_, (size_t)E_};
    check(xnn_define_tensor_value(sg, xnn_datatype_fp32, 3, embd.data(), nullptr, 0,
                                  XNN_VALUE_FLAG_EXTERNAL_INPUT, &in_id_), "ext_in");

    // x = emb + pos_embd[:S]   (pos is static {S,E}, broadcasts over B)
    uint32_t x = binary(xnn_binary_add, in_id_, weight("pos_embd.weight"), embd);
    x = layernorm(x, "token_embd_norm.weight", "token_embd_norm.bias");

    const float attn_scale = 1.0f / std::sqrt((float)D_);
    uint32_t scale_id = scalar(attn_scale);
    Dims bshe = {(size_t)B_, (size_t)S_, (size_t)H_, (size_t)D_};
    Dims bhsd = {(size_t)B_, (size_t)H_, (size_t)S_, (size_t)D_};
    Dims bhss = {(size_t)B_, (size_t)H_, (size_t)S_, (size_t)S_};
    Dims perm_to_bhsd = {0, 2, 1, 3};

    for (int i = 0; i < h.n_layer; i++) {
        std::string p = "blk." + std::to_string(i) + ".";
        // QKV projections
        uint32_t q = fc(x, p + "attn_q.weight", p + "attn_q.bias", embd);
        uint32_t k = fc(x, p + "attn_k.weight", p + "attn_k.bias", embd);
        uint32_t v = fc(x, p + "attn_v.weight", p + "attn_v.bias", embd);
        // [B,S,E] -> [B,S,H,D] -> [B,H,S,D]
        q = transpose(reshape(q, bshe), perm_to_bhsd, bhsd);
        k = transpose(reshape(k, bshe), perm_to_bhsd, bhsd);
        v = transpose(reshape(v, bshe), perm_to_bhsd, bhsd);
        // scores = q @ k^T  (TRANSPOSE_B) -> [B,H,S,S]
        uint32_t scores = internal(bhss);
        check(xnn_define_batch_matrix_multiply(sg, q, k, scores, XNN_FLAG_TRANSPOSE_B), "bmm_qk");
        scores = binary(xnn_binary_multiply, scores, scale_id, bhss);
        uint32_t attn = internal(bhss);
        check(xnn_define_softmax(sg, scores, attn, 0), "softmax");
        // ctx = attn @ v -> [B,H,S,D]
        uint32_t ctx = internal(bhsd);
        check(xnn_define_batch_matrix_multiply(sg, attn, v, ctx, 0), "bmm_av");
        // [B,H,S,D] -> [B,S,H,D] -> [B,S,E]
        ctx = reshape(transpose(ctx, perm_to_bhsd, bshe), embd);
        uint32_t o = fc(ctx, p + "attn_o.weight", p + "attn_o.bias", embd);
        x = binary(xnn_binary_add, x, o, embd);                      // residual
        x = layernorm(x, p + "attn_norm.weight", p + "attn_norm.bias");

        // FFN
        Dims bsf = {(size_t)B_, (size_t)S_, (size_t)h.n_ffn};
        uint32_t ff = fc(x, p + "ffn_up.weight", p + "ffn_up.bias", bsf);
        ff = unary(xnn_unary_gelu, ff, bsf);
        ff = fc(ff, p + "ffn_down.weight", p + "ffn_down.bias", embd);
        x = binary(xnn_binary_add, x, ff, embd);                     // residual
        x = layernorm(x, p + "ffn_norm.weight", p + "ffn_norm.bias");
    }

    // MLM head
    uint32_t hd = fc(x, "output_proj.weight", "output_proj.bias", embd);
    hd = unary(xnn_unary_gelu, hd, embd);
    hd = layernorm(hd, "output_norm.weight", "output_norm.bias");
    Dims bsv = {(size_t)B_, (size_t)S_, (size_t)V_};
    // logits = hd @ token_embd^T + output.bias  (tied head; token_embd is [V,E])
    check(xnn_define_tensor_value(sg, xnn_datatype_fp32, 3, bsv.data(), nullptr, 1,
                                  XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &out_id_), "ext_out");
    check(xnn_define_fully_connected(sg, -INFINITY, INFINITY, hd,
                                     weight("token_embd.weight"), weight("output.bias"),
                                     out_id_, 0), "head_fc");

    xnn_runtime_t rt = nullptr;
    check(xnn_create_runtime_v2(sg, /*threadpool=*/nullptr, 0, &rt), "create_runtime");
    subgraph_ = sg; runtime_ = rt;
}

void XnnTransformer::forward(const int32_t* tokens, float* logits_out) {
    // host-side embedding gather: emb[b,s,:] = token_embd.weight[tokens[b,s]]
    const Tensor* te = m_.require("token_embd.weight");   // {E, V} -> row r at data + r*E
    const float* tw = static_cast<const float*>(te->data);
    for (int i = 0; i < B_ * S_; i++) {
        int32_t r = tokens[i];
        if (r < 0) r += V_;
        std::memcpy(&emb_[(size_t)i * E_], tw + (size_t)r * E_, E_ * sizeof(float));
    }
    xnn_external_value ext[2] = {{in_id_, emb_.data()}, {out_id_, logits_out}};
    check(xnn_setup_runtime(static_cast<xnn_runtime_t>(runtime_), 2, ext), "setup");
    check(xnn_invoke_runtime(static_cast<xnn_runtime_t>(runtime_)), "invoke");
}

XnnTransformer::~XnnTransformer() {
    if (runtime_)  xnn_delete_runtime(static_cast<xnn_runtime_t>(runtime_));
    if (subgraph_) xnn_delete_subgraph(static_cast<xnn_subgraph_t>(subgraph_));
}

} // namespace mg
