// mg-transformer.cpp — MaskGIT bidirectional transformer forward graph.
//
// Mirrors reference/maskgit/nets/bidirectional_transformer.py:
//   embed = emb_ln(tok_emb(x) + pos_emb[:S]); drop(identity)
//   per layer (post-norm): x = AttentionLN(x + MHA(x)); x = MlpLN(x + MLP(x))
//   h = LayerNorm(GELU(Linear(x)));  logits = h @ tok_emb.weight.T + bias
#include "mg-transformer.hpp"

#include <cmath>
#include <cstdlib>

namespace mg {
namespace {

// LayerNorm with affine over ne[0]: add_bias(mul(norm(x), w), b)
Tensor* layer_norm_affine(Context& c, Tensor* x, Tensor* w, Tensor* b, float eps) {
    return add_bias(c, mul(c, norm(c, x, eps), w), b);
}

// Linear y = act(x @ W^T + bias) + residual, with the bias / activation / residual
// fused into the matmul epilogue (mul_mat_ex) so they aren't separate graph nodes.
// act: 0=none, 1=gelu, 2=silu. residual may be null.
Tensor* linear(Context& c, const Model& m, const std::string& wname,
               const std::string& bname, Tensor* x, int act = 0, Tensor* residual = nullptr) {
    Tensor* bias = bname.empty() ? nullptr : m.require(bname);
    return mul_mat_ex(c, m.require(wname), x, bias, act, residual);
}

} // namespace

Tensor* build_transformer(Context& c, const Model& m, Tensor* token_ids) {
    const auto& h = m.hparams();
    const int64_t S = token_ids->ne[0];
    const int64_t B = token_ids->ne[1];
    const int H = h.n_head, D = h.head_dim;          // 16, 48
    const float eps = h.ln_eps;
    const float attn_scale = 1.0f / std::sqrt((float)D);

    // --- embeddings ---
    Tensor* tok = get_rows(c, m.require("token_embd.weight"), token_ids);  // {768,S,B}
    Tensor* pe = m.require("pos_embd.weight");                              // {768,257}
    int64_t pne[MAX_DIMS] = {h.n_embd, S, 1, 1};
    Tensor* pos = c.external(Type::F32, 2, pne, pe->data);                  // prefix view {768,S}
    Tensor* x = add(c, tok, pos);
    x = layer_norm_affine(c, x, m.require("token_embd_norm.weight"),
                          m.require("token_embd_norm.bias"), eps);

    // --- transformer layers (post-norm) ---
    // Flash-attention is default-on as of M6 #8. Set MG_NO_FLASH_ATTN=1 to fall back
    // to the unfused MulMat(K,Q) → SoftMax → MulMat(Vᵀ, scores) chain (e.g. for A/B
    // testing or debugging a backend kernel). Quality verified within noise on
    // Quick-5 IS/top-k; cosine 0.99999979 vs the unfused chain. See DEEP_DIVE §13.6.
    const bool no_flash_attn = std::getenv("MG_NO_FLASH_ATTN") != nullptr;
    for (int i = 0; i < h.n_layer; i++) {
        std::string p = "blk." + std::to_string(i) + ".";

        // self-attention
        Tensor* q = linear(c, m, p + "attn_q.weight", p + "attn_q.bias", x);  // {768,S,B}
        Tensor* k = linear(c, m, p + "attn_k.weight", p + "attn_k.bias", x);
        Tensor* v = linear(c, m, p + "attn_v.weight", p + "attn_v.bias", x);

        // split heads: {768,S,B} -> {D,H,S,B} -> permute to {D,S,H,B}
        q = permute(c, reshape(c, q, {D, H, S, B}), 0, 2, 1, 3);   // {D,S,H,B}
        k = permute(c, reshape(c, k, {D, H, S, B}), 0, 2, 1, 3);   // {D,S,H,B}
        v = permute(c, reshape(c, v, {D, H, S, B}), 0, 2, 1, 3);   // {D,S,H,B}

        Tensor* attn;
        if (!no_flash_attn) {
            // Default path: single fused op replaces the QK·softmax·V chain. Inputs
            // must be contiguous — cont() the permuted views (a strided-read FA
            // kernel could later avoid these copies).
            attn = flash_attention(c, cont(c, q), cont(c, k), cont(c, v), attn_scale); // {D,S,H,B}
        } else {
            // Legacy unfused path (debug/A-B). scores[t,s,h,b] = sum_d K[d,t]·Q[d,s].
            Tensor* scores = mul_mat(c, k, q);                          // {S(keys),S(query),H,B}
            scores = soft_max(c, scores, attn_scale);
            Tensor* vp = permute(c, v, 1, 0, 2, 3);                     // {S(t),D,H,B}
            attn = mul_mat(c, vp, scores);                              // {D,S(query),H,B}
        }

        // merge heads: {D,S,H,B} -> {D,H,S,B} -> cont -> {768,S,B}
        attn = cont(c, permute(c, attn, 0, 2, 1, 3));               // {D,H,S,B} contiguous
        attn = reshape(c, attn, {(int64_t)h.n_embd, S, B});

        // output proj + residual fused into the matmul epilogue (was o=Wo·attn+b; x+=o)
        x = linear(c, m, p + "attn_o.weight", p + "attn_o.bias", attn, /*act=*/0, /*resid=*/x);
        x = layer_norm_affine(c, x, m.require(p + "attn_norm.weight"),
                              m.require(p + "attn_norm.bias"), eps);  // post-norm

        // MLP: FFN-up fuses GELU; FFN-down fuses the residual add
        Tensor* ff = linear(c, m, p + "ffn_up.weight", p + "ffn_up.bias", x, /*act=gelu*/1);  // {3072,S,B}
        x = linear(c, m, p + "ffn_down.weight", p + "ffn_down.bias", ff, /*act=*/0, /*resid=*/x); // {768,S,B}
        x = layer_norm_affine(c, x, m.require(p + "ffn_norm.weight"),
                              m.require(p + "ffn_norm.bias"), eps);   // post-norm
    }

    // --- MLM head ---  output proj fuses GELU
    Tensor* hd = linear(c, m, "output_proj.weight", "output_proj.bias", x, /*act=gelu*/1);  // {768,S,B}
    hd = layer_norm_affine(c, hd, m.require("output_norm.weight"),
                           m.require("output_norm.bias"), eps);
    // logits = h @ tok_emb.weight^T + output.bias (bias fused)
    Tensor* logits = mul_mat_ex(c, m.require("token_embd.weight"), hd,
                                m.require("output.bias"), /*act=*/0, /*resid=*/nullptr);  // {vocab,S,B}
    return logits;
}

} // namespace mg
