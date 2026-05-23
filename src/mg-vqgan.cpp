// mg-vqgan.cpp — MaskGIT VQGAN decoder forward graph.
//
// Mirrors reference/maskgit/nets/vqgan_tokenizer.py Decoder + ResBlock.
// Activations use layout ne = {W, H, C, N}. GroupNorm has 32 groups, eps 1e-5
// (PyTorch default), swish activation. The ResBlock shortcut conv is applied to
// the PROCESSED tensor (faithful reproduction of the upstream quirk).
#include "mg-vqgan.hpp"

#include <string>
#include <vector>

namespace mg {
namespace {

constexpr float GN_EPS = 1e-5f;   // PyTorch nn.GroupNorm default
constexpr int   GROUPS = 32;

Tensor* swish(Context& c, Tensor* x) { return silu(c, x); }

// GroupNorm + per-channel affine. Channels are dim2 in {W,H,C,N}.
Tensor* gn_affine(Context& c, const Model& m, const std::string& prefix, Tensor* x) {
    const int64_t C = x->ne[2];
    Tensor* n = group_norm(c, x, GROUPS, GN_EPS);
    Tensor* w = reshape(c, m.require(prefix + ".weight"), {1, 1, C, 1});
    Tensor* b = reshape(c, m.require(prefix + ".bias"),   {1, 1, C, 1});
    return add(c, mul(c, n, w), b);
}

// Conv2d (+ optional per-output-channel bias).
Tensor* conv(Context& c, const Model& m, const std::string& prefix, Tensor* x,
             int stride, int pad, bool use_bias) {
    Tensor* y = conv_2d(c, m.require(prefix + ".weight"), x, stride, pad);
    if (use_bias) {
        const int64_t OC = y->ne[2];
        y = add(c, y, reshape(c, m.require(prefix + ".bias"), {1, 1, OC, 1}));
    }
    return y;
}

// ResBlock: norm0->swish->conv0 -> norm1->swish->conv1 -> (+ shortcut).
Tensor* resblock(Context& c, const Model& m, const std::string& pfx,
                 Tensor* x, int in_dim, int out_dim) {
    Tensor* residual = x;
    Tensor* h = gn_affine(c, m, pfx + "norm0", x);
    h = swish(c, h);
    h = conv(c, m, pfx + "conv0", h, 1, 1, /*bias=*/false);   // in->out, 3x3, pad1
    h = gn_affine(c, m, pfx + "norm1", h);
    h = swish(c, h);
    h = conv(c, m, pfx + "conv1", h, 1, 1, /*bias=*/false);   // out->out, 3x3, pad1
    if (in_dim != out_dim) {
        // QUIRK: shortcut conv applies to the processed h, not the input residual.
        residual = conv(c, m, pfx + "conv_res", h, 1, 0, /*bias=*/false);  // 1x1
    }
    return add(c, h, residual);
    (void)out_dim;
}

} // namespace

Tensor* build_vqgan_decoder(Context& c, const Model& m, Tensor* grid_ids) {
    const auto& h = m.hparams();
    const int filters = m.hparams().vq_filters;            // 128
    const auto& mult = m.hparams().vq_channel_mult;        // [1,1,2,2,4]
    const int nrb = m.hparams().vq_num_res_blocks;         // 2
    const int num_blocks = (int)mult.size();               // 5
    int64_t L = 1; while (L * L < h.n_tokens) L++;          // latent side = sqrt(n_tokens) = 16

    // codebook lookup -> {C, W, H, 1} -> {W, H, C, 1}
    Tensor* feat = get_rows(c, m.require("vqgan.quantizer.codebook.weight"), grid_ids); // {256, n}
    feat = reshape(c, feat, {(int64_t)h.vq_embedding_dim, L, L, 1});   // {C,W,H,1}
    feat = cont(c, permute(c, feat, 2, 0, 1, 3));                      // {W,H,C,1}

    int curr = filters * mult[num_blocks - 1];             // 512
    Tensor* x = conv(c, m, "vqgan.decoder.conv_in", feat, 1, 1, /*bias=*/true);  // 256->512

    // mid block (res_blocks.0): nrb x ResBlock(curr,curr)
    for (int l = 0; l < nrb; l++)
        x = resblock(c, m, "vqgan.decoder.res_blocks.0." + std::to_string(l) + ".", x, curr, curr);

    int prev = curr;
    int blockidx = 1;
    for (int i = num_blocks - 1; i >= 0; i--) {
        curr = filters * mult[i];
        for (int l = 0; l < nrb; l++) {
            std::string pfx = "vqgan.decoder.res_blocks." + std::to_string(blockidx) + "." + std::to_string(l) + ".";
            x = resblock(c, m, pfx, x, prev, curr);
            prev = curr;
        }
        if (i > 0) {
            x = upscale(c, x, 2);
            // upsample conv lives at layer index nrb+1 (== 3 for nrb=2)
            std::string up = "vqgan.decoder.res_blocks." + std::to_string(blockidx) + "." + std::to_string(nrb + 1);
            x = conv(c, m, up, x, 1, 1, /*bias=*/true);    // curr->curr, 3x3, pad1
        }
        blockidx++;
    }

    x = gn_affine(c, m, "vqgan.decoder.norm_out", x);      // 128 ch
    x = swish(c, x);
    x = conv(c, m, "vqgan.decoder.conv_out", x, 1, 1, /*bias=*/true);  // 128->3
    return x;                                              // {W,H,3,1}
}

} // namespace mg
