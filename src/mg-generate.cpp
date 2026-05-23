// mg-generate.cpp — host-side iterative masked decoding + VQGAN decode.
//
// Mirrors reference/maskgit/libml/parallel_decode.py: cosine mask schedule,
// per-position categorical sampling over the codebook, confidence = log(p) +
// temperature*gumbel, re-mask the lowest-confidence tokens each step. RNG differs
// from PyTorch so pixels won't bit-match the reference, but the class-conditional
// result is equivalent in distribution.
#include "mg-generate.hpp"
#include "mg-transformer.hpp"
#include "mg-vqgan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace mg {
namespace {

inline float gumbel(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(1e-12, 1.0);
    return (float)(-std::log(-std::log(u(rng))));
}

// sample an index in [0, n) from probs (already normalized)
int sample_categorical(const float* probs, int n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng), acc = 0.0;
    for (int i = 0; i < n; i++) { acc += probs[i]; if (r <= acc) return i; }
    return n - 1;
}

} // namespace

Image generate(const Model& m, const GenConfig& cfg, bool verbose) {
    const auto& h = m.hparams();
    const int CB = h.vq_codebook_size;        // 1024
    const int MASK = h.mask_token_id;          // 2024
    const int S = h.n_tokens + 1;              // 257 (incl. class token)
    const float choice_temp = (cfg.temperature < 0) ? h.choice_temperature : cfg.temperature;

    std::mt19937_64 rng(cfg.seed);

    // init: class token at pos 0, all image positions masked
    std::vector<int32_t> tokens(S, MASK);
    tokens[0] = cfg.class_id + CB;             // label shifted by codebook size
    const int unknown_begin = h.n_tokens;      // all image positions start masked

    std::vector<int32_t> sampled(S), result(S);
    std::vector<float>   selp(S), conf(S);
    std::vector<float>   probs(CB);

    Context ctx(1536ull << 20);                // transformer arena (~1.5GB), reused per step

    for (int step = 0; step < cfg.steps; step++) {
        ctx.reset();
        Tensor* tok = ctx.tensor2d(Type::I32, S, 1);
        std::memcpy(tok->data, tokens.data(), S * sizeof(int32_t));
        Tensor* logits = build_transformer(ctx, m, tok);   // {vocab, S, 1}
        Graph g; g.build_forward(logits); compute(ctx, g);
        const float* L = static_cast<const float*>(logits->data);  // elem(v,s)=L[s*vocab+v]
        const int64_t vocab = logits->ne[0];

        // sample each position over the codebook (first CB logits)
        for (int s = 0; s < S; s++) {
            const float* row = L + (int64_t)s * vocab;
            float mx = -INFINITY;
            for (int v = 0; v < CB; v++) mx = std::fmax(mx, row[v]);
            float sum = 0.f;
            for (int v = 0; v < CB; v++) { float e = std::exp(row[v] - mx); probs[v] = e; sum += e; }
            float inv = 1.f / sum;
            for (int v = 0; v < CB; v++) probs[v] *= inv;
            int sid = sample_categorical(probs.data(), CB, rng);
            bool unknown = (tokens[s] == MASK);
            sampled[s] = unknown ? sid : tokens[s];
            selp[s]    = unknown ? probs[sid] : INFINITY;   // known tokens: max confidence
        }
        result = sampled;   // final_seqs[:, step] (pre re-mask) — last iter is the output

        // cosine schedule
        float ratio = (float)(step + 1) / cfg.steps;
        float mask_ratio = std::cos((float)M_PI / 2.f * ratio);
        int n_unknown_now = 0;
        for (int s = 0; s < S; s++) n_unknown_now += (tokens[s] == MASK);
        int mask_len = (int)std::floor(unknown_begin * mask_ratio);
        mask_len = std::max(1, std::min(mask_len, n_unknown_now - 1));

        // confidence + re-mask the mask_len lowest
        float temp = choice_temp * (1.f - ratio);
        for (int s = 0; s < S; s++) conf[s] = std::log(selp[s]) + temp * gumbel(rng);
        std::vector<float> sorted(conf);
        std::nth_element(sorted.begin(), sorted.begin() + mask_len, sorted.end());
        float cutoff = sorted[mask_len];
        for (int s = 0; s < S; s++)
            tokens[s] = (conf[s] < cutoff) ? MASK : sampled[s];

        if (verbose) { std::printf("[gen] step %d/%d  mask_len=%d  unknown_now=%d\n",
                                   step + 1, cfg.steps, mask_len, n_unknown_now); std::fflush(stdout); }
    }

    // final token grid (drop class token) -> VQGAN decode (its own, larger arena)
    int n_tok = h.n_tokens;
    Context vctx(3ull << 30);                  // VQGAN arena (~3GB; 256x256 activations)
    Tensor* grid = vctx.tensor1d(Type::I32, n_tok);
    std::memcpy(grid->data, result.data() + 1, n_tok * sizeof(int32_t));
    if (verbose) { std::printf("[gen] decoding image (VQGAN)...\n"); std::fflush(stdout); }
    Tensor* imgT = build_vqgan_decoder(vctx, m, grid);   // {W,H,3,1}
    Graph g; g.build_forward(imgT); compute(vctx, g);
    if (verbose) { std::printf("[gen] VQGAN arena used: %.2f GB\n", vctx.used() / 1e9); std::fflush(stdout); }

    const int64_t W = imgT->ne[0], H = imgT->ne[1];
    const float* o = static_cast<const float*>(imgT->data);
    Image img; img.width = (int)W; img.height = (int)H; img.channels = 3;
    img.rgb.resize((size_t)W * H * 3);
    for (int64_t y = 0; y < H; y++)
        for (int64_t x = 0; x < W; x++)
            for (int c = 0; c < 3; c++) {
                float v = o[x + W * (y + H * c)];           // ours {W,H,C}
                v = std::fmin(std::fmax(v, 0.f), 1.f) * 255.f + 0.5f;
                img.rgb[(y * W + x) * 3 + c] = (uint8_t)v;   // [H,W,3]
            }
    return img;
}

} // namespace mg
