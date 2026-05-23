// mg-model-info.cpp — load a GGUF and print hparams + tensor checksums.
// Used to verify the C++ loader round-trips the converter output.
#include "mg-model.hpp"
#include "mg-tensor.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace mg;

static double checksum(const Tensor* t) {
    double s = 0.0;
    int64_t n = t->nelements();
    const float* d = static_cast<const float*>(t->data);
    for (int64_t i = 0; i < n; i++) s += d[i];
    return s;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
    Context ctx(8 << 20);  // metadata only; weights are mmap'd externals
    auto model = Model::load(argv[1], ctx);
    const HParams& h = model->hparams();

    std::printf("arch=%s  name=%s  resolution=%d\n", h.arch.c_str(), h.name.c_str(), h.resolution);
    std::printf("transformer: n_layer=%d n_head=%d n_embd=%d n_ffn=%d head_dim=%d\n",
                h.n_layer, h.n_head, h.n_embd, h.n_ffn, h.head_dim);
    std::printf("             vocab=%d n_tokens=%d n_positions=%d mask_token_id=%d ln_eps=%g\n",
                h.vocab_size, h.n_tokens, h.n_positions, h.mask_token_id, h.ln_eps);
    std::printf("vqgan: codebook=%d embd=%d filters=%d res_blocks=%d gn_groups=%d mult=[",
                h.vq_codebook_size, h.vq_embedding_dim, h.vq_filters, h.vq_num_res_blocks, h.vq_gn_groups);
    for (size_t i = 0; i < h.vq_channel_mult.size(); i++)
        std::printf("%d%s", h.vq_channel_mult[i], i + 1 < h.vq_channel_mult.size() ? "," : "");
    std::printf("]  choice_temp=%g\n", h.choice_temperature);
    std::printf("tensors: %zu\n", model->n_tensors());

    const char* spot[] = {
        "token_embd.weight", "pos_embd.weight", "blk.0.attn_q.weight",
        "blk.23.ffn_down.weight", "output.bias",
        "vqgan.quantizer.codebook.weight", "vqgan.decoder.conv_out.weight",
    };
    std::printf("checksums (float64 sum):\n");
    for (const char* nm : spot) {
        Tensor* t = model->get(nm);
        if (!t) { std::printf("    %-42s MISSING\n", nm); continue; }
        std::printf("    %-42s ne=[%lld,%lld,%lld,%lld] sum=%.6f\n", nm,
                    (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                    checksum(t));
    }
    return 0;
}
