// verify-transformer.cpp — run the C++ transformer on the fixed reference input
// and compare logits against the PyTorch dump (reference/export/verify_*.bin).
#include "mg-model.hpp"
#include "mg-tensor.hpp"
#include "mg-transformer.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace mg;

template <class T>
static std::vector<T> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    auto n = f.tellg(); f.seekg(0);
    std::vector<T> v(n / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf export_dir\n", argv[0]); return 2; }
    std::string gguf = argv[1], ed = std::string(argv[2]);

    Context wctx(64 << 20);
    auto model = Model::load(gguf, wctx);
    const auto& h = model->hparams();

    std::vector<int32_t> tokens = read_bin<int32_t>(ed + "/verify_tokens.bin");
    std::vector<float>   ref    = read_bin<float>(ed + "/verify_logits.bin");
    const int64_t S = (int64_t)tokens.size();
    const int64_t V = h.vocab_size;
    std::printf("S=%lld vocab=%lld  (ref logits=%zu, expect %lld)\n",
                (long long)S, (long long)V, ref.size(), (long long)(S * V));

    Context ctx(1ull << 30);                 // 1 GB activation arena
    Tensor* tok = ctx.tensor2d(Type::I32, S, 1);
    std::memcpy(tok->data, tokens.data(), tokens.size() * sizeof(int32_t));

    Tensor* logits = build_transformer(ctx, *model, tok);   // {vocab, S, 1}
    Graph g; g.build_forward(logits);
    std::printf("graph nodes=%zu, building+computing...\n", g.nodes.size());
    compute(ctx, g);

    // compare: ref is [S, V] row-major (s outer, v inner); ours is {V,S} -> flat s*V+v
    double max_abs = 0, sum_abs = 0, dot = 0, na = 0, nb = 0;
    const float* o = static_cast<const float*>(logits->data);
    for (int64_t s = 0; s < S; s++)
        for (int64_t v = 0; v < V; v++) {
            float a = o[s * V + v];        // ours (contiguous {V,S})
            float b = ref[s * V + v];
            double d = std::fabs(a - b);
            max_abs = std::fmax(max_abs, d); sum_abs += d;
            dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
        }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    std::printf("logits: max_abs_diff=%.4e  mean_abs_diff=%.4e  cosine=%.8f\n",
                max_abs, sum_abs / (S * V), cos);
    bool ok = max_abs < 2e-2 && cos > 0.99999;
    std::printf("%s\n", ok ? "TRANSFORMER VERIFY: PASS" : "TRANSFORMER VERIFY: FAIL");
    return ok ? 0 : 1;
}
