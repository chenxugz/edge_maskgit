// verify-xnn-transformer.cpp — XNNPACK transformer subgraph vs PyTorch reference logits.
#include "mg-model.hpp"
#include "mg-tensor.hpp"
#include "mg-xnn.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    std::string gguf = argv[1], ed = argv[2];

    Context wctx(64 << 20);
    auto model = Model::load(gguf, wctx);

    std::vector<int32_t> tokens = read_bin<int32_t>(ed + "/verify_tokens.bin");
    std::vector<float>   ref    = read_bin<float>(ed + "/verify_logits.bin");
    const int S = (int)tokens.size();
    const int V = model->hparams().vocab_size;

    auto t0 = std::chrono::steady_clock::now();
    XnnTransformer xt(*model, /*batch=*/1, S);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> logits((size_t)S * V);
    xt.forward(tokens.data(), logits.data());
    auto t2 = std::chrono::steady_clock::now();

    double build_s = std::chrono::duration<double>(t1 - t0).count();
    double fwd_s   = std::chrono::duration<double>(t2 - t1).count();

    double max_abs = 0, sum_abs = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < logits.size(); i++) {
        float a = logits[i], b = ref[i];
        double d = std::fabs(a - b);
        max_abs = std::fmax(max_abs, d); sum_abs += d;
        dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    std::printf("xnn transformer: build=%.3fs forward=%.3fs\n", build_s, fwd_s);
    std::printf("logits: max_abs_diff=%.4e mean_abs_diff=%.4e cosine=%.8f\n",
                max_abs, sum_abs / logits.size(), cos);
    bool ok = max_abs < 5e-2 && cos > 0.99999;
    std::printf("%s\n", ok ? "XNN TRANSFORMER VERIFY: PASS" : "XNN TRANSFORMER VERIFY: FAIL");
    return ok ? 0 : 1;
}
