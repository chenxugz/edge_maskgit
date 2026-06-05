// verify-opencl-transformer.cpp — run the full transformer forward graph on the
// GPU (OpenCL backend) and compare logits to the PyTorch reference dump.
#include "mg-model.hpp"
#include "mg-opencl.hpp"
#include "mg-tensor.hpp"
#include "mg-transformer.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>   // mkdir
#include <vector>

using namespace mg;

template <class T>
static std::vector<T> read_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
    auto n = f.tellg(); f.seekg(0);
    std::vector<T> v(n / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), n);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf export_dir [--dump-dir DIR]\n", argv[0]);
        return 2;
    }
    std::string gguf = argv[1], ed = argv[2];
    std::string dump_dir;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dump-dir" && i+1 < argc) dump_dir = argv[++i];
    }

    Context wctx(64 << 20);
    auto model = Model::load(gguf, wctx);
    std::vector<int32_t> tokens = read_bin<int32_t>(ed + "/verify_tokens.bin");
    std::vector<float>   ref    = read_bin<float>(ed + "/verify_logits.bin");
    const int S = (int)tokens.size();
    const int V = model->hparams().vocab_size;

    Context ctx(2ull << 30);
    Tensor* tok = ctx.tensor2d(Type::I32, S, 1);
    std::memcpy(tok->data, tokens.data(), tokens.size() * sizeof(int32_t));
    Tensor* logits = build_transformer(ctx, *model, tok);   // {vocab, S, 1}
    Graph g; g.build_forward(logits);

    OpenCLRuntime ocl;
    std::printf("GPU device: %s   graph nodes=%zu\n", ocl.device_name().c_str(), g.nodes.size());
    auto t0 = std::chrono::steady_clock::now();
    ocl.compute(g);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // ---- M3 per-layer dump: write every named graph node as raw float32 ----
    // The naming convention lives in build_transformer() (embd_post_norm,
    // blk.{i}.{attn,ffn}_post, output_norm, output_logits). The PyTorch oracle
    // dumps the same names — verification/compare_intermediates.py walks both
    // and computes per-tensor cosine / max_abs_diff.
    if (!dump_dir.empty()) {
        // mkdir -p (only the leaf — caller is expected to put it under
        // verification/dumps/<run-name>).
        ::mkdir(dump_dir.c_str(), 0755);
        int n_dumped = 0;
        for (Tensor* t : g.nodes) {
            if (t->name.empty()) continue;
            // Tensors that are views (Permute/Reshape/View/Cont) share storage
            // with their src; the kernel writes data through the underlying buffer
            // already. Skip view nodes; we want the materialized post-kernel data.
            if (t->op == Op::Reshape || t->op == Op::Permute || t->op == Op::View) continue;
            std::string path = dump_dir + "/" + t->name + ".bin";
            std::ofstream f(path, std::ios::binary);
            if (!f) { std::fprintf(stderr, "dump: cannot open %s\n", path.c_str()); continue; }
            f.write(static_cast<const char*>(t->data), t->nbytes());
            n_dumped++;
        }
        std::printf("dumped %d named tensors → %s\n", n_dumped, dump_dir.c_str());
    }

    const float* o = static_cast<const float*>(logits->data);
    double max_abs = 0, sum_abs = 0, dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < (int64_t)S * V; i++) {
        float a = o[i], b = ref[i];
        double d = std::fabs(a - b);
        max_abs = std::fmax(max_abs, d); sum_abs += d;
        dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    std::printf("opencl transformer: forward=%.3fs  max_abs_diff=%.4e mean_abs_diff=%.4e cosine=%.8f\n",
                secs, max_abs, sum_abs / ((double)S * V), cos);
    bool quantized = model->hparams().quant != "f32";   // ggml Q8_0 etc. diverge in magnitude
    double max_tol = quantized ? 1.0 : 2e-2;
    double cos_tol = quantized ? 0.9999 : 0.99999;
    bool ok = max_abs < max_tol && cos > cos_tol;
    std::printf("quant=%s  %s\n", model->hparams().quant.c_str(),
                ok ? "OPENCL TRANSFORMER VERIFY: PASS" : "OPENCL TRANSFORMER VERIFY: FAIL");
    return ok ? 0 : 1;
}
