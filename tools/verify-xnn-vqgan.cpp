// verify-xnn-vqgan.cpp — XNNPACK VQGAN decoder vs PyTorch reference image.
#include "mg-model.hpp"
#include "mg-tensor.hpp"
#include "mg-xnn-vqgan.hpp"

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
    if (argc < 3) { std::fprintf(stderr, "usage: %s model.gguf export_dir [f32|q8]\n", argv[0]); return 2; }
    std::string gguf = argv[1], ed = argv[2];
    std::string qs = argc > 3 ? argv[3] : "f32";
    Quant quant = (qs == "q8" || qs == "q4") ? Quant::Q8 : Quant::F32;  // conv: qc8w only

    Context wctx(64 << 20);
    auto model = Model::load(gguf, wctx);

    std::vector<int32_t> grid = read_bin<int32_t>(ed + "/verify_grid.bin");   // [16*16]
    std::vector<float>   ref  = read_bin<float>(ed + "/verify_image.bin");    // [H,W,3] HWC

    std::printf("quant = %s\n", qs.c_str());
    auto t0 = std::chrono::steady_clock::now();
    XnnVqgan vq(*model, quant);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> img((size_t)vq.height() * vq.width() * 3);
    vq.decode(grid.data(), img.data());
    auto t2 = std::chrono::steady_clock::now();

    double max_abs = 0, sum_abs = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < img.size(); i++) {
        float a = img[i], b = ref[i];
        double d = std::fabs(a - b);
        max_abs = std::fmax(max_abs, d); sum_abs += d;
        dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    std::printf("xnn vqgan: build=%.3fs decode=%.3fs  (%dx%d)\n",
                std::chrono::duration<double>(t1-t0).count(),
                std::chrono::duration<double>(t2-t1).count(), vq.width(), vq.height());
    std::printf("image: max_abs_diff=%.4e mean_abs_diff=%.4e cosine=%.8f\n",
                max_abs, sum_abs / img.size(), cos);
    double max_tol = quant == Quant::F32 ? 2e-2 : 0.4;   // int8 conv diverges more in magnitude
    double cos_tol = quant == Quant::F32 ? 0.99999 : 0.999;
    bool ok = max_abs < max_tol && cos > cos_tol;
    std::printf("%s\n", ok ? "XNN VQGAN VERIFY: PASS" : "XNN VQGAN VERIFY: FAIL");
    return ok ? 0 : 1;
}
