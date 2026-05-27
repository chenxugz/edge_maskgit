// verify-opencl-vqgan.cpp — run the VQGAN decoder forward graph on the GPU
// (OpenCL backend) and compare the decoded image to the PyTorch reference dump.
#include "mg-model.hpp"
#include "mg-opencl.hpp"
#include "mg-tensor.hpp"
#include "mg-vqgan.hpp"

#include <chrono>
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
static std::vector<T> read_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(2); }
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
    std::vector<int32_t> grid = read_bin<int32_t>(ed + "/verify_grid.bin");
    std::vector<float>   ref  = read_bin<float>(ed + "/verify_image.bin");   // [H,W,3] HWC

    Context ctx(4ull << 30);
    Tensor* ids = ctx.tensor1d(Type::I32, (int64_t)grid.size());
    std::memcpy(ids->data, grid.data(), grid.size() * sizeof(int32_t));
    Tensor* img = build_vqgan_decoder(ctx, *model, ids);    // {W,H,3,1}
    Graph g; g.build_forward(img);

    OpenCLRuntime ocl;
    std::printf("GPU device: %s   graph nodes=%zu\n", ocl.device_name().c_str(), g.nodes.size());
    auto t0 = std::chrono::steady_clock::now();
    ocl.compute(g);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    const float* o = static_cast<const float*>(img->data);
    const int64_t W = img->ne[0], H = img->ne[1];
    double max_abs = 0, sum_abs = 0, dot = 0, na = 0, nb = 0;
    for (int64_t h = 0; h < H; h++)
        for (int64_t w = 0; w < W; w++)
            for (int c = 0; c < 3; c++) {
                float a = o[w + W * (h + H * c)];          // ours {W,H,C}
                float b = ref[(h * W + w) * 3 + c];        // ref [H,W,3]
                double d = std::fabs(a - b);
                max_abs = std::fmax(max_abs, d); sum_abs += d;
                dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    std::printf("opencl vqgan: decode=%.3fs  max_abs_diff=%.4e mean_abs_diff=%.4e cosine=%.8f\n",
                secs, max_abs, sum_abs / ((double)H * W * 3), cos);
    // Tolerance covers the default int8 VQGAN conv (arm_dot, per-block activation quant:
    // cos ~0.99997, max_abs ~0.06). The F32 conv path (MG_NO_ARM_CONV=1, or non-arm_dot
    // devices) is exact (cos 1.0, max_abs ~1e-4). Both pass.
    bool ok = max_abs < 0.1 && cos > 0.9999;
    std::printf("%s\n", ok ? "OPENCL VQGAN VERIFY: PASS" : "OPENCL VQGAN VERIFY: FAIL");
    return ok ? 0 : 1;
}
