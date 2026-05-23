// verify-vqgan.cpp — run the C++ VQGAN decoder on the fixed reference token grid
// and compare the decoded image against the PyTorch dump.
#include "mg-model.hpp"
#include "mg-tensor.hpp"
#include "mg-vqgan.hpp"

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
    std::string gguf = argv[1], ed = argv[2];

    Context wctx(64 << 20);
    auto model = Model::load(gguf, wctx);

    std::vector<int32_t> grid = read_bin<int32_t>(ed + "/verify_grid.bin");   // [16*16] row-major
    std::vector<float>   ref  = read_bin<float>(ed + "/verify_image.bin");    // [H,W,3]
    const int64_t n = (int64_t)grid.size();
    const int64_t HW = 256;
    std::printf("grid tokens=%lld  ref image floats=%zu (expect %lld)\n",
                (long long)n, ref.size(), (long long)(HW * HW * 3));

    Context ctx(4ull << 30);                       // 4 GB activation arena
    Tensor* ids = ctx.tensor1d(Type::I32, n);
    std::memcpy(ids->data, grid.data(), grid.size() * sizeof(int32_t));

    Tensor* img = build_vqgan_decoder(ctx, *model, ids);   // {W,H,3,1}
    Graph g; g.build_forward(img);
    std::printf("graph nodes=%zu, computing...\n", g.nodes.size());
    compute(ctx, g);

    // ours: contiguous {W,H,3,1} -> elem(w,h,c) = o[w + W*(h + H*c)]
    // ref:  [H,W,3] row-major   -> elem(h,w,c) = ref[(h*W + w)*3 + c]
    const float* o = static_cast<const float*>(img->data);
    const int64_t W = img->ne[0], H = img->ne[1];
    double max_abs = 0, sum_abs = 0, dot = 0, na = 0, nb = 0;
    for (int64_t h = 0; h < H; h++)
        for (int64_t w = 0; w < W; w++)
            for (int64_t cch = 0; cch < 3; cch++) {
                float a = o[w + W * (h + H * cch)];
                float b = ref[(h * W + w) * 3 + cch];
                double d = std::fabs(a - b);
                max_abs = std::fmax(max_abs, d); sum_abs += d;
                dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
            }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    int64_t total = H * W * 3;
    std::printf("image: max_abs_diff=%.4e  mean_abs_diff=%.4e  cosine=%.8f\n",
                max_abs, sum_abs / total, cos);
    bool ok = max_abs < 1e-2 && cos > 0.99999;
    std::printf("%s\n", ok ? "VQGAN VERIFY: PASS" : "VQGAN VERIFY: FAIL");
    return ok ? 0 : 1;
}
