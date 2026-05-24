// test-opencl.cpp — validate the OpenCL graph executor against the reference CPU
// backend on an FFN-shaped graph (mul_mat -> +bias -> gelu -> mul_mat -> +bias).
#include "mg-opencl.hpp"
#include "mg-tensor.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace mg;

static void fill(Tensor* t, float ph) {
    int64_t n = t->nelements();
    for (int64_t i = 0; i < n; i++) t->set_f32(i, std::sin(0.05f * i + ph) * 0.5f);
}

int main() {
    Context c(256 << 20);
    const int K = 768, S = 16, F = 3072;
    Tensor* x  = c.tensor2d(Type::F32, K, S); fill(x, 0.1f);
    Tensor* W1 = c.tensor2d(Type::F32, K, F); fill(W1, 0.3f);   // {K,F}
    Tensor* b1 = c.tensor1d(Type::F32, F);    fill(b1, 0.7f);
    Tensor* W2 = c.tensor2d(Type::F32, F, K); fill(W2, 0.9f);   // {F,K}
    Tensor* b2 = c.tensor1d(Type::F32, K);    fill(b2, 1.3f);

    Tensor* h = add_bias(c, mul_mat(c, W1, x), b1);            // {F,S}
    Tensor* g = gelu(c, h);
    Tensor* o = add_bias(c, mul_mat(c, W2, g), b2);            // {K,S}

    // reference
    Graph gr; gr.build_forward(o); compute(c, gr);
    std::vector<float> ref(o->nelements());
    for (int64_t i = 0; i < o->nelements(); i++) ref[i] = o->f32(i);

    // opencl (re-reads the same leaves, recomputes into o->data)
    OpenCLRuntime ocl;
    std::printf("GPU device: %s\n", ocl.device_name().c_str());
    Graph gg; gg.build_forward(o); ocl.compute(gg);

    double max_abs = 0, sum_abs = 0, dot = 0, na = 0, nb = 0;
    for (int64_t i = 0; i < o->nelements(); i++) {
        float a = o->f32(i), b = ref[i];
        double d = std::fabs(a - b);
        max_abs = std::fmax(max_abs, d); sum_abs += d;
        dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));
    std::printf("opencl vs reference: max_abs_diff=%.3e mean_abs_diff=%.3e cosine=%.8f\n",
                max_abs, sum_abs / o->nelements(), cos);
    bool ok = max_abs < 1e-3 && cos > 0.999999;
    std::printf("%s\n", ok ? "OPENCL GRAPH VERIFY: PASS" : "OPENCL GRAPH VERIFY: FAIL");
    return ok ? 0 : 1;
}
