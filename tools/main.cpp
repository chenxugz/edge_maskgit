// main.cpp — mg-generate CLI.
//   ./mg-generate -m model.gguf --class-id 207 --steps 8 --seed 42 -o out.png
#include "mg-generate.hpp"
#include "mg-model.hpp"
#include "mg-tensor.hpp"
#include "mg-transformer.hpp"
#include "mg-vqgan.hpp"
#include "mg-xnn.hpp"
#include "mg-xnn-vqgan.hpp"
#ifdef MG_HAS_OPENCL
#include "mg-opencl.hpp"
#endif

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/resource.h>

static double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    return ru.ru_maxrss / (1024.0 * 1024.0);   // bytes
#else
    return ru.ru_maxrss / 1024.0;              // Linux/Android: kilobytes
#endif
}

using namespace mg;

static void usage(const char* p) {
    std::fprintf(stderr,
        "usage: %s -m model.gguf [--class-id N] [--steps N] [--seed N]\n"
        "          [--temperature F] [-o out.png]\n", p);
}

int main(int argc, char** argv) {
    std::string model_path, out = "output.png", backend = "reference", quant = "";  // ""=from model
    GenConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--class-id") cfg.class_id = std::atoi(next().c_str());
        else if (a == "--steps") cfg.steps = std::atoi(next().c_str());
        else if (a == "--seed") cfg.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--temperature") cfg.temperature = std::atof(next().c_str());
        else if (a == "--backend") backend = next();   // reference | xnnpack
        else if (a == "--quant") quant = next();        // f32 | q8 | q4 (xnnpack only)
        else if (a == "-o" || a == "--output") out = next();
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty()) { usage(argv[0]); return 2; }
    if (cfg.class_id < 0 || cfg.class_id > 999) {
        std::fprintf(stderr, "--class-id must be in [0,999]\n"); return 2;
    }

    Context wctx(64 << 20);
    auto model = Model::load(model_path, wctx);
    std::printf("[mg-generate] model: %s  class=%d steps=%d seed=%llu temp=%.2f\n",
                model->hparams().name.c_str(), cfg.class_id, cfg.steps,
                (unsigned long long)cfg.seed,
                cfg.temperature < 0 ? model->hparams().choice_temperature : cfg.temperature);

    // Optional accelerated backends (transformer + VQGAN forward overrides).
    mg::TransformerFwd fwd = nullptr;
    mg::VqganFwd vfwd = nullptr;
    std::unique_ptr<XnnTransformer> xt;
    std::unique_ptr<XnnVqgan> xv;
#ifdef MG_HAS_OPENCL
    std::unique_ptr<mg::OpenCLRuntime> ocl;
    std::unique_ptr<Context> octx;        // transformer scratch, reset each step
#endif
    if (backend == "opencl") {
#ifdef MG_HAS_OPENCL
        const auto& hp = model->hparams();
        const int S = hp.n_tokens + 1;
        const int64_t vocab = hp.vocab_size;
        ocl  = std::make_unique<mg::OpenCLRuntime>();
        octx = std::make_unique<Context>(1536ull << 20);
        std::printf("[mg-generate] backend: OpenCL (%s)  quant from model=%s\n",
                    ocl->device_name().c_str(), hp.quant.c_str());
        // Transformer forward on GPU: weights upload once (cached by Tensor*); the
        // scratch arena is reset each step so tensor addresses are stable and only
        // the changed token leaf is re-uploaded (invalidate).
        fwd = [&, S, vocab](const int32_t* toks, float* out) {
            octx->reset();
            Tensor* tok = octx->tensor2d(Type::I32, S, 1);
            std::memcpy(tok->data, toks, S * sizeof(int32_t));
            Tensor* logits = build_transformer(*octx, *model, tok);   // {vocab,S,1}
            ocl->invalidate(tok);
            Graph g; g.build_forward(logits); ocl->compute(g);
            std::memcpy(out, logits->data, (size_t)S * vocab * sizeof(float));
        };
        // VQGAN decode on GPU (convs stay F32 in gq8/gq4 files); runs once.
        vfwd = [&](const int32_t* grid, float* hwc) {
            const int n_tok = model->hparams().n_tokens;
            Context vctx(3ull << 30);
            Tensor* gt = vctx.tensor1d(Type::I32, n_tok);
            std::memcpy(gt->data, grid, n_tok * sizeof(int32_t));
            Tensor* imgT = build_vqgan_decoder(vctx, *model, gt);     // {W,H,3,1}
            Graph g; g.build_forward(imgT); ocl->compute(g);
            const int64_t W = imgT->ne[0], H = imgT->ne[1];
            const float* o = static_cast<const float*>(imgT->data);
            for (int64_t y = 0; y < H; y++)
                for (int64_t x = 0; x < W; x++)
                    for (int c = 0; c < 3; c++)
                        hwc[(y * W + x) * 3 + c] = o[x + W * (y + H * c)];  // {W,H,C}->HWC
        };
#else
        std::fprintf(stderr, "this build has no OpenCL backend (rebuild with MG_HAS_OPENCL)\n");
        return 2;
#endif
    } else if (backend == "xnnpack") {
        if (quant.empty()) quant = model->hparams().quant;   // default to the model's quant level
        mg::Quant q = quant == "q8" ? mg::Quant::Q8 : (quant == "q4" ? mg::Quant::Q4 : mg::Quant::F32);
        xt = std::make_unique<XnnTransformer>(*model, /*batch=*/1, model->hparams().n_tokens + 1, q);
        xv = std::make_unique<XnnVqgan>(*model, q);   // conv uses int8 whenever q != F32
        fwd  = [&](const int32_t* toks, float* out) { xt->forward(toks, out); };
        vfwd = [&](const int32_t* grid, float* img) { xv->decode(grid, img); };
        std::printf("[mg-generate] backend: XNNPACK (transformer quant=%s, VQGAN conv quant=%s)\n",
                    quant.c_str(), quant == "f32" ? "f32" : "q8");
    } else if (backend != "reference") {
        std::fprintf(stderr, "unknown --backend '%s' (use reference|xnnpack|opencl)\n", backend.c_str());
        return 2;
    }

    auto t0 = std::chrono::steady_clock::now();
    Image img = generate(*model, cfg, /*verbose=*/true, fwd, vfwd);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    if (!stbi_write_png(out.c_str(), img.width, img.height, 3, img.rgb.data(), img.width * 3)) {
        std::fprintf(stderr, "failed to write %s\n", out.c_str()); return 1;
    }
    std::printf("[mg-generate] wrote %s (%dx%d) in %.1fs  peak_rss=%.0f MB\n",
                out.c_str(), img.width, img.height, secs, peak_rss_mb());
    return 0;
}
