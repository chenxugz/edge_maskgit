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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
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
    bool bench = false; int n_runs = 10, warmup = 2;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--class-id") cfg.class_id = std::atoi(next().c_str());
        else if (a == "--steps") cfg.steps = std::atoi(next().c_str());
        else if (a == "--seed") cfg.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--temperature") cfg.temperature = std::atof(next().c_str());
        else if (a == "--backend") backend = next();   // reference | xnnpack | opencl
        else if (a == "--quant") quant = next();        // f32 | q8 | q4 (xnnpack only)
        else if (a == "-o" || a == "--output") out = next();
        else if (a == "--bench") bench = true;          // benchmark mode (M5)
        else if (a == "--n-runs") n_runs = std::atoi(next().c_str());
        else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }
    if (model_path.empty()) { usage(argv[0]); return 2; }
    if (cfg.class_id < 0 || cfg.class_id > 999) {
        std::fprintf(stderr, "--class-id must be in [0,999]\n"); return 2;
    }

    auto load0 = std::chrono::steady_clock::now();
    Context wctx(64 << 20);
    auto model = Model::load(model_path, wctx);
    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load0).count();
    std::printf("[mg-generate] model: %s  class=%d steps=%d seed=%llu temp=%.2f\n",
                model->hparams().name.c_str(), cfg.class_id, cfg.steps,
                (unsigned long long)cfg.seed,
                cfg.temperature < 0 ? model->hparams().choice_temperature : cfg.temperature);

    // Optional accelerated backends (transformer + VQGAN forward overrides).
    mg::TransformerFwd fwd = nullptr;
    mg::VqganFwd vfwd = nullptr;
    // Called once after the 8-step decoding loop finishes (before VQGAN). Used by
    // the OpenCL backend to drop the transformer scratch arena (~500 MB) before
    // VQGAN allocates its own ~1 GB working set. Cost: ~10-30 ms munmap, <0.5%
    // of end-to-end. nullptr for backends that don't need it (XNNPACK manages
    // its own memory via XNNPACK's allocator).
    mg::OnTransformerDone on_done = nullptr;
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
        // Zero-copy weight upload (Mali cl_arm_import_memory). When supported the
        // GPU reads weights directly from the mmap'd GGUF — saves ~3-5 s of cold
        // start per process. The fallback path (clEnqueueWriteBuffer) is hit
        // tensor-by-tensor on devices/regions where import fails, so this is safe
        // to always call.
        if (model->mmap_base() && ocl->import_host_region(model->mmap_base(), model->mmap_size()))
            std::printf("[mg-generate] zero-copy weight upload via cl_arm_import_memory\n");
        // Transformer scratch arena: bump-allocator with no per-tensor free, so the
        // whole graph (all 24 layers' Q/K/V/scores/softmax/FFN) lives in here at once.
        // Attention scores are the M²·heads·layers term; M-linear tensors (Q/K/V/FFN/
        // residual) dominate at small M. Use 1.5 GB baseline + 3× the scores term.
        size_t arena_bytes = (size_t)1536 * 1024 * 1024
            + (size_t)S * (size_t)S * (size_t)hp.n_head * (size_t)hp.n_layer * 4 * 3;
        octx = std::make_unique<Context>(arena_bytes);
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
        // After the 8-step transformer loop completes, drop the scratch arena —
        // VQGAN is about to allocate its own and we don't need the transformer
        // intermediates anymore. Frees ~400-600 MB of touched pages back to OS.
        // Weight cl_mems stay cached on the GPU (their Tensor*s live in wctx).
        on_done = [&]() { octx.reset(); };
        // VQGAN decode on GPU (convs stay F32 in gq8/gq4 files); runs once.
        vfwd = [&](const int32_t* grid, float* hwc) {
            const int n_tok = model->hparams().n_tokens;
            // VQGAN arena: 3 GB fits n_tokens=256 (the production size). Larger M
            // needs more but the device can't hold it alongside the transformer arena;
            // for the seq-len bench, set MG_BENCH_SKIP_VQGAN=1 to skip this leg.
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

    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    // ---- Benchmark mode (M5) -------------------------------------------------
    if (bench) {
        const char* dev = "host CPU";
#ifdef MG_HAS_OPENCL
        if (backend == "opencl" && ocl) dev = ocl->device_name().c_str();
#endif
        std::printf("[bench] backend=%s quant=%s device=\"%s\" steps=%d  warmup=%d runs=%d\n",
                    backend.c_str(), (quant.empty()?model->hparams().quant:quant).c_str(),
                    dev, cfg.steps, warmup, n_runs);

        // Env-toggled VQGAN skip: at very large M the VQGAN scratch arena overflows
        // device RAM. The seq-len sweep only cares about transformer scaling, so we
        // replace vfwd with a no-op (the per-component breakdown still reports the
        // transformer total accurately; the "VQGAN decode" row drops to ~0).
        if (std::getenv("MG_BENCH_SKIP_VQGAN")) {
            vfwd = [](const int32_t*, float*) { /* no-op for seq-len bench */ };
            std::printf("[bench] MG_BENCH_SKIP_VQGAN set: VQGAN decode skipped\n");
        }

        for (int i = 0; i < warmup; i++) generate(*model, cfg, false, fwd, vfwd);

        std::vector<double> e2e;            // per-run end-to-end ms (no profiling -> real latency)
        GenStats agg{};
        for (int r = 0; r < n_runs; r++) {
            GenStats st{};
            auto b0 = clk::now();
            generate(*model, cfg, false, fwd, vfwd, &st);
            e2e.push_back(ms(b0, clk::now()));
            agg.transformer_ms += st.transformer_ms; agg.sampling_ms += st.sampling_ms;
            agg.vqgan_ms += st.vqgan_ms;
        }
        std::sort(e2e.begin(), e2e.end());
        int n = (int)e2e.size();
        auto pct = [&](double p){ return e2e[std::min(n-1, (int)(p*n))]; };
        double mean = 0; for (double v : e2e) mean += v; mean /= n;
        double var = 0; for (double v : e2e) var += (v-mean)*(v-mean); var /= n;

        std::printf("\n### Benchmark: %s / %s / %s\n\n", backend.c_str(),
                    (quant.empty()?model->hparams().quant:quant).c_str(), dev);
        std::printf("| metric | value |\n|---|---|\n");
        std::printf("| model load | %.0f ms |\n", load_ms);
        std::printf("| end-to-end p50 | %.0f ms |\n", pct(0.50));
        std::printf("| end-to-end p90 | %.0f ms |\n", pct(0.90));
        std::printf("| end-to-end p99 | %.0f ms |\n", pct(0.99));
        std::printf("| end-to-end mean +/- sd | %.0f +/- %.0f ms |\n", mean, std::sqrt(var));
        std::printf("| end-to-end min | %.0f ms |\n", e2e.front());
        std::printf("| peak RSS | %.0f MB |\n", peak_rss_mb());
        std::printf("\n**Component breakdown (mean per run, %d steps):**\n\n", cfg.steps);
        std::printf("| component | ms | %% |\n|---|--:|--:|\n");
        double tt = (agg.transformer_ms + agg.sampling_ms + agg.vqgan_ms) / n;
        std::printf("| transformer (x%d) | %.0f | %.0f%% |\n", cfg.steps,
                    agg.transformer_ms/n, 100*agg.transformer_ms/(tt*n>0?tt*n:1));
        std::printf("| sampling/masking | %.0f | %.0f%% |\n",
                    agg.sampling_ms/n, 100*agg.sampling_ms/(tt*n>0?tt*n:1));
        std::printf("| VQGAN decode | %.0f | %.0f%% |\n",
                    agg.vqgan_ms/n, 100*agg.vqgan_ms/(tt*n>0?tt*n:1));

#ifdef MG_HAS_OPENCL
        if (backend == "opencl" && ocl) {   // one extra serialized run for the per-op split
            ocl->profile_reset(); ocl->profile_enable(true);
            generate(*model, cfg, false, fwd, vfwd);
            ocl->profile_enable(false);
            std::printf("\n**GPU per-op-type profile (1 run, clFinish-serialized; relative split is the signal):**\n\n");
            std::printf("| op | dispatches | ms | %% |\n|---|--:|--:|--:|\n");
            auto rep = ocl->profile_report();
            double tot = 0; for (auto& e : rep) tot += e.ms;
            for (auto& e : rep)
                std::printf("| %s | %d | %.0f | %.0f%% |\n", e.op.c_str(), e.count, e.ms,
                            100*e.ms/(tot>0?tot:1));
        }
#endif
        if (backend == "xnnpack" && xt) {   // XNNPACK per-op profile (one extra run; XNN_FLAG_BASIC_PROFILING)
            xt->profile_reset(); xv->profile_reset();
            xt->profile_enable(true); xv->profile_enable(true);
            generate(*model, cfg, false, fwd, vfwd);
            xt->profile_enable(false); xv->profile_enable(false);
            auto rt = xt->profile_report(); auto rv = xv->profile_report();
            // merge transformer + VQGAN buckets
            std::unordered_map<std::string,double> ms; std::unordered_map<std::string,int> cnt;
            for (auto& e : rt) { ms[e.op]+=e.ms; cnt[e.op]+=e.count; }
            for (auto& e : rv) { ms[e.op]+=e.ms; cnt[e.op]+=e.count; }
            std::vector<std::pair<std::string,double>> r;
            for (auto& kv : ms) r.push_back({kv.first, kv.second});
            std::sort(r.begin(), r.end(), [](auto& a, auto& b){ return a.second > b.second; });
            double tot = 0; for (auto& e : r) tot += e.second;
            std::printf("\n**CPU per-op-type profile (1 run, XNNPACK XNN_FLAG_BASIC_PROFILING, transformer + VQGAN):**\n\n");
            std::printf("| op | ops | ms | %% |\n|---|--:|--:|--:|\n");
            for (auto& e : r)
                std::printf("| %s | %d | %.0f | %.0f%% |\n", e.first.c_str(), cnt[e.first], e.second,
                            100*e.second/(tot>0?tot:1));
        }
        std::printf("\n");
        return 0;
    }

    auto t0 = clk::now();
    // Production single-image path: pass on_done so the OpenCL backend drops the
    // ~500 MB transformer scratch arena before VQGAN runs. Bench mode (above)
    // intentionally doesn't pass it — those generate() loops reuse octx across
    // iterations, and recreating it each call would muddy the timing.
    Image img = generate(*model, cfg, /*verbose=*/true, fwd, vfwd, /*stats=*/nullptr, on_done);
    double secs = ms(t0, clk::now()) / 1000.0;

    if (!stbi_write_png(out.c_str(), img.width, img.height, 3, img.rgb.data(), img.width * 3)) {
        std::fprintf(stderr, "failed to write %s\n", out.c_str()); return 1;
    }
    std::printf("[mg-generate] wrote %s (%dx%d) in %.1fs  peak_rss=%.0f MB\n",
                out.c_str(), img.width, img.height, secs, peak_rss_mb());
    return 0;
}
