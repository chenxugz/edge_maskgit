// main.cpp — mg-generate CLI.
//   ./mg-generate -m model.gguf --class-id 207 --steps 8 --seed 42 -o out.png
#include "mg-generate.hpp"
#include "mg-model.hpp"
#include "mg-tensor.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace mg;

static void usage(const char* p) {
    std::fprintf(stderr,
        "usage: %s -m model.gguf [--class-id N] [--steps N] [--seed N]\n"
        "          [--temperature F] [-o out.png]\n", p);
}

int main(int argc, char** argv) {
    std::string model_path, out = "output.png";
    GenConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "--class-id") cfg.class_id = std::atoi(next().c_str());
        else if (a == "--steps") cfg.steps = std::atoi(next().c_str());
        else if (a == "--seed") cfg.seed = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--temperature") cfg.temperature = std::atof(next().c_str());
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

    auto t0 = std::chrono::steady_clock::now();
    Image img = generate(*model, cfg, /*verbose=*/true);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();

    if (!stbi_write_png(out.c_str(), img.width, img.height, 3, img.rgb.data(), img.width * 3)) {
        std::fprintf(stderr, "failed to write %s\n", out.c_str()); return 1;
    }
    std::printf("[mg-generate] wrote %s (%dx%d) in %.1fs\n", out.c_str(), img.width, img.height, secs);
    return 0;
}
