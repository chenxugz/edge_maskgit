// mg-generate.hpp — iterative masked decoding (host-side) + image decode.
#pragma once
#include "mg-model.hpp"
#include "mg-tensor.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace mg {

// Optional transformer forward override: tokens[B*S] (int32) -> logits[B*S*vocab]
// (row-major s-outer, v-inner). When set, generate() uses it instead of the
// reference graph (e.g. the XNNPACK subgraph). Keeps mg-core XNNPACK-free.
using TransformerFwd = std::function<void(const int32_t*, float*)>;
// Optional VQGAN decode override: grid[n_tokens] (int32) -> image[H*W*3] HWC float.
using VqganFwd = std::function<void(const int32_t*, float*)>;
// Called once after the iterative-decoding loop finishes, before VQGAN runs.
// Lets the caller release transformer-only resources (e.g. drop the OpenCL
// backend's transformer scratch arena to give VQGAN ~500 MB of headroom).
using OnTransformerDone = std::function<void()>;

struct GenConfig {
    int      class_id   = 207;
    int      steps      = 8;
    uint64_t seed       = 42;
    float    temperature = 4.5f;   // choice temperature; <0 => use model default
};

struct Image {
    int width = 0, height = 0, channels = 3;
    std::vector<uint8_t> rgb;      // [H*W*3], row-major (h,w,c), clipped to [0,255]
};

// Optional per-component timing (filled by generate() when a pointer is passed).
// Backend-agnostic wall time; transformer_ms is summed over all decode steps.
struct GenStats {
    double transformer_ms = 0;   // all n_steps transformer forwards
    double sampling_ms    = 0;   // host-side sampling / masking / confidence
    double vqgan_ms       = 0;   // VQGAN decode (once)
    int    steps          = 0;
};

// Full pipeline: class id -> iterative masked decoding -> VQGAN decode -> image.
// wctx must hold the loaded weights (Model). A scratch context is created
// internally for activations. Pass stats to capture per-component timing.
Image generate(const Model& m, const GenConfig& cfg, bool verbose = true,
               const TransformerFwd& xnn_fwd = nullptr,
               const VqganFwd& vqgan_fwd = nullptr,
               GenStats* stats = nullptr,
               const OnTransformerDone& on_done = nullptr);

} // namespace mg
