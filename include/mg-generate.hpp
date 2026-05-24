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

// Full pipeline: class id -> iterative masked decoding -> VQGAN decode -> image.
// wctx must hold the loaded weights (Model). A scratch context is created
// internally for activations.
Image generate(const Model& m, const GenConfig& cfg, bool verbose = true,
               const TransformerFwd& xnn_fwd = nullptr);

} // namespace mg
