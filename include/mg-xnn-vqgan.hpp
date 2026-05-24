// mg-xnn-vqgan.hpp — XNNPACK subgraph for the MaskGIT VQGAN decoder.
//
// Built entirely in NHWC (XNNPACK's conv layout). Codebook gather is host-side;
// conv weights are repacked OIHW->OHWI; GroupNorm is composed; nearest 2x
// upsample is a depthwise all-ones transposed conv. Output is HWC float.
#pragma once
#include "mg-model.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace mg {

class XnnVqgan {
public:
    explicit XnnVqgan(const Model& m, Quant quant = Quant::F32);
    ~XnnVqgan();
    XnnVqgan(const XnnVqgan&) = delete;
    XnnVqgan& operator=(const XnnVqgan&) = delete;

    // grid: int32 [n_tokens] (row-major h*W+w); image_out: float [H*W*3] HWC in [~0,1]
    void decode(const int32_t* grid, float* image_out);
    int width() const { return WH_; }
    int height() const { return WH_; }

private:
    const Model& m_;
    int E_, WH_;                       // embedding dim, output side (256)
    void* subgraph_ = nullptr;
    void* runtime_  = nullptr;
    uint32_t in_id_ = 0, out_id_ = 0;
    std::vector<float> feat_;          // host codebook-gather buffer [n_tokens*E]
    std::deque<std::vector<float>> packed_;  // repacked conv weights / ones kernels / zero biases
    std::deque<float> consts_;
    Quant quant_ = Quant::F32;
    std::deque<std::vector<int8_t>> qw_;     // quantized conv weights (OHWI int8)
    std::deque<std::vector<float>>  qs_;     // per-output-channel scales
};

} // namespace mg
