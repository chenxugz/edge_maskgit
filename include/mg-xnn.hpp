// mg-xnn.hpp — XNNPACK subgraph backend for the MaskGIT transformer.
//
// Builds the full transformer forward as an xnn_subgraph (built once, invoked
// per decode step). The embedding gather is done host-side (XNNPACK has no
// gather op): forward() takes token ids, gathers token_embd rows into the
// external input, runs the runtime, and returns logits. Validated against the
// reference kernels.
#pragma once
#include "mg-model.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace mg {

enum class Quant { F32, Q8, Q4 };   // FC weights: f32 | per-channel int8 | per-channel int4

class XnnTransformer {
public:
    XnnTransformer(const Model& m, int batch, int seq_len, Quant quant = Quant::F32);
    ~XnnTransformer();
    XnnTransformer(const XnnTransformer&) = delete;
    XnnTransformer& operator=(const XnnTransformer&) = delete;

    // tokens: int32 [B*S]; logits_out: float [B*S*vocab]
    void forward(const int32_t* tokens, float* logits_out);

    int seq_len() const { return S_; }
    int vocab() const { return V_; }

private:
    const Model& m_;
    int B_, S_, V_, H_, D_, E_;
    void* subgraph_ = nullptr;   // xnn_subgraph_t
    void* runtime_  = nullptr;   // xnn_runtime_t
    uint32_t in_id_ = 0, out_id_ = 0;
    std::vector<float> emb_;     // host embedding-gather buffer [B*S*E]
    std::deque<float>  consts_;  // stable storage for scalar static tensors
    Quant quant_ = Quant::F32;
    std::deque<std::vector<int8_t>> qw_;   // quantized weight storage (int8, or packed int4)
    std::deque<std::vector<float>>  qs_;   // per-output-channel scales
};

} // namespace mg
