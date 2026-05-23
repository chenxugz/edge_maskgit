// mg-transformer.hpp — build the MaskGIT bidirectional-transformer forward graph.
#pragma once
#include "mg-model.hpp"
#include "mg-tensor.hpp"

namespace mg {

// Build logits for one forward pass.
//   token_ids: I32 tensor, ne = {S, B}  (S positions = n_tokens+1 incl. class token)
//   returns:   F32 logits, ne = {vocab_size, S, B}
// Matches the PyTorch BidirectionalTransformer (post-norm, 16 heads, MLM head
// tied to the token embedding + output.bias).
Tensor* build_transformer(Context& ctx, const Model& m, Tensor* token_ids);

} // namespace mg
