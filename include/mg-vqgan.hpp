// mg-vqgan.hpp — build the MaskGIT VQGAN decoder forward graph.
#pragma once
#include "mg-model.hpp"
#include "mg-tensor.hpp"

namespace mg {

// Decode a token grid to an image.
//   grid_ids: I32 tensor, ne = {n_tokens}  (flattened row-major, idx = h*W + w)
//   returns:  F32 image, ne = {W, H, 3, 1}  (W innermost)
// Mirrors reference/maskgit/nets/vqgan_tokenizer.py Decoder, including the
// ResBlock shortcut quirk (shortcut conv applied to the processed tensor).
Tensor* build_vqgan_decoder(Context& ctx, const Model& m, Tensor* grid_ids);

} // namespace mg
