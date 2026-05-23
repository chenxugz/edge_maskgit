#!/usr/bin/env python
# Copyright 2026 edge_maskgit
# Licensed under the Apache License, Version 2.0
"""Dump fixed-input reference tensors for verifying the C++ runtime (M2/M3).

Writes to reference/export/:
  verify_tokens.bin   int32[S]            fixed transformer input (class 207)
  verify_logits.bin   float32[S, vocab]   PyTorch transformer logits for it
  verify_grid.bin     int32[h, w]         fixed VQGAN token grid
  verify_image.bin    float32[H, W, 3]    PyTorch VQGAN decode of that grid
  verify_meta.json    shapes

These isolate kernel/graph correctness from sampling RNG: the C++ runtime is fed
the SAME fixed inputs and must match these outputs.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE); sys.path.insert(0, str(_HERE))
from maskgit.inference import ImageNet_class_conditional_generator  # noqa: E402

OUT = _HERE / "export"


def main():
    gen = ImageNet_class_conditional_generator(image_size=256)
    gen.maskgit_cf.eval_batch_size = 1

    # --- transformer: fixed input for class 207 (label token + all-mask) ---
    tokens = gen.create_input_tokens_normal(207)          # [1, S] long
    with torch.no_grad():
        logits = gen.transformer_model(tokens)            # [1, S, vocab]
    tok_np = tokens[0].cpu().numpy().astype(np.int32)
    log_np = logits[0].cpu().numpy().astype(np.float32)
    tok_np.tofile(OUT / "verify_tokens.bin")
    log_np.tofile(OUT / "verify_logits.bin")

    # --- vqgan: fixed token grid -> decoded image ---
    latent = gen.transformer_latent_size                  # 16
    rng = np.random.RandomState(0)
    grid = rng.randint(0, gen.maskgit_cf.vqvae.codebook_size, size=(latent, latent)).astype(np.int32)
    grid_t = torch.from_numpy(grid).long().unsqueeze(0)   # [1,16,16]
    with torch.no_grad():
        img = gen.tokenizer_model.decode_from_indices({"encoding_indices": grid_t})  # [1,3,H,W]
    img_np = img[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)   # [H,W,3]
    grid.tofile(OUT / "verify_grid.bin")
    img_np.tofile(OUT / "verify_image.bin")

    meta = {
        "tokens_shape": list(tok_np.shape),
        "logits_shape": list(log_np.shape),
        "grid_shape": list(grid.shape),
        "image_shape": list(img_np.shape),
        "class_id": 207,
    }
    (OUT / "verify_meta.json").write_text(json.dumps(meta, indent=2))
    print("[dump_verify] tokens", tok_np.shape, "logits", log_np.shape,
          "grid", grid.shape, "image", img_np.shape)
    print("[dump_verify] logits[0,:5] =", log_np[0, :5])
    print("[dump_verify] image sum =", float(img_np.sum()))


if __name__ == "__main__":
    main()
