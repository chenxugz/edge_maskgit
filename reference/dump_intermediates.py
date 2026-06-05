#!/usr/bin/env python
# Copyright 2026 edge_maskgit
# Licensed under the Apache License, Version 2.0
"""Dump per-layer intermediate activations from one fixed forward pass of the
PyTorch transformer — the M3 oracle for per-layer numerical verification of
the C++ runtime.

We reuse `reference/export/verify_tokens.bin` as the input (= step-1 input:
class token at position 0, MASK at the rest), so this is consistent with what
verify-opencl-transformer already consumes for end-of-pipeline logit checks.

Hooked outputs, written as raw float32 [S, E] (row-major) under
reference/export/intermediates_step1/<name>.bin:

  embd_post_norm        # x after token_embd + pos_embd + emb_ln
  blk.{0..n-1}.attn_post # x after self-attention block + residual + AttentionLN
  blk.{0..n-1}.ffn_post  # x after MLP block + residual + MlpLN
  output_norm            # h after the final LayerNorm (before output_proj)
  output_logits          # final logits [S, vocab] (== verify_logits.bin)

A sibling meta.json records names + shapes for downstream tooling.
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

OUT_DIR = _HERE / "export" / "intermediates_step1"


def main():
    gen = ImageNet_class_conditional_generator(image_size=256)
    gen.maskgit_cf.eval_batch_size = 1
    m = gen.transformer_model
    m.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- register forward hooks ----
    # Each hook stashes (name, tensor[0].cpu().numpy().astype(f32)).
    captured: dict[str, np.ndarray] = {}

    def make_hook(name: str):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach()[0].cpu().numpy().astype(np.float32)
        return hook

    handles = []
    handles.append(m.emb_ln.register_forward_hook(make_hook("embd_post_norm")))
    n_layers = len(m.blocks)
    for i, blk in enumerate(m.blocks):
        handles.append(blk.AttentionLN.register_forward_hook(make_hook(f"blk.{i}.attn_post")))
        handles.append(blk.MlpLN.register_forward_hook(make_hook(f"blk.{i}.ffn_post")))
    # The output head is Sequential(Linear, GELU, LayerNorm). Final LN is index -1.
    handles.append(m.Token_Prediction[-1].register_forward_hook(make_hook("output_norm")))

    # ---- run forward on the canonical verify input ----
    tokens = torch.from_numpy(
        np.fromfile(_HERE / "export" / "verify_tokens.bin", dtype=np.int32).astype(np.int64)
    ).unsqueeze(0)                            # [1, S]
    with torch.no_grad():
        logits = m(tokens)                    # [1, S, vocab]
    captured["output_logits"] = logits.detach()[0].cpu().numpy().astype(np.float32)

    for h in handles:
        h.remove()

    # ---- write tensors + meta ----
    meta: dict = {"n_layers": n_layers, "input_path": "verify_tokens.bin", "tensors": {}}
    for name, arr in captured.items():
        path = OUT_DIR / f"{name}.bin"
        arr.tofile(path)
        meta["tensors"][name] = {"shape": list(arr.shape), "dtype": "float32",
                                 "n_elements": int(arr.size)}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    # ---- quick stdout summary ----
    print(f"[dump_intermediates] wrote {len(captured)} tensors to {OUT_DIR}")
    for n in ("embd_post_norm", "blk.0.attn_post", "blk.0.ffn_post",
              f"blk.{n_layers-1}.attn_post", f"blk.{n_layers-1}.ffn_post",
              "output_norm", "output_logits"):
        a = captured[n]
        print(f"  {n:32s} shape={a.shape} sum={a.sum():.4f} mean={a.mean():.6e}")


if __name__ == "__main__":
    main()
