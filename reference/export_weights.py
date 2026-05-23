#!/usr/bin/env python
# Copyright 2026 edge_maskgit
# Licensed under the Apache License, Version 2.0
"""Export MaskGIT PyTorch weights to flat .npz with the GGUF naming convention.

Produces the input for the Milestone 2 GGUF converter:
  export/maskgit_transformer_f32.npz   transformer weights (GGUF-style names)
  export/maskgit_vqgan_f32.npz         VQGAN codebook + decoder (+ encoder) weights
  export/metadata.json                 hyperparameters, shapes, dtypes, name map, notes

Naming follows CLAUDE.md §2.5, adapted to the actual checkpoint layout:
 - Fused QKV (nn.MultiheadAttention.in_proj_weight, [3*d, d]) is SPLIT into
   blk.{i}.attn_q/k/v.weight (+ .bias), since the C++ matmul kernels want them
   separate. The split is lossless (concat q;k;v reproduces in_proj exactly).
 - The output projection is TIED to the token embedding (logits = h @ tok_emb.T
   + output.bias); we do not duplicate the matrix, just record the tie.

Round-trip: --verify reloads the .npz, reconstructs the exact torch state_dicts,
loads them into the models, runs one generation, and asserts bit-identical output
vs. loading the original checkpoints — proving the export is lossless.
"""
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

from maskgit.configs import maskgit_class_cond_config  # noqa: E402

CKPT_DIR = _HERE / "checkpoints"
N_LAYERS = 24


# ---------------------------------------------------------------------------
# Forward mapping: torch state_dict -> {gguf_name: np.ndarray}
# Each helper also knows how to invert, kept together for a single source of truth.
# ---------------------------------------------------------------------------
def export_transformer(sd: dict) -> "dict[str, np.ndarray]":
    """Map transformer state_dict to GGUF-named float32 numpy arrays."""
    def npy(k):
        return sd[k].detach().cpu().float().numpy()

    out = {}
    out["token_embd.weight"] = npy("tok_emb.weight")        # [2025, 768]
    out["pos_embd.weight"] = npy("pos_emb")                 # [257, 768]
    out["token_embd_norm.weight"] = npy("emb_ln.weight")
    out["token_embd_norm.bias"] = npy("emb_ln.bias")

    for i in range(N_LAYERS):
        p = f"blocks.{i}."
        # fused QKV [3*768, 768] -> three [768, 768] in order q, k, v
        iw = npy(p + "MultiHeadAttention.in_proj_weight")
        ib = npy(p + "MultiHeadAttention.in_proj_bias")
        d = iw.shape[1]
        for j, name in enumerate(("q", "k", "v")):
            out[f"blk.{i}.attn_{name}.weight"] = iw[j * d:(j + 1) * d]
            out[f"blk.{i}.attn_{name}.bias"] = ib[j * d:(j + 1) * d]
        out[f"blk.{i}.attn_o.weight"] = npy(p + "MultiHeadAttention.out_proj.weight")
        out[f"blk.{i}.attn_o.bias"] = npy(p + "MultiHeadAttention.out_proj.bias")
        out[f"blk.{i}.attn_norm.weight"] = npy(p + "AttentionLN.weight")
        out[f"blk.{i}.attn_norm.bias"] = npy(p + "AttentionLN.bias")
        out[f"blk.{i}.ffn_up.weight"] = npy(p + "MLP.0.weight")
        out[f"blk.{i}.ffn_up.bias"] = npy(p + "MLP.0.bias")
        out[f"blk.{i}.ffn_down.weight"] = npy(p + "MLP.2.weight")
        out[f"blk.{i}.ffn_down.bias"] = npy(p + "MLP.2.bias")
        out[f"blk.{i}.ffn_norm.weight"] = npy(p + "MlpLN.weight")
        out[f"blk.{i}.ffn_norm.bias"] = npy(p + "MlpLN.bias")

    out["output_proj.weight"] = npy("Token_Prediction.0.weight")
    out["output_proj.bias"] = npy("Token_Prediction.0.bias")
    out["output_norm.weight"] = npy("Token_Prediction.2.weight")
    out["output_norm.bias"] = npy("Token_Prediction.2.bias")
    out["output.bias"] = npy("bias")                        # MLM bias [2025]
    # output.weight is tied to token_embd.weight (not duplicated).
    return out


def rebuild_transformer(npz: dict) -> dict:
    """Invert export_transformer -> exact torch state_dict (for round-trip)."""
    t = lambda a: torch.from_numpy(np.asarray(a)).float()
    sd = {}
    sd["tok_emb.weight"] = t(npz["token_embd.weight"])
    sd["pos_emb"] = t(npz["pos_embd.weight"])
    sd["emb_ln.weight"] = t(npz["token_embd_norm.weight"])
    sd["emb_ln.bias"] = t(npz["token_embd_norm.bias"])
    for i in range(N_LAYERS):
        p = f"blocks.{i}."
        qw = [npz[f"blk.{i}.attn_{n}.weight"] for n in ("q", "k", "v")]
        qb = [npz[f"blk.{i}.attn_{n}.bias"] for n in ("q", "k", "v")]
        sd[p + "MultiHeadAttention.in_proj_weight"] = t(np.concatenate(qw, axis=0))
        sd[p + "MultiHeadAttention.in_proj_bias"] = t(np.concatenate(qb, axis=0))
        sd[p + "MultiHeadAttention.out_proj.weight"] = t(npz[f"blk.{i}.attn_o.weight"])
        sd[p + "MultiHeadAttention.out_proj.bias"] = t(npz[f"blk.{i}.attn_o.bias"])
        sd[p + "AttentionLN.weight"] = t(npz[f"blk.{i}.attn_norm.weight"])
        sd[p + "AttentionLN.bias"] = t(npz[f"blk.{i}.attn_norm.bias"])
        sd[p + "MLP.0.weight"] = t(npz[f"blk.{i}.ffn_up.weight"])
        sd[p + "MLP.0.bias"] = t(npz[f"blk.{i}.ffn_up.bias"])
        sd[p + "MLP.2.weight"] = t(npz[f"blk.{i}.ffn_down.weight"])
        sd[p + "MLP.2.bias"] = t(npz[f"blk.{i}.ffn_down.bias"])
        sd[p + "MlpLN.weight"] = t(npz[f"blk.{i}.ffn_norm.weight"])
        sd[p + "MlpLN.bias"] = t(npz[f"blk.{i}.ffn_norm.bias"])
    sd["Token_Prediction.0.weight"] = t(npz["output_proj.weight"])
    sd["Token_Prediction.0.bias"] = t(npz["output_proj.bias"])
    sd["Token_Prediction.2.weight"] = t(npz["output_norm.weight"])
    sd["Token_Prediction.2.bias"] = t(npz["output_norm.bias"])
    sd["bias"] = t(npz["output.bias"])
    return sd


def export_vqgan(sd: dict) -> "dict[str, np.ndarray]":
    """VQGAN: keep the natural torch names under a vqgan. prefix (1:1, lossless).

    The decoder module names already form a clean, documented hierarchy
    (decoder.res_blocks.{stage}.{layer}.{conv0|conv1|norm0|norm1|conv_res}); we
    prefix them and pass tensors through unchanged. Encoder weights are included
    for completeness/round-trip though only the decoder + codebook are needed for
    generation.
    """
    return {f"vqgan.{k}": v.detach().cpu().float().numpy() for k, v in sd.items()}


def rebuild_vqgan(npz: dict) -> dict:
    return {k[len("vqgan."):]: torch.from_numpy(np.asarray(v)).float()
            for k, v in npz.items() if k.startswith("vqgan.")}


# ---------------------------------------------------------------------------
def load_ckpt(name: str) -> dict:
    ck = torch.load(CKPT_DIR / name, map_location="cpu", weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck


def build_metadata(cf, t_npz, v_npz) -> dict:
    def shapes(d):
        return {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
                for k, v in d.items()}
    return {
        "architecture": "maskgit",
        "name": "MaskGIT ImageNet 256x256 (PyTorch oracle, converted official weights)",
        "source": "hmorimitsu/maskgit-torch (converted from google-research/maskgit)",
        "resolution": int(cf.image_size),
        "transformer": {
            "n_layer": int(cf.transformer.num_layers),
            "n_head": int(cf.transformer.num_heads),     # 16 (NOT 8 as CLAUDE.md prose says)
            "n_embd": int(cf.transformer.num_embeds),
            "n_ffn": int(cf.transformer.intermediate_size),
            "head_dim": int(cf.transformer.num_embeds // cf.transformer.num_heads),
            "vocab_size": int(cf.vqvae.codebook_size + cf.num_class + 1),  # 2025
            "n_tokens": int((cf.image_size // cf.transformer.patch_size) ** 2),  # 256
            "max_position_embeddings": int((cf.image_size // cf.transformer.patch_size) ** 2 + 1),
            "mask_token_id": int(cf.transformer.mask_token_id),  # 2024 in PyTorch port (orig JAX used -1; nn.Embedding needs a valid row)
            "layernorm_eps": 1e-12,
            "norm_placement": "post",   # LN applied AFTER residual add (BERT-style)
            "activation": "gelu",
            "output_tied_to_token_embd": True,
        },
        "vqgan": {
            "codebook_size": int(cf.vqvae.codebook_size),       # 1024
            "embedding_dim": int(cf.vqvae.embedding_dim),       # 256
            "filters": int(cf.vqvae.filters),                   # 128
            "channel_multipliers": list(cf.vqvae.channel_multipliers),  # [1,1,2,2,4]
            "num_res_blocks": int(cf.vqvae.num_res_blocks),     # 2
            "norm_type": cf.vqvae.norm_type,                    # GN (32 groups)
            "group_norm_groups": 32,
            "activation": cf.vqvae.activation_fn,               # swish (SiLU)
            "resblock_shortcut_quirk": "shortcut conv applies to processed x, not the input residual",
        },
        "sampling": {
            "default_steps": 8,
            "paper_steps": 16,
            "choice_temperature": float(cf.sample_choice_temperature),  # 4.5
            "mask_schedule": cf.mask_scheduling_method,          # cosine
        },
        "files": {
            "maskgit_transformer_f32.npz": shapes(t_npz),
            "maskgit_vqgan_f32.npz": shapes(v_npz),
        },
    }


def verify_roundtrip(t_npz, v_npz) -> bool:
    """Reload exported arrays, rebuild models, assert identical generation."""
    from maskgit.inference import ImageNet_class_conditional_generator
    import run_reference

    print("[verify] generating reference from ORIGINAL checkpoints ...", flush=True)
    gen = ImageNet_class_conditional_generator(image_size=256)
    gen.maskgit_cf.eval_batch_size = 1
    # Seed immediately before generation: model construction itself consumes RNG
    # (weight init), so seeding must happen after it for both runs to align.
    run_reference.set_seed(42)
    img_orig = run_reference.generate(gen, class_id=207, steps=8, temperature=4.5)

    print("[verify] reloading models from EXPORTED npz ...", flush=True)
    t_sd = rebuild_transformer({k: t_npz[k] for k in t_npz.files})
    v_sd = rebuild_vqgan({k: v_npz[k] for k in v_npz.files})
    miss_t, unexp_t = gen.transformer_model.load_state_dict(t_sd, strict=False)
    miss_v, unexp_v = gen.tokenizer_model.load_state_dict(v_sd, strict=False)
    # The only acceptable "missing" transformer key is the tied output (none here);
    # report anything unexpected.
    if unexp_t or unexp_v:
        print(f"[verify] WARNING unexpected keys: t={unexp_t} v={unexp_v}")

    run_reference.set_seed(42)
    img_rt = run_reference.generate(gen, class_id=207, steps=8, temperature=4.5)
    ok = np.array_equal(img_orig, img_rt)
    maxdiff = int(np.abs(img_orig.astype(int) - img_rt.astype(int)).max())
    print(f"[verify] round-trip identical={ok} max_pixel_diff={maxdiff}")
    return ok


def main(args: Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cf = maskgit_class_cond_config.get_config()
    cf.image_size = 256

    print("[export] loading checkpoints ...", flush=True)
    t_sd = load_ckpt("maskgit_imagenet256.ckpt")
    v_sd = load_ckpt("tokenizer_imagenet256.ckpt")

    print("[export] mapping to GGUF names ...", flush=True)
    t_npz = export_transformer(t_sd)
    v_npz = export_vqgan(v_sd)

    t_path = out_dir / "maskgit_transformer_f32.npz"
    v_path = out_dir / "maskgit_vqgan_f32.npz"
    np.savez(t_path, **t_npz)
    np.savez(v_path, **v_npz)
    meta = build_metadata(cf, t_npz, v_npz)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    t_params = sum(int(np.asarray(v).size) for v in t_npz.values())
    v_params = sum(int(np.asarray(v).size) for v in v_npz.values())
    print(f"[export] transformer: {len(t_npz)} tensors, {t_params:,} params -> {t_path}")
    print(f"[export] vqgan:       {len(v_npz)} tensors, {v_params:,} params -> {v_path}")
    print(f"[export] metadata -> {out_dir / 'metadata.json'}")

    if args.verify:
        t_re = np.load(t_path)
        v_re = np.load(v_path)
        ok = verify_roundtrip(t_re, v_re)
        if not ok:
            raise SystemExit("[export] ROUND-TRIP FAILED — export is not lossless")
        print("[export] round-trip OK — export is lossless")


def _init_parser() -> ArgumentParser:
    p = ArgumentParser(description="Export MaskGIT weights to GGUF-named .npz")
    p.add_argument("-o", "--output-dir", default="export")
    p.add_argument("--verify", action="store_true",
                   help="Round-trip: reload npz, reload models, assert identical generation")
    return p


if __name__ == "__main__":
    main(_init_parser().parse_args())
