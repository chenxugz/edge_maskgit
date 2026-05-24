#!/usr/bin/env python
# Copyright 2026 edge_maskgit
# Licensed under the Apache License, Version 2.0
"""Convert the M1 weight export (reference/export/*.npz + metadata.json) into a
single GGUF v3 file for the C++ runtime.

Writes maskgit-256-f32.gguf containing both the transformer (GGUF-named tensors,
QKV already split) and the VQGAN (vqgan.* names), plus model hyperparameters as
GGUF metadata.

GGUF layout (v3, little-endian):
  magic 'GGUF' | u32 version=3 | u64 n_tensors | u64 n_kv
  metadata KVs: key(str) value_type(u32) value
  tensor infos: name(str) n_dims(u32) dims[u64..] type(u32) offset(u64)
  pad to alignment | tensor data (each aligned to general.alignment=32)

Dims are stored innermost-first (ggml ne order) = reversed numpy shape; data is
the raw C-order float32 buffer, so ne[0] == last numpy axis.
"""
from argparse import ArgumentParser
import json
from pathlib import Path
import struct
import sys

import numpy as np

ALIGN = 32
GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian

# GGUF metadata value types
T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_INT64 = 4, 5, 6, 7, 8, 9, 11
# tensor type codes (GGML F32; I8=24; MG_I4=100 = our packed signed int4)
GGML_F32 = 0
GGML_I8 = 24
MG_I4 = 100


def quant_int8(arr):
    """Per-output-channel (axis 0) symmetric int8. Returns (int8 [OC,IC], scale[OC])."""
    oc = arr.shape[0]
    flat = arr.reshape(oc, -1)
    amax = np.abs(flat).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(flat / scale[:, None]), -127, 127).astype(np.int8)
    return q.reshape(arr.shape), scale


def quant_int4(arr):
    """Per-output-channel signed int4 packed two-per-byte (low=even IC). [OC,IC]->[OC,IC/2]."""
    oc, ic = arr.shape
    amax = np.abs(arr).max(axis=1)
    scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(arr / scale[:, None]), -7, 7).astype(np.int32)
    lo = q[:, 0::2] & 0xF
    hi = q[:, 1::2] & 0xF
    packed = (lo | (hi << 4)).astype(np.uint8)        # [OC, IC/2]
    return packed, scale


def repack_oihw_to_ohwi(arr):
    """conv weight [OC,IC,KH,KW] -> [OC,KH,KW,IC] (XNNPACK filter layout)."""
    return np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def raw(self, b): self.buf += b
    def u32(self, v): self.buf += struct.pack("<I", v)
    def i32(self, v): self.buf += struct.pack("<i", v)
    def u64(self, v): self.buf += struct.pack("<Q", v)
    def f32(self, v): self.buf += struct.pack("<f", v)
    def string(self, s):
        b = s.encode("utf-8")
        self.u64(len(b)); self.raw(b)


def kv_header(w, key, vtype):
    w.string(key); w.u32(vtype)

def kv_u32(w, key, v):   kv_header(w, key, T_UINT32);  w.u32(v)
def kv_i32(w, key, v):   kv_header(w, key, T_INT32);   w.i32(v)
def kv_f32(w, key, v):   kv_header(w, key, T_FLOAT32); w.f32(float(v))
def kv_str(w, key, v):   kv_header(w, key, T_STRING);  w.string(v)
def kv_i32_array(w, key, vals):
    kv_header(w, key, T_ARRAY); w.u32(T_INT32); w.u64(len(vals))
    for v in vals: w.i32(int(v))


def pad_to(buf, align):
    while len(buf) % align != 0:
        buf += b"\x00"


def main():
    ap = ArgumentParser()
    ap.add_argument("--export-dir", default="reference/export")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--quant", choices=["f32", "q8", "q4"], default="f32",
                    help="f32 | q8 (int8 weights) | q4 (int4 transformer FC, int8 conv)")
    ap.add_argument("--check", action="store_true", help="print per-tensor checksums")
    args = ap.parse_args()
    if args.output is None:
        args.output = f"models/maskgit-256-{args.quant}.gguf"

    ed = Path(args.export_dir)
    meta = json.loads((ed / "metadata.json").read_text())
    t_npz = np.load(ed / "maskgit_transformer_f32.npz")
    v_npz = np.load(ed / "maskgit_vqgan_f32.npz")

    # Merge tensors (names already disjoint: transformer GGUF names vs vqgan.*).
    tensors = {}
    for npz in (t_npz, v_npz):
        for name in npz.files:
            tensors[name] = np.ascontiguousarray(npz[name].astype(np.float32))

    tf = meta["transformer"]; vq = meta["vqgan"]; sm = meta["sampling"]

    # ---- metadata KVs ----
    kw = Writer()
    n_kv = 0
    def emit(fn, *a):
        nonlocal n_kv; fn(kw, *a); n_kv += 1
    emit(kv_str, "general.architecture", meta["architecture"])
    emit(kv_str, "general.name", meta["name"])
    emit(kv_u32, "general.alignment", ALIGN)
    emit(kv_u32, "general.file_type", 0)  # all F32
    emit(kv_u32, "maskgit.resolution", meta["resolution"])
    emit(kv_u32, "maskgit.n_layer", tf["n_layer"])
    emit(kv_u32, "maskgit.n_head", tf["n_head"])
    emit(kv_u32, "maskgit.n_embd", tf["n_embd"])
    emit(kv_u32, "maskgit.n_ffn", tf["n_ffn"])
    emit(kv_u32, "maskgit.head_dim", tf["head_dim"])
    emit(kv_u32, "maskgit.vocab_size", tf["vocab_size"])
    emit(kv_u32, "maskgit.n_tokens", tf["n_tokens"])
    emit(kv_u32, "maskgit.n_positions", tf["max_position_embeddings"])
    emit(kv_i32, "maskgit.mask_token_id", tf["mask_token_id"])
    emit(kv_f32, "maskgit.layernorm_eps", tf["layernorm_eps"])
    emit(kv_u32, "maskgit.vqgan.codebook_size", vq["codebook_size"])
    emit(kv_u32, "maskgit.vqgan.embedding_dim", vq["embedding_dim"])
    emit(kv_u32, "maskgit.vqgan.filters", vq["filters"])
    emit(kv_u32, "maskgit.vqgan.num_res_blocks", vq["num_res_blocks"])
    emit(kv_u32, "maskgit.vqgan.group_norm_groups", vq["group_norm_groups"])
    emit(kv_i32_array, "maskgit.vqgan.channel_multipliers", vq["channel_multipliers"])
    emit(kv_f32, "maskgit.sampling.choice_temperature", sm["choice_temperature"])
    emit(kv_str, "maskgit.quant", args.quant)

    # ---- decide per-tensor representation ----
    FC_SUFFIX = ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_o.weight",
                 "ffn_up.weight", "ffn_down.weight")
    def is_fc(name):  # transformer FC weights (quantizable); NOT token_embd / biases
        return name.endswith(FC_SUFFIX) or name == "output_proj.weight"
    def is_conv(name, arr):  # VQGAN conv weights (4D)
        return name.startswith("vqgan.decoder.") and name.endswith(".weight") and arr.ndim == 4

    # out_tensors: list of (name, ne[innermost-first], type_code, bytes)
    out_tensors = []
    def add(name, ne, type_code, data):
        out_tensors.append((name, list(ne), type_code, bytes(data)))

    for name, arr in tensors.items():
        if name.startswith("vqgan.encoder."):
            continue          # encoder is unused (we only decode) — drop to shrink the file
        q = args.quant
        if q != "f32" and is_fc(name):
            if q == "q4":
                packed, scale = quant_int4(arr)            # [OC, IC/2] uint8
                add(name, reversed(arr.shape), MG_I4, packed.tobytes("C"))  # ne logical [IC,OC]
            else:
                qi, scale = quant_int8(arr)                # [OC, IC] int8
                add(name, reversed(arr.shape), GGML_I8, qi.tobytes("C"))
            add(name + ".scale", [arr.shape[0]], GGML_F32, scale.astype(np.float32).tobytes("C"))
        # NOTE: VQGAN conv weights are kept F32 in the file and quantized on load.
        # Pre-quantized conv (qd8-f32-qc8w from stored qcint8) currently NaNs in
        # XNNPACK despite byte-identical weights/scales to the working on-load path;
        # the on-load conv quant is proven, so we use that. (FC pre-quant is fine.)
        else:
            add(name, reversed(arr.shape), GGML_F32, arr.tobytes("C"))

    # ---- tensor infos + data ----
    offsets, cursor = {}, 0
    for name, ne, tc, data in out_tensors:
        offsets[name] = cursor
        cursor = (cursor + len(data) + ALIGN - 1) // ALIGN * ALIGN

    ti = Writer()
    for name, ne, tc, data in out_tensors:
        ti.string(name); ti.u32(len(ne))
        for d in ne: ti.u64(int(d))
        ti.u32(tc); ti.u64(offsets[name])

    out = bytearray()
    out += struct.pack("<IIQQ", GGUF_MAGIC, 3, len(out_tensors), n_kv)
    out += kw.buf
    out += ti.buf
    pad_to(out, ALIGN)
    data_start = len(out)
    for name, ne, tc, data in out_tensors:
        while len(out) < data_start + offsets[name]: out += b"\x00"
        out += data

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(out)
    total_params = sum(int(a.size) for a in tensors.values())
    print(f"[gguf] wrote {outp}  (quant={args.quant}, {len(out)/1e6:.1f} MB, "
          f"{len(out_tensors)} tensors, {total_params:,} src params, {n_kv} metadata kv)")

    if args.check:
        print("[gguf] checksums (float64 sum) for spot tensors:")
        for name in ["token_embd.weight", "pos_embd.weight",
                     "blk.0.attn_q.weight", "blk.23.ffn_down.weight",
                     "output.bias", "vqgan.quantizer.codebook.weight",
                     "vqgan.decoder.conv_out.weight"]:
            if name in tensors:
                a = tensors[name].astype(np.float64)
                print(f"    {name:42s} shape={tuple(tensors[name].shape)} "
                      f"sum={a.sum():.6f} mean={a.mean():.8f}")


if __name__ == "__main__":
    main()
