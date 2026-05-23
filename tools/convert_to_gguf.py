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
# GGML tensor types
GGML_F32 = 0


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
    ap.add_argument("-o", "--output", default="models/maskgit-256-f32.gguf")
    ap.add_argument("--check", action="store_true", help="print per-tensor checksums")
    args = ap.parse_args()

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

    # ---- tensor infos + data (two passes: compute offsets, then emit) ----
    names = list(tensors.keys())
    # data section: each tensor aligned to ALIGN
    offsets = {}
    cursor = 0
    for name in names:
        offsets[name] = cursor
        nbytes = tensors[name].nbytes
        cursor += nbytes
        cursor = (cursor + ALIGN - 1) // ALIGN * ALIGN

    ti = Writer()
    for name in names:
        arr = tensors[name]
        ne = list(reversed(arr.shape))           # innermost-first
        ti.string(name)
        ti.u32(len(ne))
        for d in ne: ti.u64(int(d))
        ti.u32(GGML_F32)
        ti.u64(offsets[name])

    # ---- assemble ----
    out = bytearray()
    out += struct.pack("<IIQQ", GGUF_MAGIC, 3, len(names), n_kv)
    out += kw.buf
    out += ti.buf
    pad_to(out, ALIGN)                            # data section starts aligned
    data_start = len(out)
    for name in names:
        arr = tensors[name]
        # pad current position to this tensor's offset
        target = data_start + offsets[name]
        while len(out) < target: out += b"\x00"
        out += arr.tobytes(order="C")

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(out)
    total_params = sum(int(a.size) for a in tensors.values())
    print(f"[gguf] wrote {outp}  ({len(out)/1e6:.1f} MB, {len(names)} tensors, "
          f"{total_params:,} params, {n_kv} metadata kv)")

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
