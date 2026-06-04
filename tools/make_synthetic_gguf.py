#!/usr/bin/env python
"""Fabricate a synthetic MaskGIT GGUF at arbitrary sequence length, for
latency-only benchmarking. Quality is irrelevant — random weights — but the
file is shape- and format-correct so the existing loader, transformer graph,
and VQGAN decoder run end-to-end.

Mechanism: read an existing real GGUF's metadata + tensor-info section
(skipping tensor data), override `maskgit.n_tokens` and the `pos_embd.weight`
row count, then write a fresh GGUF with random tensor data sized per type.

Use:
  python tools/make_synthetic_gguf.py \\
      --base models/maskgit-256-q8.gguf  --n-tokens 1024 \\
      -o models/synth-1024-q8.gguf

The base file's quantization scheme is preserved (q8 base -> q8 synth,
gq8 base -> gq8 synth). For Q8_0 / I8 / MG_I4 tensors we emit randomized
int8 quants but **bounded fp16 scales** (~uniform in [0.005, 0.02]) so
quantized matmul kernels don't see NaN/Inf scales and short-circuit.
"""
from argparse import ArgumentParser
from pathlib import Path
import struct
import sys

import numpy as np

# Reuse the GGUF writer + constants from convert_to_gguf.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_to_gguf import (  # noqa: E402
    Writer, ALIGN, GGUF_MAGIC,
    GGML_F32, GGML_Q8_0, GGML_Q4_K, GGML_I8, MG_I4,
    T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_INT64,
    pad_to,
)


# ----------------------------------------------------------------------------
# Reader: just enough to walk metadata + tensor infos. We skip tensor data.
# ----------------------------------------------------------------------------

class R:
    def __init__(self, buf): self.b = buf; self.p = 0
    def take(self, n):       v = self.b[self.p:self.p+n]; self.p += n; return v
    def u32(self):           v = struct.unpack_from("<I", self.b, self.p)[0]; self.p += 4; return v
    def i32(self):           v = struct.unpack_from("<i", self.b, self.p)[0]; self.p += 4; return v
    def i64(self):           v = struct.unpack_from("<q", self.b, self.p)[0]; self.p += 8; return v
    def u64(self):           v = struct.unpack_from("<Q", self.b, self.p)[0]; self.p += 8; return v
    def f32(self):           v = struct.unpack_from("<f", self.b, self.p)[0]; self.p += 4; return v
    def u8(self):            v = self.b[self.p]; self.p += 1; return v
    def string(self):
        n = self.u64()
        s = self.b[self.p:self.p+n].decode("utf-8"); self.p += n; return s


def read_kv_value(r):
    t = r.u32()
    if   t == T_UINT32:  return ("u32", r.u32())
    elif t == T_INT32:   return ("i32", r.i32())
    elif t == T_FLOAT32: return ("f32", r.f32())
    elif t == T_BOOL:    return ("bool", r.u8())
    elif t == T_STRING:  return ("str", r.string())
    elif t == T_INT64:   return ("i64", r.i64())
    elif t == T_ARRAY:
        et = r.u32(); n = r.u64()
        items = [read_kv_inner(r, et) for _ in range(n)]
        return ("arr", et, items)
    else:
        raise RuntimeError(f"unsupported kv type {t}")


def read_kv_inner(r, t):
    if   t == T_UINT32:  return r.u32()
    elif t == T_INT32:   return r.i32()
    elif t == T_FLOAT32: return r.f32()
    elif t == T_BOOL:    return r.u8()
    elif t == T_STRING:  return r.string()
    elif t == T_INT64:   return r.i64()
    else: raise RuntimeError(f"unsupported nested kv type {t}")


def parse_gguf(path: Path):
    buf = path.read_bytes()
    r = R(buf)
    magic, version, n_tensors, n_kv = struct.unpack_from("<IIQQ", buf, 0)
    r.p = 24
    if magic != GGUF_MAGIC or version != 3:
        raise RuntimeError(f"bad magic/version: {magic:#x} {version}")
    kvs = []
    for _ in range(n_kv):
        key = r.string()
        val = read_kv_value(r)
        kvs.append((key, val))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        ne = [r.u64() for _ in range(nd)]
        tc = r.u32()
        off = r.u64()
        infos.append({"name": name, "ne": ne, "type": tc, "offset": off})
    return kvs, infos


# ----------------------------------------------------------------------------
# Random data generators per tensor type — sane scales so kernels don't NaN.
# ----------------------------------------------------------------------------

def _bounded_scales_fp16(n: int, lo=0.005, hi=0.02):
    """n bounded positive fp16 scales -> uint8 raw bytes [n*2]."""
    s = np.random.uniform(lo, hi, n).astype(np.float16)
    return s.view(np.uint8).tobytes()


def random_tensor_bytes(name: str, ne_in_order, type_code: int) -> bytes:
    """`ne_in_order` is innermost-first (GGUF/ggml convention)."""
    # Total element count.
    numel = 1
    for d in ne_in_order: numel *= d

    if type_code == GGML_F32:
        # `.scale` tensors (q8/q4 per-channel scales): XNNPACK validates these as
        # strictly positive — randn would hit negatives. Bounded uniform here.
        if name.endswith(".scale"):
            return np.random.uniform(0.005, 0.02, numel).astype(np.float32).tobytes()
        # norm `.weight` is the gain (positive ~1) and norm `.bias` is the shift.
        # Generic tensors: small N(0, 0.02²)-ish so accumulated activations stay
        # finite through 24 layers of matmul + LN.
        return (np.random.randn(numel).astype(np.float32) * 0.02).tobytes()

    if type_code == GGML_Q8_0:
        # blocks of 32 along ne[0] (inner). 34 bytes/block: fp16 scale + 32 int8.
        ic = ne_in_order[0]
        assert ic % 32 == 0, f"Q8_0 needs inner dim%32==0, got {ic} on {name}"
        nblk = numel // 32
        out = bytearray(nblk * 34)
        scales = _bounded_scales_fp16(nblk)
        quants = np.random.randint(-127, 128, size=nblk * 32, dtype=np.int8).tobytes()
        for b in range(nblk):
            out[b*34:b*34+2]   = scales[b*2:b*2+2]
            out[b*34+2:b*34+34] = quants[b*32:b*32+32]
        return bytes(out)

    if type_code == GGML_Q4_K:
        # 144-byte super-blocks over ne[0]; emit "valid-ish" structure: fp16 d, fp16
        # dmin, 12B sc/min nibbles (random 0..63), 128B 4-bit quants. We don't
        # need real Q4_K reconstruction quality — just non-NaN scales.
        ic = ne_in_order[0]
        assert ic % 256 == 0, f"Q4_K needs inner dim%256==0, got {ic} on {name}"
        nsb = numel // 256
        out = bytearray(nsb * 144)
        d  = np.random.uniform(0.01, 0.05, nsb).astype(np.float16).view(np.uint8).reshape(-1)
        dm = np.random.uniform(0.0,  0.02, nsb).astype(np.float16).view(np.uint8).reshape(-1)
        scales = np.random.randint(0, 64, size=nsb * 12, dtype=np.uint8).tobytes()
        quants = np.random.randint(0, 256, size=nsb * 128, dtype=np.uint8).tobytes()
        for b in range(nsb):
            out[b*144:b*144+2]      = d[b*2:b*2+2]
            out[b*144+2:b*144+4]    = dm[b*2:b*2+2]
            out[b*144+4:b*144+16]   = scales[b*12:b*12+12]
            out[b*144+16:b*144+144] = quants[b*128:b*128+128]
        return bytes(out)

    if type_code == GGML_I8:
        return np.random.randint(-127, 128, size=numel, dtype=np.int8).tobytes()

    if type_code == MG_I4:
        # packed two-per-byte. numel here = OC * (IC/2) since the GGUF stores the
        # already-halved inner dim.
        return np.random.randint(0, 256, size=numel, dtype=np.uint8).tobytes()

    raise RuntimeError(f"random_tensor_bytes: unhandled type {type_code} on {name}")


# ----------------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------------

def write_kv(w: Writer, key: str, val: tuple):
    w.string(key)
    t = val[0]
    if   t == "u32":  w.u32(T_UINT32);  w.u32(val[1])
    elif t == "i32":  w.u32(T_INT32);   w.i32(val[1])
    elif t == "f32":  w.u32(T_FLOAT32); w.f32(val[1])
    elif t == "bool": w.u32(T_BOOL);    w.buf += bytes([val[1]])
    elif t == "str":  w.u32(T_STRING);  w.string(val[1])
    elif t == "i64":  w.u32(T_INT64);   w.buf += struct.pack("<q", val[1])
    elif t == "arr":
        _, et, items = val
        w.u32(T_ARRAY); w.u32(et); w.u64(len(items))
        for x in items:
            if   et == T_UINT32:  w.u32(x)
            elif et == T_INT32:   w.i32(x)
            elif et == T_FLOAT32: w.f32(x)
            elif et == T_STRING:  w.string(x)
            elif et == T_INT64:   w.buf += struct.pack("<q", x)
            else: raise RuntimeError(f"nested array type {et} unsupported")
    else:
        raise RuntimeError(f"unknown kv type {t}")


def main():
    ap = ArgumentParser()
    ap.add_argument("--base", required=True, help="real GGUF to clone the structure from")
    ap.add_argument("--n-tokens", type=int, required=True,
                    help="target sequence length (n_tokens; M = n_tokens+1 incl class)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    base = Path(args.base)
    kvs, infos = parse_gguf(base)

    # Patch: override maskgit.n_tokens and resize pos_embd.weight rows.
    new_n_tokens = args.n_tokens
    found_nt = False
    out_kvs = []
    for k, v in kvs:
        if k == "maskgit.n_tokens":
            out_kvs.append((k, ("u32", new_n_tokens)))
            found_nt = True
        elif k == "general.name":
            out_kvs.append((k, ("str", f"synthetic n_tokens={new_n_tokens} (random weights)")))
        elif k == "general.file_type":
            out_kvs.append((k, v))
        else:
            out_kvs.append((k, v))
    if not found_nt:
        raise RuntimeError("base GGUF has no maskgit.n_tokens metadata")

    # Patch tensor info for pos_embd.weight: ne[0]=E, ne[1]=n_tokens+1.
    found_pos = False
    out_infos = []
    for t in infos:
        if t["name"] == "pos_embd.weight":
            ne = list(t["ne"])
            # GGUF ne is innermost-first; pos_embd ne=[E, n_tokens+1].
            assert len(ne) == 2, f"pos_embd shape unexpected: {ne}"
            ne[1] = new_n_tokens + 1
            out_infos.append({"name": t["name"], "ne": ne, "type": t["type"]})
            found_pos = True
        else:
            out_infos.append({"name": t["name"], "ne": list(t["ne"]), "type": t["type"]})
    if not found_pos:
        raise RuntimeError("base GGUF has no pos_embd.weight tensor")

    # Compute offsets + emit.
    # First pass: build tensor data blobs (random per type).
    blobs = []
    for t in out_infos:
        b = random_tensor_bytes(t["name"], t["ne"], t["type"])
        blobs.append(b)

    # Compute offsets relative to the data segment.
    offsets, cursor = [], 0
    for b in blobs:
        offsets.append(cursor)
        cursor = (cursor + len(b) + ALIGN - 1) // ALIGN * ALIGN

    # Header + metadata + tensor infos.
    out = bytearray()
    out += struct.pack("<IIQQ", GGUF_MAGIC, 3, len(out_infos), len(out_kvs))
    kw = Writer()
    for k, v in out_kvs:
        write_kv(kw, k, v)
    out += kw.buf

    ti = Writer()
    for t, off in zip(out_infos, offsets):
        ti.string(t["name"])
        ti.u32(len(t["ne"]))
        for d in t["ne"]: ti.u64(int(d))
        ti.u32(t["type"])
        ti.u64(off)
    out += ti.buf
    pad_to(out, ALIGN)
    data_start = len(out)

    # Tensor data section.
    for b, off in zip(blobs, offsets):
        while len(out) < data_start + off: out += b"\x00"
        out += b

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(bytes(out))
    print(f"[synth] wrote {outp}  n_tokens={new_n_tokens}  "
          f"{len(out)/1e6:.1f} MB  {len(out_infos)} tensors  {len(out_kvs)} kv")


if __name__ == "__main__":
    main()
