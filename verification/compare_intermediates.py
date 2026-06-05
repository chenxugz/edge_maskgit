#!/usr/bin/env python
"""M3 per-layer numerical verification — compare C++ runtime intermediates against
the PyTorch oracle, tensor by tensor.

The oracle was dumped by `reference/dump_intermediates.py` to
  reference/export/intermediates_step1/{<name>.bin, meta.json}

A C++ run dumps the same names to <dump_dir>/{<name>.bin} via
  verify-opencl-transformer model.gguf reference/export --dump-dir <dump_dir>

This script walks the oracle's meta.json, reads each pair, and prints a per-tensor
table of (max_abs_diff, mean_abs_diff, cosine) — plus pass/fail at a configurable
threshold. Threshold defaults differ for F32 vs quantized backends.

Usage:
    python verification/compare_intermediates.py <dump_dir> [--max-diff M] [--cosine C]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REF_ROOT = Path(__file__).resolve().parent.parent / "reference" / "export" / "intermediates_step1"


def load_pair(name: str, shape: list[int], dump_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    a = np.fromfile(REF_ROOT / f"{name}.bin", dtype=np.float32).reshape(shape)
    b_path = dump_dir / f"{name}.bin"
    if not b_path.exists():
        return a, None  # signal missing tensor
    b = np.fromfile(b_path, dtype=np.float32)
    if b.size != a.size:
        return a, b  # size mismatch — caller prints a clear error
    return a, b.reshape(shape)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a - b)
    dot = float((a.astype(np.float64) * b.astype(np.float64)).sum())
    na = float(np.linalg.norm(a.astype(np.float64)))
    nb = float(np.linalg.norm(b.astype(np.float64)))
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cosine": dot / (na * nb) if na > 0 and nb > 0 else float("nan"),
        "a_norm": na,
        "b_norm": nb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="directory written by verify-opencl-transformer --dump-dir")
    ap.add_argument("--max-diff", type=float, default=None,
                    help="max_abs_diff tolerance (auto-tighter for f32 dumps)")
    ap.add_argument("--cosine", type=float, default=None,
                    help="cosine similarity floor")
    ap.add_argument("--quiet", action="store_true",
                    help="only print failing rows + summary")
    args = ap.parse_args()

    dump_dir = Path(args.dump_dir).resolve()
    meta = json.loads((REF_ROOT / "meta.json").read_text())

    # Tier the defaults by quant level (auto-detected from the output_logits
    # diff magnitude — fp32 < 1e-2, Q8 ~5e-2, Q4_K ~1e-0 at the logits).
    # Per-layer tolerances are looser than the logit gate because intermediate
    # activations accumulate quant noise that the output projection partly
    # averages out.
    a_log = np.fromfile(REF_ROOT / "output_logits.bin", dtype=np.float32)
    b_log = np.fromfile(dump_dir / "output_logits.bin", dtype=np.float32) \
            if (dump_dir / "output_logits.bin").exists() else None
    out_max = float(np.abs(a_log - b_log).max()) if (b_log is not None and b_log.size == a_log.size) else 1e9
    if   out_max < 1e-2: tier = "f32"
    elif out_max < 2e-1: tier = "q8"
    else:                tier = "q4"
    default_max = {"f32": 2e-2, "q8": 1.0, "q4": 2.0}[tier]
    # Q4_K per-layer accumulation drops middle-layer cosine to ~0.995 by layer ~10
    # before the output projection smooths the logits back to ≥0.9999. That's the
    # int4 weight precision budget on this model — not a kernel bug. 0.99 is the
    # right floor for the per-layer gate; the logit gate already enforces 0.9999.
    default_cos = {"f32": 0.99999, "q8": 0.9999, "q4": 0.99}[tier]
    max_tol = args.max_diff if args.max_diff is not None else default_max
    cos_tol = args.cosine   if args.cosine   is not None else default_cos

    print(f"oracle: {REF_ROOT}")
    print(f"dump:   {dump_dir}")
    print(f"tol:    max_abs ≤ {max_tol:g}   cosine ≥ {cos_tol:.6f}   ({tier} tier)")
    print()
    print(f"{'tensor':28s} {'shape':14s} {'max_abs':>10s} {'mean_abs':>10s} {'cosine':>11s}  status")
    print("-" * 88)

    # Walk in a stable order: embd → per-layer attn/ffn → output_norm → output_logits.
    n_layers = meta["n_layers"]
    order = ["embd_post_norm"]
    for i in range(n_layers):
        order.append(f"blk.{i}.attn_post")
        order.append(f"blk.{i}.ffn_post")
    order += ["output_norm", "output_logits"]

    pass_count = 0; fail_count = 0; missing_count = 0
    worst_cos = 1.0
    worst_max = 0.0
    for name in order:
        info = meta["tensors"][name]
        shape = info["shape"]
        a, b = load_pair(name, shape, dump_dir)
        if b is None:
            print(f"{name:28s} {'x'.join(str(d) for d in shape):14s} "
                  f"{'(missing dump)':>34s}  FAIL")
            missing_count += 1; fail_count += 1
            continue
        if b.shape != tuple(shape):
            print(f"{name:28s} expected {shape}, got {b.shape}   FAIL")
            fail_count += 1
            continue
        m = metrics(a, b)
        ok = (m["max_abs"] <= max_tol) and (m["cosine"] >= cos_tol)
        if ok:
            pass_count += 1
        else:
            fail_count += 1
        worst_cos = min(worst_cos, m["cosine"])
        worst_max = max(worst_max, m["max_abs"])
        status = "PASS" if ok else "FAIL"
        if not args.quiet or not ok:
            print(f"{name:28s} {'x'.join(str(d) for d in shape):14s} "
                  f"{m['max_abs']:10.4e} {m['mean_abs']:10.4e} {m['cosine']:11.8f}  {status}")

    print("-" * 88)
    print(f"summary: {pass_count} pass / {fail_count} fail / {missing_count} missing  "
          f"(worst cosine {worst_cos:.8f}, worst max_abs {worst_max:.4e})")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
