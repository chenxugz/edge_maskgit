#!/usr/bin/env python
"""Batch wrapper around run_reference.py: load the MaskGIT model once and crunch
a list of (class_id, seed, output_path) jobs. The serial subprocess-per-image
flow paid ~17 s/image of model load — this amortizes it across the whole batch.

Used by evaluation/eval_runner.py when --backend reference is selected.

Job stream: one JSON object per line on stdin, like
    {"class_id": 207, "seed": 0, "out": "/path/to/c207_s00000.png"}

We print one line per completed job to stdout so the caller can track progress:
    [batch_reference] 1/40 c207_s00000.png 0.85s
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

# Reuse the loader + generation kernel so we behave identically to run_reference.py.
from run_reference import build_generator, generate, set_seed  # noqa: E402


def main():
    ap = ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=4.5)
    ap.add_argument("--image-size", type=int, choices=(256, 512), default=256)
    args = ap.parse_args()

    print(f"[batch_reference] loading model (image_size={args.image_size}) ...", flush=True)
    t0 = time.time()
    gen = build_generator(args.image_size, batch_size=1)
    print(f"[batch_reference] model ready in {time.time()-t0:.1f}s", flush=True)

    jobs = [json.loads(line) for line in sys.stdin if line.strip()]
    print(f"[batch_reference] {len(jobs)} jobs queued", flush=True)
    for i, job in enumerate(jobs, 1):
        out = Path(job["out"])
        if out.exists() and out.stat().st_size > 0:
            print(f"[batch_reference] {i}/{len(jobs)} {out.name} cached", flush=True)
            continue
        set_seed(int(job["seed"]))
        t1 = time.time()
        images = generate(gen, int(job["class_id"]), args.steps, args.temperature)
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images[0]).save(out)
        print(f"[batch_reference] {i}/{len(jobs)} {out.name} {time.time()-t1:.2f}s",
              flush=True)


if __name__ == "__main__":
    main()
