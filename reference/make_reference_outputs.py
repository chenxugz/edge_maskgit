#!/usr/bin/env python
# Copyright 2026 edge_maskgit
# Licensed under the Apache License, Version 2.0
"""Generate the 10 fixed (class_id, seed) reference images for Milestone 1.

Loads the model once and writes reference_outputs/<name>.png plus a manifest.json
recording the exact (class_id, seed, steps, temperature) and the image's md5, so
later runs (and the C++ runtime) can be checked for an exact match.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image

_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

import run_reference  # noqa: E402
from maskgit.inference import ImageNet_class_conditional_generator  # noqa: E402

# 10 fixed (class_id, seed, label) pairs. Classes chosen for visual diversity.
CASES = [
    (207, 42, "golden_retriever"),
    (933, 42, "cheeseburger"),
    (88, 42, "macaw"),
    (972, 42, "cliff"),
    (1, 7, "goldfish"),
    (130, 7, "flamingo"),
    (285, 123, "egyptian_cat"),
    (388, 123, "giant_panda"),
    (980, 99, "volcano"),
    (417, 99, "balloon"),
]
STEPS = 16          # paper-quality setting for the headline reference set
TEMPERATURE = 4.5


def md5_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    out_dir = _HERE / "reference_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ref] loading model ...", flush=True)
    gen = ImageNet_class_conditional_generator(image_size=256)
    gen.maskgit_cf.eval_batch_size = 1

    manifest = {"steps": STEPS, "temperature": TEMPERATURE, "image_size": 256, "cases": []}
    for class_id, seed, label in CASES:
        run_reference.set_seed(seed)
        imgs = run_reference.generate(gen, class_id, STEPS, TEMPERATURE)
        fname = f"{class_id:03d}_{label}_s{seed}.png"
        path = out_dir / fname
        Image.fromarray(imgs[0]).save(path)
        digest = md5_file(path)
        manifest["cases"].append(
            {"class_id": class_id, "seed": seed, "label": label,
             "file": fname, "md5": digest})
        print(f"[ref] {fname}  md5={digest}", flush=True)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[ref] wrote {len(CASES)} images + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
