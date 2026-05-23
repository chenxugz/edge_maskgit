#!/usr/bin/env python
# Copyright 2026 edge_maskgit
# Licensed under the Apache License, Version 2.0
"""MaskGIT reference inference (PyTorch oracle).

End-to-end class-conditional generation: ImageNet class id -> iterative masked
decoding (cosine schedule + Gumbel confidence sampling) -> VQGAN decode -> PNG.

This is the Milestone 1 ground-truth oracle for the custom C/C++ runtime.

NOTE: The original Google JAX/Flax checkpoints (gs://maskgit-public) are no
longer publicly available, so this reference uses the faithful PyTorch port and
the genuine converted official weights from hmorimitsu/maskgit-torch. The model
math (24-layer bidirectional transformer, 16 heads, VQGAN decoder) is unchanged.
"""
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

# Run relative to this file so the vendored package + ./checkpoints resolve
# regardless of the caller's cwd.
_HERE = Path(__file__).resolve().parent
os.chdir(_HERE)
sys.path.insert(0, str(_HERE))

from maskgit.inference import ImageNet_class_conditional_generator  # noqa: E402

# ImageNet-1k class-id -> name map (tools/imagenet_classes.json at repo root).
_CLASS_MAP_PATH = _HERE.parent / "tools" / "imagenet_classes.json"


def load_class_map() -> dict:
    if _CLASS_MAP_PATH.exists():
        return json.loads(_CLASS_MAP_PATH.read_text())
    return {}


def class_name(class_id: int) -> str:
    entry = load_class_map().get(str(class_id))
    return entry["name"] if entry else "<unknown>"


def resolve_class_name(query: str) -> int:
    """Resolve a class name (exact, else substring) to its class id."""
    cmap = load_class_map()
    if not cmap:
        raise SystemExit(f"class map not found at {_CLASS_MAP_PATH}; use --class-id")
    q = query.strip().lower()
    exact = [int(i) for i, e in cmap.items() if e["name"].lower() == q]
    if exact:
        return exact[0]
    matches = [(int(i), e["name"]) for i, e in cmap.items() if q in e["name"].lower()]
    if not matches:
        raise SystemExit(f"no ImageNet class matches '{query}'")
    if len(matches) > 1:
        preview = ", ".join(f"{i}={n}" for i, n in sorted(matches)[:10])
        raise SystemExit(f"'{query}' is ambiguous ({len(matches)} matches): {preview} ...")
    return matches[0][0]


def set_seed(seed: int) -> None:
    """Make generation deterministic for a fixed seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_generator(image_size: int, batch_size: int) -> ImageNet_class_conditional_generator:
    gen = ImageNet_class_conditional_generator(image_size=image_size)
    gen.maskgit_cf.eval_batch_size = batch_size
    return gen


def generate(
    gen: ImageNet_class_conditional_generator,
    class_id: int,
    steps: int,
    temperature: float,
) -> np.ndarray:
    """Return a uint8 [batch, H, W, 3] array of generated images."""
    gen.maskgit_cf.sample_choice_temperature = temperature
    input_tokens = gen.create_input_tokens_normal(class_id)
    with torch.no_grad():
        results = gen.generate_samples(input_tokens, num_iterations=steps)
    results = results.detach().cpu().numpy()              # [b, 3, H, W]
    results = np.transpose(results, (0, 2, 3, 1))         # [b, H, W, 3]
    return (np.clip(results, 0.0, 1.0) * 255).astype(np.uint8)


def _init_parser() -> ArgumentParser:
    p = ArgumentParser(description="MaskGIT reference inference (PyTorch oracle)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--class-id", type=int, help="ImageNet class id (0-999)")
    g.add_argument("--class-name", type=str,
                   help="ImageNet class name (exact or substring), e.g. 'golden retriever'")
    p.add_argument("--steps", type=int, default=8, help="Iterative decoding steps")
    p.add_argument("--seed", type=int, default=42, help="RNG seed (deterministic)")
    p.add_argument("--temperature", type=float, default=4.5,
                   help="Sampling choice temperature (default: paper value 4.5)")
    p.add_argument("--batch-size", type=int, default=1, help="Images to generate at once")
    p.add_argument("--image-size", type=int, choices=(256, 512), default=256)
    p.add_argument("-o", "--output", type=str, default="output.png",
                   help="Output PNG path (batch>1 appends _<i>)")
    return p


def main(args: Namespace) -> None:
    class_id = args.class_id if args.class_id is not None else resolve_class_name(args.class_name)
    if not (0 <= class_id <= 999):
        raise SystemExit(f"class id must be in [0, 999], got {class_id}")
    label = class_name(class_id)
    set_seed(args.seed)
    print(f"[run_reference] loading model (image_size={args.image_size}) ...", flush=True)
    gen = build_generator(args.image_size, args.batch_size)
    print(f"[run_reference] generating class={class_id} ({label}) steps={args.steps} "
          f"seed={args.seed} temp={args.temperature} ...", flush=True)
    args.class_id = class_id
    images = generate(gen, args.class_id, args.steps, args.temperature)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if images.shape[0] == 1:
        Image.fromarray(images[0]).save(out)
        print(f"[run_reference] wrote {out}")
    else:
        for i, img in enumerate(images):
            p = out.with_name(f"{out.stem}_{i}{out.suffix}")
            Image.fromarray(img).save(p)
            print(f"[run_reference] wrote {p}")


if __name__ == "__main__":
    main(_init_parser().parse_args())
