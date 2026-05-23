# Copyright 2026 edge_maskgit
# Licensed under the Apache License, Version 2.0
"""Milestone 1 reproducibility + parameter-count tests.

Run:  cd reference && python -m pytest tests/ -v
(The conda env `maskgit-ref` must have torch, numpy, pillow, pytest, ml_collections.)
"""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REF_DIR = Path(__file__).resolve().parent.parent
os.chdir(REF_DIR)
sys.path.insert(0, str(REF_DIR))

import export_weights as E       # noqa: E402
import run_reference            # noqa: E402
from maskgit.inference import ImageNet_class_conditional_generator  # noqa: E402

# Measured exact parameter counts of the converted official weights.
EXPECTED_TRANSFORMER_PARAMS = 172_457_193
EXPECTED_VQGAN_PARAMS = 54_515_587


@pytest.fixture(scope="session")
def generator():
    gen = ImageNet_class_conditional_generator(image_size=256)
    gen.maskgit_cf.eval_batch_size = 1
    return gen


def test_parameter_counts():
    """Loaded checkpoints must have the exact expected parameter counts."""
    t = E.load_ckpt("maskgit_imagenet256.ckpt")
    v = E.load_ckpt("tokenizer_imagenet256.ckpt")
    t_params = sum(int(x.numel()) for x in t.values())
    v_params = sum(int(x.numel()) for x in v.values())
    assert t_params == EXPECTED_TRANSFORMER_PARAMS, t_params
    assert v_params == EXPECTED_VQGAN_PARAMS, v_params


def test_transformer_shape_invariants():
    """Key architectural dims must match the (corrected) config: 16 heads, vocab 2025."""
    t = E.load_ckpt("maskgit_imagenet256.ckpt")
    assert t["tok_emb.weight"].shape == (2025, 768)        # vocab, hidden
    assert t["pos_emb"].shape == (257, 768)                # 256 tokens + 1 class
    # fused QKV is 3*hidden; 768 / 16 heads = head_dim 48
    assert t["blocks.0.MultiHeadAttention.in_proj_weight"].shape == (2304, 768)
    assert 768 % 16 == 0


def test_determinism_same_seed(generator):
    """Same (class, seed, steps) must produce bit-identical images."""
    run_reference.set_seed(42)
    a = run_reference.generate(generator, class_id=207, steps=8, temperature=4.5)
    run_reference.set_seed(42)
    b = run_reference.generate(generator, class_id=207, steps=8, temperature=4.5)
    assert np.array_equal(a, b)


def test_different_seed_differs(generator):
    """Different seeds should (almost surely) produce different images."""
    run_reference.set_seed(1)
    a = run_reference.generate(generator, class_id=207, steps=8, temperature=4.5)
    run_reference.set_seed(2)
    b = run_reference.generate(generator, class_id=207, steps=8, temperature=4.5)
    assert not np.array_equal(a, b)


def test_manifest_first_case_reproducible(generator):
    """Regenerating the first manifest case must reproduce its recorded md5."""
    manifest = json.loads((REF_DIR / "reference_outputs" / "manifest.json").read_text())
    case = manifest["cases"][0]
    run_reference.set_seed(case["seed"])
    imgs = run_reference.generate(
        generator, case["class_id"], manifest["steps"], manifest["temperature"])
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.fromarray(imgs[0]).save(buf, format="PNG")
    # Compare pixels to the stored reference (PNG md5 is encoder-stable here, but
    # pixel comparison is the robust check).
    ref = np.asarray(Image.open(REF_DIR / "reference_outputs" / case["file"]))
    assert np.array_equal(imgs[0], ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
