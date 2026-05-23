# Milestone 1 — MaskGIT Reference Oracle

Ground-truth reference for the custom C/C++ MaskGIT runtime: takes an ImageNet
class id and produces a generated 256×256 image, using the **genuine official
pretrained weights**. All later milestones (M2 runtime, M3 numerical
verification, M4 evaluation) are validated against this.

## ⚠️ Important deviation from CLAUDE.md: PyTorch, not JAX

CLAUDE.md §M1 specifies a **JAX/Flax** reference loading the official checkpoints
from `gs://maskgit-public`. **That bucket has been fully decommissioned** — every
object (checkpoints *and* the demo images) now returns `AccessDenied`, and the
original flax/msgpack checkpoints are no longer obtainable anywhere public.

The only surviving copy of the *genuine official weights* is a faithful PyTorch
conversion: [`hmorimitsu/maskgit-torch`](https://github.com/hmorimitsu/maskgit-torch),
which hosts `tokenizer_imagenet256.ckpt` + `maskgit_imagenet256.ckpt` on GitHub
Releases with notebooks documenting the exact JAX↔PyTorch parameter mapping.

So this reference is **PyTorch-based**. The model math is unchanged (24-layer
bidirectional transformer, 16 heads, VQGAN decoder); only the framework differs.
The vendored `maskgit/` package is from that repo (Apache-2.0).

## Corrections to CLAUDE.md prose

The authoritative upstream config differs from CLAUDE.md's prose:

| Item | CLAUDE.md says | Actual (checkpoint) |
|---|---|---|
| Attention heads | 8 (head_dim 96) | **16** (head_dim 48) |
| Default decode steps | 8 | upstream default 16 (we keep 8 as CLI default) |
| Transformer vocab | — | 1024 codebook + 1000 class + 1 = **2025** |
| LayerNorm placement | pre-norm (diagram) | **post-norm** (BERT-style, after residual add) |
| Choice temperature | — | **4.5** |
| VQGAN ResBlock | standard residual | shortcut conv applies to *processed* x (faithful quirk) |

## Setup

```bash
conda create -n maskgit-ref python=3.11 -y
conda activate maskgit-ref
pip install -r requirements.txt
./download_checkpoints.sh          # ~880 MB into checkpoints/
```

## Usage

```bash
# Single image (golden retriever) — by id or by name
python run_reference.py --class-id 207 --seed 42 --steps 8 -o output.png
python run_reference.py --class-name "golden retriever" --seed 42 --steps 8 -o output.png

# Flags: --class-id 0-999 | --class-name <substring>  --steps  --seed
#        --temperature  --batch-size  --image-size {256,512}
# Class names come from tools/imagenet_classes.json (id -> wnid + name).

# Regenerate the 10 fixed reference images + manifest.json
python make_reference_outputs.py

# Export weights to GGUF-named .npz for the C++ runtime (+ lossless round-trip check)
python export_weights.py --verify

# Tests (determinism + parameter counts + manifest reproducibility)
python -m pytest tests/ -v
```

## Layout

```
reference/
├── run_reference.py          # end-to-end inference CLI (M1 deliverable)
├── export_weights.py         # weights -> GGUF-named .npz + metadata.json (--verify round-trips)
├── make_reference_outputs.py # generates the 10 fixed reference images + manifest
├── download_checkpoints.sh   # fetch converted official weights
├── requirements.txt          # pinned deps (Python 3.11)
├── Dockerfile                # reproducible CPU environment
├── maskgit/                  # vendored PyTorch model package (hmorimitsu/maskgit-torch)
├── checkpoints/              # *.ckpt (downloaded, gitignored)
├── export/                   # *.npz weight dump + metadata.json (gitignored)
├── reference_outputs/        # 10 reference PNGs + manifest.json
└── tests/test_reproducibility.py
```

## Exported weights (`export/`)

`export_weights.py` writes GGUF-convention tensors (CLAUDE.md §2.5):
- `maskgit_transformer_f32.npz` — fused QKV **split** into `blk.{i}.attn_q/k/v`,
  output tied to `token_embd` (logits = h·tok_embᵀ + `output.bias`).
- `maskgit_vqgan_f32.npz` — codebook + decoder (+ encoder), `vqgan.` prefix.
- `metadata.json` — hyperparameters, per-tensor shapes/dtypes, naming notes.

`--verify` reloads the `.npz`, rebuilds the models, and asserts **bit-identical**
generation vs. the original checkpoints (max_pixel_diff = 0).
