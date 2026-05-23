#!/usr/bin/env bash
# Download the converted official MaskGIT ImageNet-256 weights.
#
# The original Google checkpoints (https://storage.googleapis.com/maskgit-public/
# checkpoints/{maskgit,tokenizer}_imagenet256_checkpoint) are NO LONGER PUBLIC
# (the bucket returns AccessDenied). We use the genuine official weights converted
# to PyTorch by hmorimitsu/maskgit-torch instead.
set -euo pipefail
cd "$(dirname "$0")/checkpoints"
BASE="https://github.com/hmorimitsu/maskgit-torch/releases/download/weights"
for f in tokenizer_imagenet256.ckpt maskgit_imagenet256.ckpt; do
  if [ -f "$f" ]; then
    echo "exists: $f (skipping)"
  else
    echo "downloading: $f"
    curl -L -o "$f" "$BASE/$f"
  fi
done
echo "done. checkpoints in $(pwd)"
