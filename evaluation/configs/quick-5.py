"""Quick-5: 5 classes x 8 images = 40 images. Smoke test (~few minutes per variant)
for the eval-runner + clean-fid plumbing. Not enough for a stable FID number — use
quick-20 (or larger) for real numbers — but exercises the whole path."""
CONFIG = {
    "resolution": 256,
    # 5 well-known ImageNet classes (mix of animal / object / scene)
    "classes": [
        207,   # golden retriever
        933,   # cheeseburger
         88,   # macaw
        972,   # cliff
        281,   # tabby cat
    ],
    "n_per_class": 8,
    "seed_base": 0,
}
