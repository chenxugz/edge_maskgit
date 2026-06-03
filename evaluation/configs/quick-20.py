"""Quick-20 (per CLAUDE.md §4 M4): 20 classes x 50 images = 1000 images. Intended as
the standard "fast sanity" FID gate for each precision variant (target run-time per
variant: ~25-60 min depending on backend). 1000 samples is below the ~10k-50k typical
for paper-grade FID, but it gives a stable relative number for ranking variants.

Classes span a diverse mix (animal/object/scene/texture) chosen for visual variety —
not a "easy" subset. Seeds 0..49 per class so two runs of the same config sample the
same (class, seed) slots across backends, making the per-precision FID delta direct."""
CONFIG = {
    "resolution": 256,
    "classes": [
        207, 933,  88, 972, 281,
        417, 248, 510, 717, 850,
        977, 805, 644, 130, 949,
         40, 359, 689, 471, 105,
    ],
    "n_per_class": 50,
    "seed_base": 0,
}
