"""
Milestone 4 — Evaluation runner.

Orchestrates batch image generation via `mg-generate` (or the PyTorch oracle when
``--backend reference``) and computes three metrics in one InceptionV3 pass:

  * Inception Score (IS, mean ± std over 10 splits) — no-reference; quantifies how
    confidently a classifier predicts a single class AND how diverse those predictions
    are across the set. The MaskGIT paper reports ~182 IS on full 1000-class ImageNet
    sampling — Quick-5/Quick-20 will be much lower simply because the class set is
    narrower; the number is for **relative** comparison between runtime variants.
  * Top-1 / Top-5 ImageNet classifier accuracy against the target class encoded in the
    filename (cheap: argmax of the same InceptionV3 softmax). A class-conditioning
    sanity check — if a runtime variant produces images the classifier doesn't believe
    are the requested class, accuracy collapses.
  * PSNR (peak signal-to-noise ratio, dB) against a paired oracle folder. Optional;
    skipped if no oracle images exist for the matched (class, seed). Set the oracle
    folder via ``--oracle-dir``, or generate one with ``--backend reference``.

Resumable: skips PNGs that already exist for the (class, seed) under the configured
output directory.

Usage:
    # 1. Generate the PyTorch reference once (slow but the PSNR baseline)
    python evaluation/eval_runner.py --config quick-5 --backend reference --skip-eval

    # 2. Evaluate each runtime variant, paired vs. the reference for PSNR
    python evaluation/eval_runner.py --config quick-5 --backend xnnpack
    python evaluation/eval_runner.py --config quick-20 --backend opencl \\
        --model models/maskgit-256-gq8.gguf

    # 3. Or just (re-)compute metrics on already-generated images
    python evaluation/eval_runner.py --config quick-20 --skip-generate

Configs are YAML-free dicts in evaluation/configs/*.py (a CONFIG dict per file). They
fix the (class_id, seed) pairs so two runs of the same config sample the same RNG slots
across backends, which is what makes per-variant deltas direct.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIN = ROOT / "bazel-bin" / "mg-generate"


def load_config(name: str) -> dict:
    """Load CONFIG dict from evaluation/configs/<name>.py."""
    p = ROOT / "evaluation" / "configs" / f"{name}.py"
    if not p.exists():
        raise FileNotFoundError(f"no config at {p}")
    spec = importlib.util.spec_from_file_location(f"cfg_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


def jobs_for_config(cfg: dict) -> list[tuple[int, int]]:
    """Return the full list of (class_id, seed) pairs for the config."""
    classes = cfg["classes"]
    n = cfg["n_per_class"]
    seed_base = cfg.get("seed_base", 0)
    return [(c, seed_base + i) for c in classes for i in range(n)]


def out_path(out_dir: Path, class_id: int, seed: int) -> Path:
    return out_dir / f"c{class_id:03d}_s{seed:05d}.png"


def gen_one(args, mg_bin: str, model: str, backend: str, steps: int, out: Path,
            oracle: bool, ref_python: str) -> tuple[Path, float, str | None]:
    """Run mg-generate (or the PyTorch oracle if `oracle`) for one (class, seed).
    The oracle path needs the reference/ env (`conda activate maskgit-ref`); pass its
    python via `ref_python` because the runner itself may live in a different env."""
    class_id, seed = args
    if out.exists() and out.stat().st_size > 0:
        return (out, 0.0, None)  # skip; already present
    if oracle:
        cmd = [ref_python, str(ROOT / "reference" / "run_reference.py"),
               "--class-id", str(class_id), "--seed", str(seed),
               "--steps", str(steps), "-o", str(out)]
        cwd = str(ROOT / "reference")  # run_reference.py imports `maskgit.*` relative to here
    else:
        cmd = [mg_bin, "-m", model, "--class-id", str(class_id), "--seed", str(seed),
               "--steps", str(steps), "--backend", backend, "-o", str(out)]
        cwd = None
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    dt = time.time() - t0
    if p.returncode != 0:
        return (out, dt, f"exit={p.returncode}: {p.stderr.strip()[:200]}")
    return (out, dt, None)


def _run_oracle_batch(todo: list[tuple[int, int]], out_dir: Path, steps: int,
                      ref_python: str) -> int:
    """Pipe jobs into reference/batch_reference.py so the model loads exactly once
    for the whole batch (vs. once per image with run_reference.py)."""
    cmd = [ref_python, str(ROOT / "reference" / "batch_reference.py"),
           "--steps", str(steps)]
    job_lines = "\n".join(
        json.dumps({"class_id": c, "seed": s, "out": str(out_path(out_dir, c, s))})
        for (c, s) in todo
    ) + "\n"
    print(f"[gen] launching oracle batch ({len(todo)} jobs)")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          cwd=str(ROOT / "reference"))
    p.stdin.write(job_lines); p.stdin.close()
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
    return p.wait()


def run_batch(jobs: list[tuple[int, int]], out_dir: Path, mg_bin: str, model: str,
              backend: str, steps: int, workers: int, oracle: bool = False,
              ref_python: str = sys.executable):
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(c, s) for (c, s) in jobs if not out_path(out_dir, c, s).exists()]
    label = "oracle (PyTorch)" if oracle else f"backend={backend}"
    print(f"[gen] {len(jobs)} total; {len(jobs) - len(todo)} cached; {len(todo)} to generate "
          f"on {label}" + ("" if oracle else f" with {workers} workers"))
    if not todo:
        return
    if oracle:
        # Single batch_reference.py process — model load amortized across all jobs.
        rc = _run_oracle_batch(todo, out_dir, steps, ref_python)
        if rc != 0:
            print(f"[gen] oracle batch exited rc={rc}")
        return
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(gen_one, j, mg_bin, model, backend, steps,
                             out_path(out_dir, *j), False, ref_python) for j in todo]
        done = 0; errors = []
        for f in cf.as_completed(futures):
            done += 1
            p, dt, err = f.result()
            if err:
                errors.append((p.name, err))
            if done % max(1, len(todo) // 20) == 0 or done == len(todo):
                print(f"  [gen] {done}/{len(todo)} done", flush=True)
        if errors:
            print(f"[gen] {len(errors)} errors — first 3:")
            for n, e in errors[:3]:
                print(f"    {n}: {e}")


def _pick_device() -> str:
    """Pick the best available torch device (MPS on Apple Silicon, then CUDA, else CPU)."""
    try:
        import torch
        if torch.cuda.is_available(): return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    except Exception:
        pass
    return "cpu"


# Filename schema is c{class:03d}_s{seed:05d}.png — see out_path() above.
_FNAME_RE = __import__("re").compile(r"^c(\d{3})_s(\d{5})\.png$")


def _list_images(out_dir: Path) -> list[tuple[Path, int, int]]:
    """Return [(path, class_id, seed)] for every well-named PNG in out_dir."""
    items = []
    for p in sorted(out_dir.glob("c*_s*.png")):
        m = _FNAME_RE.match(p.name)
        if not m:
            continue
        items.append((p, int(m.group(1)), int(m.group(2))))
    return items


def _inception_softmax(items: list[tuple[Path, int, int]], device_str: str,
                       batch_size: int = 32) -> "tuple[any, list[int]]":
    """Run pretrained InceptionV3 over the images; return (softmax probs [N,1000],
    target class ids [N]). Standard IS preprocessing: 299x299 bilinear, ImageNet
    normalize, RGB. We deliberately use weights='IMAGENET1K_V1' (the original
    InceptionV3 weights) — IS literature is pinned to this checkpoint."""
    import torch
    import torch.nn.functional as F
    from torchvision import models, transforms
    from PIL import Image

    device = torch.device(device_str)
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, aux_logits=True)
    model.fc = model.fc  # keep classifier
    model.eval().to(device)

    # Manual preprocess — the weights' default transform also does center-crop, which we
    # don't want (our images are already 256x256 generations; we just need 299x299).
    preprocess = transforms.Compose([
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),  # [0,1] CHW
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    probs_all, targets = [], []
    batch_imgs, batch_tgt = [], []

    @torch.inference_mode()
    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(device)
        # In eval(), inception_v3 returns just the main logits tensor.
        logits = model(x)
        if isinstance(logits, tuple):  # safety (some torchvision versions)
            logits = logits[0]
        probs_all.append(F.softmax(logits, dim=1).float().cpu())
        targets.extend(batch_tgt)
        batch_imgs.clear(); batch_tgt.clear()

    for path, cls, _seed in items:
        with Image.open(path) as im:
            im = im.convert("RGB")
            batch_imgs.append(preprocess(im))
        batch_tgt.append(cls)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    probs = torch.cat(probs_all, dim=0) if probs_all else torch.empty(0, 1000)
    return probs, targets


def _inception_score(probs, splits: int = 10) -> tuple[float, float]:
    """Standard split-based IS: split N images into `splits` groups, compute
    exp(mean_x KL(p(y|x) || mean_x p(y|x))) per group, return (mean, std). For very
    small sets we fall back to a single split — std becomes 0 but mean is still valid."""
    import torch
    N = probs.shape[0]
    if N == 0:
        return float("nan"), 0.0
    s = max(1, min(splits, N))  # can't have more splits than samples
    chunk = N // s
    scores = []
    for i in range(s):
        lo, hi = i * chunk, (i + 1) * chunk if i < s - 1 else N
        p = probs[lo:hi]                              # [n, 1000]
        py = p.mean(dim=0, keepdim=True).clamp(min=1e-16)
        kl = (p.clamp(min=1e-16) * (p.clamp(min=1e-16).log() - py.log())).sum(dim=1)
        scores.append(kl.mean().exp().item())
    return float(sum(scores) / len(scores)), float(torch.tensor(scores).std(unbiased=False).item())


def _top_k_accuracy(probs, targets: list[int], ks=(1, 5)) -> dict:
    """Top-k against target class encoded in filename. Cheap given probs."""
    import torch
    if probs.shape[0] == 0:
        return {f"top{k}": float("nan") for k in ks}
    tgt = torch.tensor(targets, dtype=torch.long)
    out = {}
    for k in ks:
        topk = probs.topk(k, dim=1).indices                 # [N, k]
        correct = (topk == tgt.unsqueeze(1)).any(dim=1)
        out[f"top{k}"] = float(correct.float().mean().item())
    return out


def _psnr_paired(out_dir: Path, oracle_dir: Path) -> dict:
    """Per-image PSNR vs an oracle folder, matched by filename. Returns mean/min/std/N
    over the intersection. Skips silently if no pairs found."""
    import numpy as np
    from PIL import Image
    psnrs = []
    pairs = 0
    for path, cls, seed in _list_images(out_dir):
        ref = oracle_dir / path.name
        if not ref.exists():
            continue
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(ref ).convert("RGB"), dtype=np.float64)
        if a.shape != b.shape:
            continue
        mse = float(((a - b) ** 2).mean())
        if mse <= 0.0:
            psnrs.append(float("inf"))  # identical (rare; same backend re-run)
        else:
            psnrs.append(20.0 * np.log10(255.0 / np.sqrt(mse)))
        pairs += 1
    if not psnrs:
        return {"n_pairs": 0}
    finite = [v for v in psnrs if v != float("inf")]
    return {
        "n_pairs": pairs,
        "psnr_mean": float(sum(finite) / max(1, len(finite))),
        "psnr_min":  float(min(finite)) if finite else float("inf"),
        "psnr_max":  float(max(finite)) if finite else float("inf"),
    }


def compute_metrics(out_dir: Path, oracle_dir: Path | None) -> dict:
    """One InceptionV3 pass over out_dir for IS + top-1/top-5, then optional paired
    PSNR vs oracle_dir."""
    items = _list_images(out_dir)
    device = _pick_device()
    print(f"[eval] {len(items)} images in {out_dir} (device={device})")
    t0 = time.time()
    probs, targets = _inception_softmax(items, device)
    is_mean, is_std = _inception_score(probs, splits=10)
    acc = _top_k_accuracy(probs, targets)
    inception_seconds = time.time() - t0
    metrics = {
        "n_images": len(items),
        "is_mean": is_mean, "is_std": is_std,
        "top1": acc["top1"], "top5": acc["top5"],
        "inception_seconds": inception_seconds,
    }
    if oracle_dir is not None and oracle_dir.exists():
        print(f"[eval] PSNR pairing vs oracle: {oracle_dir}")
        metrics.update(_psnr_paired(out_dir, oracle_dir))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="quick-5", help="config name under evaluation/configs/")
    ap.add_argument("--model", default=str(ROOT / "models" / "maskgit-256-q8.gguf"),
                    help="GGUF model path (ignored when --backend reference)")
    ap.add_argument("--backend", default="xnnpack", choices=["reference", "xnnpack", "opencl"],
                    help="`reference` invokes reference/run_reference.py (PyTorch oracle); "
                         "others invoke mg-generate")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4, help="parallel generation workers "
                    "(forced to 1 for --backend reference)")
    ap.add_argument("--mg-bin", default=str(DEFAULT_BIN))
    ap.add_argument("--ref-python", default=os.environ.get(
        "MASKGIT_REF_PYTHON",
        str(Path.home() / "anaconda3" / "envs" / "maskgit-ref" / "bin" / "python")),
        help="Python for the PyTorch oracle (needs the reference/requirements.txt env). "
             "Override with $MASKGIT_REF_PYTHON or this flag.")
    ap.add_argument("--out", default=None, help="output dir (default: results/<config>/<backend>)")
    ap.add_argument("--oracle-dir", default=None,
                    help="oracle folder for paired PSNR (default: results/<config>/reference)")
    ap.add_argument("--skip-generate", action="store_true", help="just (re-)compute metrics")
    ap.add_argument("--skip-eval", action="store_true", help="just generate, skip metrics")
    args = ap.parse_args()

    cfg = load_config(args.config)
    oracle_mode = (args.backend == "reference")
    out_dir = Path(args.out) if args.out else ROOT / "evaluation" / "results" / args.config / args.backend
    oracle_dir = Path(args.oracle_dir) if args.oracle_dir else \
        ROOT / "evaluation" / "results" / args.config / "reference"
    print(f"[run] config={args.config} backend={args.backend} "
          f"model={Path(args.model).name if not oracle_mode else 'pytorch-oracle'}")
    print(f"      classes={len(cfg['classes'])} n_per_class={cfg['n_per_class']} -> "
          f"{len(cfg['classes'])*cfg['n_per_class']} images")
    print(f"      output -> {out_dir}")

    jobs = jobs_for_config(cfg)
    if not args.skip_generate:
        if not oracle_mode:
            if not Path(args.mg_bin).exists():
                sys.exit(f"mg-generate not found at {args.mg_bin}; run `bazel build //:mg-generate`")
            if not Path(args.model).exists():
                sys.exit(f"model not found at {args.model}")
        t0 = time.time()
        run_batch(jobs, out_dir, args.mg_bin, args.model, args.backend, args.steps,
                  args.workers, oracle=oracle_mode, ref_python=args.ref_python)
        print(f"[gen] total wall: {time.time()-t0:.1f}s")

    if args.skip_eval:
        return
    # PSNR only applies when there's a separate oracle folder; for the reference run
    # itself, the "pair" would be the image against itself (PSNR=inf), so skip.
    psnr_oracle = None if oracle_mode else oracle_dir
    metrics = compute_metrics(out_dir, oracle_dir=psnr_oracle)
    summary = {
        "config": args.config, "backend": args.backend,
        "model": Path(args.model).name if not oracle_mode else "pytorch-oracle",
        "steps": args.steps, "metrics": metrics,
    }
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir.parent / f"{args.backend}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print()
    print(f"[eval] === {args.config} / {args.backend} ({metrics['n_images']} images) ===")
    print(f"  IS         = {metrics['is_mean']:.3f} ± {metrics['is_std']:.3f}")
    print(f"  Top-1 acc  = {100*metrics['top1']:.2f}%   "
          f"Top-5 acc = {100*metrics['top5']:.2f}%")
    if "n_pairs" in metrics and metrics["n_pairs"]:
        print(f"  PSNR(oracle ↔ runtime) = {metrics['psnr_mean']:.2f} dB  "
              f"(min {metrics['psnr_min']:.2f}, max {metrics['psnr_max']:.2f}, "
              f"n={metrics['n_pairs']})")
    elif not oracle_mode:
        print(f"  PSNR skipped — no oracle images at {oracle_dir} "
              f"(generate with `--backend reference` first)")
    print(f"  saved -> {summary_path}")


if __name__ == "__main__":
    main()
