#!/usr/bin/env bash
# M5 benchmark sweep. Runs `mg-generate --bench` across backends/precisions on the host
# and writes a Markdown report to benchmark/results/host.md.
#
# Env: RUNS (default 8), WARM (default 2).
#
# Device (Pixel 9 / Mali): build + push the OpenCL CLI, then run the same flags via adb:
#   ./scripts/build_android_generate_opencl.sh
#   adb push build-android/mg-generate-opencl /data/local/tmp/ && adb shell chmod 755 /data/local/tmp/mg-generate-opencl
#   adb shell "cd /data/local/tmp && ./mg-generate-opencl -m maskgit-256-gq8.gguf --backend opencl --bench --n-runs 5 --warmup 1"
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./bazel-bin/mg-generate
OUT=benchmark/results
mkdir -p "$OUT"
RUNS=${RUNS:-8}; WARM=${WARM:-2}

[ -x "$BIN" ] || { echo "build first: bazel build //:mg-generate"; exit 1; }

run() {  # backend  model  quant(optional)  title
    echo "## $4"; echo
    "$BIN" -m "$2" --backend "$1" ${3:+--quant "$3"} --bench --n-runs "$RUNS" --warmup "$WARM" \
        2>&1 | sed -n '/### Benchmark/,$p'
    echo
}

{
    echo "# Host benchmark (run_bench.sh, RUNS=$RUNS)"; echo
    # The reference scalar CPU backend is ~700 s/run — skipped unless BENCH_REFERENCE=1
    # (and then forced to a single run via RUNS=1 WARM=0 just for that one entry).
    if [ "${BENCH_REFERENCE:-0}" = 1 ] && [ -f models/maskgit-256-f32.gguf ]; then
        RUNS=1 WARM=0 run reference models/maskgit-256-f32.gguf "" "reference F32 (CPU, 1 run)"
    fi
    [ -f models/maskgit-256-q8.gguf ]  && run xnnpack   models/maskgit-256-q8.gguf  q8 "XNNPACK int8 (CPU)"
    [ -f models/maskgit-256-gq8.gguf ] && run opencl    models/maskgit-256-gq8.gguf "" "OpenCL ggml Q8_0 (GPU)"
    [ -f models/maskgit-256-gq4.gguf ] && run opencl    models/maskgit-256-gq4.gguf "" "OpenCL ggml Q4_K (GPU)"
} | tee "$OUT/host.md"
echo "wrote $OUT/host.md"
