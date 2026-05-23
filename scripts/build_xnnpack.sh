#!/usr/bin/env bash
# Build XNNPACK from source into static libs + headers under third_party/xnn/.
# Bazel references the result via //third_party/xnn:xnn (cc_import).
# Run once after cloning the repo (the clone + .a artifacts are gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
X="$ROOT/third_party/XNNPACK"        # raw clone (gitignored, .bazelignore'd)
DST="$ROOT/third_party/xnn"          # wrapper consumed by Bazel

if [ ! -d "$X" ]; then
  echo "cloning XNNPACK..."
  git clone --depth 1 https://github.com/google/XNNPACK "$X"
fi

echo "configuring + building XNNPACK static lib..."
cmake -S "$X" -B "$X/build-static" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DXNNPACK_BUILD_TESTS=OFF \
  -DXNNPACK_BUILD_BENCHMARKS=OFF \
  -DXNNPACK_BUILD_ALL_MICROKERNELS=ON
cmake --build "$X/build-static" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" --target XNNPACK

echo "copying libs + headers to $DST ..."
mkdir -p "$DST/lib" "$DST/include"
cp "$X/build-static/libXNNPACK.a" \
   "$X/build-static/libxnnpack-microkernels-prod.a" \
   "$X/build-static/pthreadpool/libpthreadpool.a" \
   "$X/build-static/cpuinfo/libcpuinfo.a" \
   "$X/build-static/kleidiai/libkleidiai.a" "$DST/lib/"
cp "$X/include/xnnpack.h" "$X/include/experimental.h" "$DST/include/"
cp "$X/build-static/pthreadpool-source/include/pthreadpool.h" "$DST/include/"
echo "done. libs in $DST/lib, headers in $DST/include"
