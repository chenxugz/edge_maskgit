#!/usr/bin/env bash
# Cross-compile XNNPACK for Android arm64-v8a (NDK r27) into static libs under
# third_party/xnn/lib-android/. Run after scripts/build_xnnpack.sh has cloned the
# source (third_party/XNNPACK).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
X="$ROOT/third_party/XNNPACK"
DST="$ROOT/third_party/xnn/lib-android"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
ABI=arm64-v8a
API=28

[ -d "$X" ] || { echo "missing $X — run scripts/build_xnnpack.sh first"; exit 1; }

echo "configuring XNNPACK for Android ($ABI, api $API)..."
cmake -S "$X" -B "$X/build-android" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="$ABI" \
  -DANDROID_PLATFORM="android-$API" \
  -DCMAKE_BUILD_TYPE=Release \
  -DXNNPACK_BUILD_TESTS=OFF \
  -DXNNPACK_BUILD_BENCHMARKS=OFF \
  -DXNNPACK_BUILD_ALL_MICROKERNELS=ON
cmake --build "$X/build-android" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" --target XNNPACK

echo "copying android libs -> $DST"
mkdir -p "$DST"
cp "$X/build-android/libXNNPACK.a" \
   "$X/build-android/libxnnpack-microkernels-prod.a" \
   "$X/build-android/pthreadpool/libpthreadpool.a" \
   "$X/build-android/cpuinfo/libcpuinfo.a" \
   "$X/build-android/kleidiai/libkleidiai.a" "$DST/"
echo "done. android libs in $DST"
ls -lh "$DST"
