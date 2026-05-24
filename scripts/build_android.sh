#!/usr/bin/env bash
# Cross-compile the mg-generate runtime for Android arm64-v8a with the NDK r27
# clang toolchain, linking the Android XNNPACK static libs.
# Prereqs: scripts/build_xnnpack.sh (clone) + scripts/build_xnnpack_android.sh.
# Output: build-android/mg-generate (standalone PIE, self-contained libc++).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
HOST_TAG="$(ls "$NDK/toolchains/llvm/prebuilt/" | head -1)"   # e.g. darwin-x86_64
CXX="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android28-clang++"
XNN="$ROOT/third_party/xnn"

[ -x "$CXX" ] || { echo "missing NDK clang: $CXX"; exit 1; }
[ -d "$XNN/lib-android" ] || { echo "missing Android XNNPACK libs — run scripts/build_xnnpack_android.sh"; exit 1; }

mkdir -p build-android
SRCS=(
  src/mg-tensor.cpp src/mg-cpu/mg-cpu.cpp src/mg-model.cpp
  src/mg-transformer.cpp src/mg-vqgan.cpp src/mg-generate.cpp
  src/mg-xnn.cpp src/mg-xnn-vqgan.cpp tools/main.cpp
)

echo "cross-compiling mg-generate for arm64-v8a..."
"$CXX" -std=c++17 -O3 -march=armv8.4a+dotprod -static-libstdc++ \
  -Iinclude -Ithird_party/stb -I"$XNN/include" \
  "${SRCS[@]}" \
  -Wl,--start-group \
    "$XNN/lib-android/libXNNPACK.a" \
    "$XNN/lib-android/libxnnpack-microkernels-prod.a" \
    "$XNN/lib-android/libpthreadpool.a" \
    "$XNN/lib-android/libcpuinfo.a" \
    "$XNN/lib-android/libkleidiai.a" \
  -Wl,--end-group \
  -llog \
  -o build-android/mg-generate

file build-android/mg-generate
echo "built build-android/mg-generate"
