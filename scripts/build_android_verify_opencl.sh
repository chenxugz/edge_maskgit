#!/usr/bin/env bash
# Cross-compile verify-opencl-transformer for Android arm64-v8a (Mali OpenCL).
# Mirrors scripts/build_android_generate_opencl.sh but with the verify tool's sources.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
HOST_TAG="$(ls "$NDK/toolchains/llvm/prebuilt/" | head -1)"
CXX="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android28-clang++"
LIBCL="$ROOT/third_party/android-libs/libOpenCL.so"

mkdir -p build-android
SRCS=(
  src/mg-tensor.cpp src/mg-cpu/mg-cpu.cpp src/mg-model.cpp
  src/mg-transformer.cpp src/mg-vqgan.cpp
  src/mg-opencl/mg-opencl.cpp
  tools/verify-opencl-transformer.cpp
)
echo "cross-compiling verify-opencl-transformer for arm64-v8a..."
"$CXX" -std=c++17 -O3 -march=armv8.4a+dotprod -static-libstdc++ \
  -DMG_HAS_OPENCL -DCL_TARGET_OPENCL_VERSION=120 \
  -Iinclude -Ithird_party/stb -Ithird_party/opencl-headers \
  "${SRCS[@]}" \
  "$LIBCL" -llog \
  -o build-android/verify-opencl-transformer
file build-android/verify-opencl-transformer
