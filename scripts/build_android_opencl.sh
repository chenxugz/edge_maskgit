#!/usr/bin/env bash
# Cross-compile the OpenCL verify binary for Android arm64 (Mali GPU).
# Links against the device's libOpenCL.so (pulled to third_party/android-libs);
# at runtime the Pixel's public libOpenCL.so resolves it. Uses the Khronos
# OpenCL headers in third_party/opencl-headers.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
HOST_TAG="$(ls "$NDK/toolchains/llvm/prebuilt/" | head -1)"
CXX="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android28-clang++"
LIBCL="$ROOT/third_party/android-libs/libOpenCL.so"

[ -x "$CXX" ] || { echo "missing NDK clang: $CXX"; exit 1; }
# Auto-fetch the Khronos headers + the device's libOpenCL.so if absent.
[ -f "$ROOT/third_party/opencl-headers/CL/cl.h" ] || \
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers "$ROOT/third_party/opencl-headers"
if [ ! -f "$LIBCL" ]; then
    mkdir -p "$ROOT/third_party/android-libs"
    echo "pulling libOpenCL.so from the connected device..."
    adb pull /vendor/lib64/libOpenCL.so "$LIBCL" || {
        echo "could not pull libOpenCL.so — device may not expose OpenCL"; exit 1; }
fi

mkdir -p build-android
SRCS=(
  src/mg-tensor.cpp src/mg-cpu/mg-cpu.cpp src/mg-model.cpp
  src/mg-transformer.cpp src/mg-opencl/mg-opencl.cpp
  tools/verify-opencl-transformer.cpp
)
echo "cross-compiling verify-opencl-transformer for arm64 (Mali OpenCL)..."
"$CXX" -std=c++17 -O3 -march=armv8.4a+dotprod -static-libstdc++ \
  -DCL_TARGET_OPENCL_VERSION=120 \
  -Iinclude -Ithird_party/opencl-headers \
  "${SRCS[@]}" "$LIBCL" \
  -o build-android/verify-opencl-transformer
file build-android/verify-opencl-transformer
echo "built build-android/verify-opencl-transformer"
