#!/usr/bin/env bash
# Cross-compile the full mg-generate CLI for Android arm64-v8a with BOTH the XNNPACK
# CPU backend and the OpenCL GPU backend, so `--backend opencl` runs the whole
# class-id -> image pipeline on the phone's Mali/Adreno GPU.
# Prereqs: scripts/build_xnnpack_android.sh (XNNPACK libs) + scripts/build_android_opencl.sh
# (fetches the Khronos headers + pulls the device libOpenCL.so).
# Output: build-android/mg-generate-opencl (standalone PIE, self-contained libc++).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
HOST_TAG="$(ls "$NDK/toolchains/llvm/prebuilt/" | head -1)"
CXX="$NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android28-clang++"
XNN="$ROOT/third_party/xnn"
LIBCL="$ROOT/third_party/android-libs/libOpenCL.so"

[ -x "$CXX" ] || { echo "missing NDK clang: $CXX"; exit 1; }
[ -d "$XNN/lib-android" ] || { echo "missing Android XNNPACK libs — run scripts/build_xnnpack_android.sh"; exit 1; }
[ -f "$ROOT/third_party/opencl-headers/CL/cl.h" ] || { echo "missing OpenCL headers — run scripts/build_android_opencl.sh"; exit 1; }
[ -f "$LIBCL" ] || { echo "missing libOpenCL.so — run scripts/build_android_opencl.sh (with device connected)"; exit 1; }

mkdir -p build-android
SRCS=(
  src/mg-tensor.cpp src/mg-cpu/mg-cpu.cpp src/mg-model.cpp
  src/mg-transformer.cpp src/mg-vqgan.cpp src/mg-generate.cpp
  src/mg-xnn.cpp src/mg-xnn-vqgan.cpp src/mg-opencl/mg-opencl.cpp
  tools/main.cpp
)

echo "cross-compiling mg-generate-opencl (XNNPACK + OpenCL) for arm64-v8a..."
"$CXX" -std=c++17 -O3 -march=armv8.4a+dotprod -static-libstdc++ \
  -DMG_HAS_OPENCL -DCL_TARGET_OPENCL_VERSION=120 \
  -Iinclude -Ithird_party/stb -I"$XNN/include" -Ithird_party/opencl-headers \
  "${SRCS[@]}" \
  -Wl,--start-group \
    "$XNN/lib-android/libXNNPACK.a" \
    "$XNN/lib-android/libxnnpack-microkernels-prod.a" \
    "$XNN/lib-android/libpthreadpool.a" \
    "$XNN/lib-android/libcpuinfo.a" \
    "$XNN/lib-android/libkleidiai.a" \
  -Wl,--end-group \
  "$LIBCL" -llog \
  -o build-android/mg-generate-opencl

file build-android/mg-generate-opencl
echo "built build-android/mg-generate-opencl"
