#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$ROOT_DIR/app/src/main/rust/ggufllm"
JNILIBS_DIR="$ROOT_DIR/app/src/main/jniLibs"

# Locate Android SDK / NDK
SDK_DIR="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
if [[ -z "$SDK_DIR" && -f "$ROOT_DIR/local.properties" ]]; then
    SDK_DIR="$(grep -E '^sdk\.dir=' "$ROOT_DIR/local.properties" | cut -d= -f2- | tr -d '\r' || true)"
fi
if [[ -z "$SDK_DIR" && -d "$HOME/Android/Sdk" ]]; then
    SDK_DIR="$HOME/Android/Sdk"
fi

NDK_DIR="${ANDROID_NDK_HOME:-}"
if [[ -z "$NDK_DIR" && -n "$SDK_DIR" && -d "$SDK_DIR/ndk" ]]; then
    NDK_DIR="$(ls -d "$SDK_DIR"/ndk/* 2>/dev/null | sort -V | tail -n 1 || true)"
fi

if [[ -z "$NDK_DIR" || ! -d "$NDK_DIR" ]]; then
    echo "ERROR: Android NDK not found. Set ANDROID_NDK_HOME or install NDK in $SDK_DIR/ndk" >&2
    exit 1
fi

HOST_TAG="linux-x86_64"
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) HOST_TAG="darwin-x86_64" ;;
    Darwin-x86_64) HOST_TAG="darwin-x86_64" ;;
    Linux-x86_64) HOST_TAG="linux-x86_64" ;;
esac
if [[ -d "$NDK_DIR/toolchains/llvm/prebuilt/darwin-arm64" ]]; then
    HOST_TAG="darwin-arm64"
fi

TOOLCHAIN_BIN="$NDK_DIR/toolchains/llvm/prebuilt/$HOST_TAG/bin"
if [[ ! -d "$TOOLCHAIN_BIN" ]]; then
    echo "ERROR: NDK toolchain bin not found at $TOOLCHAIN_BIN" >&2
    exit 1
fi

API_LEVEL="${ANDROID_API_LEVEL:-26}"
BUILD_ABIS="${RUST_ANDROID_ABIS:-arm64-v8a x86_64}"
BUILD_HOST="${RUST_BUILD_HOST:-1}"

cd "$RUST_DIR"

build_android_abi() {
    local abi="$1"
    local target clang_prefix
    case "$abi" in
        arm64-v8a)
            target="aarch64-linux-android"
            clang_prefix="aarch64-linux-android${API_LEVEL}"
            ;;
        x86_64)
            target="x86_64-linux-android"
            clang_prefix="x86_64-linux-android${API_LEVEL}"
            ;;
        *)
            echo "ERROR: Unsupported ABI $abi" >&2
            exit 1
            ;;
    esac

    local cc="$TOOLCHAIN_BIN/${clang_prefix}-clang"
    local ar="$TOOLCHAIN_BIN/llvm-ar"
    if [[ ! -x "$cc" ]]; then
        echo "ERROR: Clang compiler not found: $cc" >&2
        exit 1
    fi

    echo "=== Building Rust engine (ggufllm) for $abi ($target) ==="
    local target_upper
    target_upper="$(echo "$target" | tr '[:lower:]-' '[:upper:]_')"

    env \
        "CC_${target//-/_}=$cc" \
        "AR_${target//-/_}=$ar" \
        "CARGO_TARGET_${target_upper}_LINKER=$cc" \
        "CARGO_TARGET_${target_upper}_AR=$ar" \
        "RUSTFLAGS=-C link-arg=-Wl,-z,max-page-size=16384 -C link-arg=-llog" \
        cargo build --release --target "$target"

    mkdir -p "$JNILIBS_DIR/$abi"
    cp "$RUST_DIR/target/$target/release/libggufllm.so" "$JNILIBS_DIR/$abi/libggufllm.so"
    ls -lh "$JNILIBS_DIR/$abi/libggufllm.so"
}

for abi in $BUILD_ABIS; do
    build_android_abi "$abi"
done

if [[ "$BUILD_HOST" == "1" ]]; then
    echo "=== Building host Rust engine (ggufllm) for JVM tests ==="
    cargo build --release
    ls -lh "$RUST_DIR/target/release/libggufllm.so"
fi
