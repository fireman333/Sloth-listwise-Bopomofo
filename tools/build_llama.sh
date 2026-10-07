#!/bin/bash
# Builds llama.cpp + ggml as static libraries for the McBopomofo app target
# (design D5 of openspec/changes/precise-mode-and-preferences).
#
# Usage: tools/build_llama.sh [llama-src] [build-dir]
#   llama-src defaults to <AFM Input>/vendor/llama.cpp (pinned at 5e03bdd)
#   build-dir defaults to <repo>/build-llama
#
# Output: <build-dir>/src/libllama.a and <build-dir>/ggml/src/libggml*.a,
# referenced by McBopomofo.xcodeproj (LIBRARY_SEARCH_PATHS).
#
# Flag notes for vendor 5e03bdd:
#   - LLAMA_CURL is deprecated in this version (llama_option_depr, ignored),
#     so it is not passed.
#   - LLAMA_OPENSSL still exists and is turned off.
#   - GGML_CPU_ARM_ARCH becomes -march=<value>
#     (ggml/src/ggml-cpu/CMakeLists.txt:211); armv8.4-a+dotprod+fp16 runs on M1.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_SRC="${1:-$REPO_DIR/../vendor/llama.cpp}"
BUILD_DIR="${2:-$REPO_DIR/build-llama}"

if [[ ! -f "$LLAMA_SRC/CMakeLists.txt" ]]; then
    echo "error: llama.cpp source not found at $LLAMA_SRC" >&2
    exit 1
fi
LLAMA_SRC="$(cd "$LLAMA_SRC" && pwd)"

# cmake: $CMAKE, else PATH, else the pip-installed one in <AFM Input>/bench/.venv
# (the one the Rescorer builds have used).
if [[ -z "${CMAKE:-}" ]]; then
    if command -v cmake >/dev/null 2>&1; then
        CMAKE="$(command -v cmake)"
    else
        CMAKE="$REPO_DIR/../bench/.venv/lib/python3.12/site-packages/cmake/data/bin/cmake"
    fi
fi
if [[ ! -x "$CMAKE" ]]; then
    echo "error: cmake not found (set CMAKE=/path/to/cmake)" >&2
    exit 1
fi

echo "llama.cpp: $LLAMA_SRC ($(git -C "$LLAMA_SRC" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "build dir: $BUILD_DIR"

"$CMAKE" -S "$LLAMA_SRC" -B "$BUILD_DIR" \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_BLAS=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.4-a+dotprod+fp16 \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_OPENSSL=OFF

"$CMAKE" --build "$BUILD_DIR" --config Release --target llama -j "$(sysctl -n hw.ncpu)"

ARCHIVES=()
while IFS= read -r a; do ARCHIVES+=("$a"); done < <(find "$BUILD_DIR" -name '*.a' | sort)
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
    echo "error: no static libraries produced" >&2
    exit 1
fi

echo "archives:"
for a in "${ARCHIVES[@]}"; do
    printf "  %-40s %10d bytes\n" "${a#"$BUILD_DIR"/}" "$(stat -f %z "$a")"
done

# Network-symbol check (design D5 acceptance): no defined curl_/SSL_/httplib
# symbols, no undefined getaddrinfo/socket/connect references.
fail=0
# Use Apple's nm (a GNU nm earlier on PATH prints a different format).
syms=$(xcrun nm -g "${ARCHIVES[@]}" 2>/dev/null)
defined=$(echo "$syms" | awk 'NF==3 && $2!="U" {print $3}' \
    | grep -E '^_?(curl_|SSL_)|httplib' || true)
undefined=$(echo "$syms" | awk 'NF==2 && $1=="U" {print $2}' \
    | grep -E '^_?(getaddrinfo|socket|connect)$' || true)
# Positive controls so a format change cannot make the check pass vacuously.
# (grep -c, not -q: with pipefail an early grep exit would SIGPIPE awk.)
ctl_def=$(awk 'NF==3 && $2=="T" {print $3}' <<<"$syms" | grep -cx '_llama_backend_init' || true)
ctl_undef=$(awk 'NF==2 && $1=="U" {print $2}' <<<"$syms" | grep -cx '_malloc' || true)
if [[ "$ctl_def" -eq 0 || "$ctl_undef" -eq 0 ]]; then
    echo "error: nm output not parsed (positive controls missing)" >&2
    exit 1
fi
if [[ -n "$defined" ]]; then
    echo "error: network symbols defined:" >&2
    echo "$defined" | sort -u | head -20 >&2
    fail=1
fi
if [[ -n "$undefined" ]]; then
    echo "error: undefined network references:" >&2
    echo "$undefined" | sort -u >&2
    fail=1
fi
if [[ $fail -ne 0 ]]; then
    exit 1
fi
echo "nm check: no curl_/SSL_/httplib definitions, no getaddrinfo/socket/connect references"
