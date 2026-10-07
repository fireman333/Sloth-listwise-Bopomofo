#!/bin/sh
# Xcode glue for AFMPersonalTrainer (personal-finetune task 3.1).
#
#   xcode_build_helper.sh build   legacy target "AFMPersonalTrainer": CMake (Release, no tests)
#                                 into $OBJROOT/AFMPersonalTrainer-cmake, then strip (Release
#                                 configuration) + sign into .../signed/
#   xcode_build_helper.sh embed   McBopomofo "Embed AFMPersonalTrainer" phase: copy the signed
#                                 helper into Contents/MacOS/
#   xcode_build_helper.sh clean
#
# The helper is always built optimized (MLX training is unusably slow at -O0).
# cmake / ninja: $AFM_CMAKE or cmake on PATH, else <AFM Input>/bench/.venv/bin.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
SRCROOT_="${SRCROOT:-$(cd "$HERE/../.." && pwd)}"
OUT="${OBJROOT:-$SRCROOT_/build}/AFMPersonalTrainer-cmake"
VENV="$SRCROOT_/../bench/.venv/bin"
CMAKE="${AFM_CMAKE:-$(command -v cmake || echo "$VENV/cmake")}"
NINJA="$(command -v ninja || echo "$VENV/ninja")"

case "${1:-${ACTION:-build}}" in
  build|install|"")
    if [ ! -f "$OUT/build.ninja" ]; then
      env -u SDKROOT -u MACOSX_DEPLOYMENT_TARGET "$CMAKE" -S "$HERE" -B "$OUT" -G Ninja \
        -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release -DENABLE_TEST=OFF \
        -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
        -DCMAKE_OBJCXX_COMPILER=/usr/bin/clang++ -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 \
        -DCMAKE_SKIP_BUILD_RPATH=ON
    fi
    env -u SDKROOT "$CMAKE" --build "$OUT" --target AFMPersonalTrainer
    # Strip (Release) and sign here: the McBopomofo script phase runs sandboxed
    # (ENABLE_USER_SCRIPT_SANDBOXING) and codesign / strip need temp files next
    # to their output. Ad-hoc now; a Developer ID build re-signs the helper with
    # the app identity (hardened runtime, no entitlements; design D5).
    SIGNED="$OUT/signed/AFMPersonalTrainer"
    mkdir -p "$OUT/signed"
    rm -f "$SIGNED"
    if [ "${CONFIGURATION:-Debug}" = "Release" ]; then
      strip -x -o "$SIGNED" "$OUT/AFMPersonalTrainer"
    else
      cp -f "$OUT/AFMPersonalTrainer" "$SIGNED"
    fi
    /usr/bin/codesign --force --sign "${AFM_TRAINER_SIGN_IDENTITY:--}" --timestamp=none \
      --identifier "io.github.fireman333.inputmethod.Sloth-listwise-Bopomofo.trainer" "$SIGNED"
    ;;
  embed)
    # McBopomofo phase "Embed AFMPersonalTrainer": copy only (declared input / output).
    SIGNED="$OUT/signed/AFMPersonalTrainer"
    [ -x "$SIGNED" ] || { echo "error: $SIGNED missing (build target AFMPersonalTrainer)"; exit 1; }
    cp -f "$SIGNED" "$TARGET_BUILD_DIR/$EXECUTABLE_FOLDER_PATH/AFMPersonalTrainer"
    ;;
  clean)
    rm -rf "$OUT"
    ;;
  *)
    echo "usage: $0 build|embed|clean"; exit 2
    ;;
esac
