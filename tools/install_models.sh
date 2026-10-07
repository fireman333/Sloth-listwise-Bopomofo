#!/bin/bash
# Copies the R-walk model files the input method needs from a SlothLM-E ranker
# export into the model directory read by AFMNeuralSelector
# (UserPhraseLocationHelper.rWalkModelDirectory), and optionally the precise
# mode heavy model + calibration (UserPhraseLocationHelper.qwenModelDirectory).
#
# Usage: tools/install_models.sh [--dry-run] [--with-qwen]
#            [--qwen-gguf <path>] [--precise-conf <path>]
#            [target-dir] [source-dir]
#   target-dir defaults to
#     ~/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk
#   source-dir defaults to <AFM Input>/train/ranker/export/clean_v1
#
#   --with-qwen      also install the precise-mode files into the sibling
#                    directory <target-dir>/../qwen:
#                      model.gguf    <- --qwen-gguf
#                        (default <AFM Input>/models/Qwen3-0.6B-Base-Q8_0.gguf)
#                      precise.conf  <- --precise-conf
#                        (default <AFM Input>/bench/out/general/precise/precise.conf)
#                    The two files are one unit: both must exist or nothing
#                    is copied. --qwen-gguf / --precise-conf imply --with-qwen.
#
# Without --with-qwen the behavior is unchanged: only the 4 files SlothRescorer
# reads are copied (~14 MB); the fp32 model and training metadata are not.
# Restart the input method afterwards (pkill -x Sloth-listwise-Bopomofo) so it
# reloads the model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE="$SCRIPT_DIR/../../train/ranker/export/clean_v1"
DEFAULT_TARGET="$HOME/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk"

DRY_RUN=0
WITH_QWEN=0
QWEN_GGUF="$SCRIPT_DIR/../../models/Qwen3-0.6B-Base-Q8_0.gguf"
PRECISE_CONF="$SCRIPT_DIR/../../bench/out/general/precise/precise.conf"
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --with-qwen) WITH_QWEN=1; shift ;;
        --qwen-gguf)
            [[ $# -ge 2 ]] || { echo "error: --qwen-gguf needs a path" >&2; exit 1; }
            QWEN_GGUF="$2"; WITH_QWEN=1; shift 2 ;;
        --precise-conf)
            [[ $# -ge 2 ]] || { echo "error: --precise-conf needs a path" >&2; exit 1; }
            PRECISE_CONF="$2"; WITH_QWEN=1; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        --*) echo "error: unknown option $1" >&2; exit 1 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

TARGET="${POSITIONAL[0]:-$DEFAULT_TARGET}"
SOURCE="${POSITIONAL[1]:-$DEFAULT_SOURCE}"
QWEN_TARGET="$(dirname "$TARGET")/qwen"

FILES=(
    "onnx/model_quantized.onnx"
    "syl_vocab.json"
    "phonetic_table.tsv"
    "tokenizer/tokenizer.json"
)

for f in "${FILES[@]}"; do
    if [[ ! -f "$SOURCE/$f" ]]; then
        echo "error: missing $SOURCE/$f" >&2
        exit 1
    fi
done
if [[ $WITH_QWEN -eq 1 ]]; then
    for f in "$QWEN_GGUF" "$PRECISE_CONF"; do
        if [[ ! -f "$f" ]]; then
            echo "error: missing $f" >&2
            exit 1
        fi
    done
fi

echo "source: $SOURCE"
echo "target: $TARGET"
total=0
for f in "${FILES[@]}"; do
    size=$(stat -f %z "$SOURCE/$f")
    total=$((total + size))
    printf "  %-28s %10d bytes\n" "$f" "$size"
done
printf "  %-28s %10d bytes (%.1f MB)\n" "total" "$total" "$(echo "$total / 1048576" | bc -l)"

if [[ $WITH_QWEN -eq 1 ]]; then
    echo "qwen target: $QWEN_TARGET"
    gsize=$(stat -f %z "$QWEN_GGUF")
    csize=$(stat -f %z "$PRECISE_CONF")
    printf "  %-28s %10d bytes  <- %s\n" "models/qwen/model.gguf" "$gsize" "$QWEN_GGUF"
    printf "  %-28s %10d bytes  <- %s\n" "models/qwen/precise.conf" "$csize" "$PRECISE_CONF"
fi

if [[ $DRY_RUN -eq 1 ]]; then
    echo "dry run: nothing copied"
    exit 0
fi

for f in "${FILES[@]}"; do
    mkdir -p "$TARGET/$(dirname "$f")"
    cp "$SOURCE/$f" "$TARGET/$f"
done
echo "installed ${#FILES[@]} files"

if [[ $WITH_QWEN -eq 1 ]]; then
    mkdir -p "$QWEN_TARGET"
    # Conf first, GGUF last via a temp name, so a reader never sees a
    # half-copied model next to a valid conf.
    cp "$PRECISE_CONF" "$QWEN_TARGET/precise.conf"
    cp "$QWEN_GGUF" "$QWEN_TARGET/model.gguf.partial"
    mv -f "$QWEN_TARGET/model.gguf.partial" "$QWEN_TARGET/model.gguf"
    echo "installed models/qwen/model.gguf and models/qwen/precise.conf"
fi
