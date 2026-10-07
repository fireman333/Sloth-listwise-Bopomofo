#!/bin/bash
# Copies the R-walk model files the input method needs from a SlothLM-E ranker
# export into the model directory read by AFMNeuralSelector
# (UserPhraseLocationHelper.rWalkModelDirectory).
#
# Usage: tools/install_models.sh [--dry-run] [target-dir] [source-dir]
#   target-dir defaults to
#     ~/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk
#   source-dir defaults to <AFM Input>/train/ranker/export/clean_v1
#
# Only the 4 files SlothRescorer reads are copied (~14 MB); the fp32 model and
# training metadata are not. Restart the input method afterwards
# (pkill -x Sloth-listwise-Bopomofo) so it reloads the model.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE="$SCRIPT_DIR/../../train/ranker/export/clean_v1"
DEFAULT_TARGET="$HOME/Library/Application Support/Sloth-listwise-Bopomofo/models/r-walk"

TARGET="${1:-$DEFAULT_TARGET}"
SOURCE="${2:-$DEFAULT_SOURCE}"

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

echo "source: $SOURCE"
echo "target: $TARGET"
total=0
for f in "${FILES[@]}"; do
    size=$(stat -f %z "$SOURCE/$f")
    total=$((total + size))
    printf "  %-28s %10d bytes\n" "$f" "$size"
done
printf "  %-28s %10d bytes (%.1f MB)\n" "total" "$total" "$(echo "$total / 1048576" | bc -l)"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "dry run: nothing copied"
    exit 0
fi

for f in "${FILES[@]}"; do
    mkdir -p "$TARGET/$(dirname "$f")"
    cp "$SOURCE/$f" "$TARGET/$f"
done
echo "installed ${#FILES[@]} files"
