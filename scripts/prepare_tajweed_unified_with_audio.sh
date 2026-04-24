#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash scripts/prepare_tajweed_unified_with_audio.sh /path/to/Quran-Data/public/audio [verses_per_surah]"
  exit 1
fi

AUDIO_ROOT="$1"
VERSES_PER_SURAH="${2:-3}"

python scripts/build_tajweed_unified_metadata.py \
  --output-dir data/raw/tajweed_unified \
  --verses-per-surah "$VERSES_PER_SURAH" \
  --train-ratio 0.9 \
  --seed 42 \
  --audio-root "$AUDIO_ROOT" \
  --audio-ext m4a \
  --save-full

echo "Prepared:"
echo "  data/raw/tajweed_unified/train_audio.csv"
echo "  data/raw/tajweed_unified/dev_audio.csv"
