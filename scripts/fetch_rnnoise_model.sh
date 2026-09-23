#!/usr/bin/env bash
# Fetch the RNNoise speech-denoise model used by the per-track "reduce noise"
# filter (ffmpeg's `arnndn`). The weights are third-party and not committed to
# this repo; this pulls them into assets/rnnoise/ on setup.
#
# Without the model the app still runs - the noise filter simply becomes a no-op.
set -euo pipefail

dest_dir="$(cd "$(dirname "$0")/.." && pwd)/assets/rnnoise"
mkdir -p "$dest_dir"
model="$dest_dir/bd.rnnn"
url="https://raw.githubusercontent.com/GregorR/rnnoise-models/master/beguiling-drafter-2018-08-30/bd.rnnn"

if [[ -f "$model" ]]; then
  echo "RNNoise model already present: $model"
  exit 0
fi

echo "Downloading RNNoise model -> $model"
curl -fsSL "$url" -o "$model"
echo "Done ($(wc -c < "$model") bytes)."
