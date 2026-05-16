#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Prepare 5-degree WeatherBench/ERA5 data for the PARADIS diffusion branch.

This script is provided for manual use. It is not run automatically.

Usage:
  bash scripts/prepare_paradis_5deg.sh RAW_ZARR_DIR PROCESSED_DIR

Example:
  bash scripts/prepare_paradis_5deg.sh ERA5/5.625deg_wb2 ERA5/5.65deg

Steps:
  1. If RAW_ZARR_DIR does not exist, download the 5.625-degree WeatherBench2 zarr.
  2. Preprocess it into this repo's stacked 5.65-degree zarr layout.

The PARADIS diffusion trainer in this branch expects PROCESSED_DIR to match
dataset.root_dir in _config/paradis_diffusion.yaml.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
    usage
    exit 0
fi

raw_dir="$1"
processed_dir="$2"

if [[ ! -d "$raw_dir" ]]; then
    echo "Raw zarr not found at $raw_dir; downloading WeatherBench2 5-degree data."
    bash scripts/download_dataset.sh "$raw_dir"
else
    echo "Using existing raw zarr at $raw_dir."
fi

echo "Preprocessing $raw_dir -> $processed_dir."
python scripts/preprocess_weatherbench_data.py \
    -i "$raw_dir" \
    -o "$processed_dir" \
    --remove-poles

cat <<EOF
Done.

Use with:
  accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py dataset.root_dir=$processed_dir
EOF
