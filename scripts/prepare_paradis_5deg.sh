#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Prepare 5-degree WeatherBench/ERA5 data for the PARADIS diffusion branch.

Usage:
  bash scripts/prepare_paradis_5deg.sh RAW_ZARR_DIR PROCESSED_DIR [BEGIN_YEAR] [END_YEAR]

Example:
  bash scripts/prepare_paradis_5deg.sh "$SCRATCH/def-era/ERA5/5.625deg_wb2" "$SCRATCH/def-era/ERA5/5.65deg" 2010 2011

Steps:
  1. Download the raw WeatherBench2 zarr if RAW_ZARR_DIR does not exist.
  2. Preprocess it into a PARADIS-compatible stacked 5.65-degree zarr layout.

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
begin_year="${3:-2010}"
end_year="${4:-2011}"
python_bin="${PYTHON:-}"

if [[ -z "$python_bin" ]]; then
    if [[ -x "$PWD/.conda/bin/python3.12" ]]; then
        python_bin="$PWD/.conda/bin/python3.12"
    else
        python_bin="python3"
    fi
fi

if [[ ! -d "$raw_dir" ]]; then
    echo "Raw zarr not found at $raw_dir. Downloading PARADIS WeatherBench2 inputs."
    mkdir -p "$(dirname "$raw_dir")"
    bash scripts/download_dataset.sh "$raw_dir"
fi

echo "Using existing raw zarr at $raw_dir."
echo "Preprocessing $raw_dir -> $processed_dir."
PYTHONPATH="$PWD/paradis/data:${PYTHONPATH:-}" "$python_bin" \
    paradis/scripts/preprocess_weatherbench_data.py \
    -i "$raw_dir" \
    -o "$processed_dir" \
    --remove-poles \
    --begin_year "$begin_year" \
    --end_year "$end_year"

cat <<EOF
Done.

Use with:
  accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py dataset.root_dir=$processed_dir
EOF
