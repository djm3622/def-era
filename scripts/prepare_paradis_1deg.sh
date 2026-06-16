#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Prepare one-degree WeatherBench2/ERA5 data for PARADIS diffusion training.

Usage:
  bash scripts/prepare_paradis_1deg.sh RAW_ZARR_DIR PROCESSED_DIR [BEGIN_YEAR] [END_YEAR]

Example:
  bash scripts/prepare_paradis_1deg.sh "$SCRATCH/def-era/ERA5/1.0deg_wb2_2015" "$SCRATCH/def-era/ERA5/1.0deg_2015" 2015 2015

Steps:
  1. Download the requested WeatherBench2 ERA5 calendar years from the public
     0.25-degree WB13 zarr onto a local one-degree, no-poles raw zarr.
  2. Preprocess that local zarr into the PARADIS-compatible stacked layout.

The default prepares a one-year smoke dataset. Wider year ranges are supported
but can be large because the raw source is the 0.25-degree WeatherBench2 store.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
    usage
    exit 0
fi

raw_dir="$1"
processed_dir="$2"
begin_year="${3:-2015}"
end_year="${4:-$begin_year}"
python_bin="${PYTHON:-}"

if [[ -z "$python_bin" ]]; then
    if [[ -x "$PWD/.conda/bin/python3.12" ]]; then
        python_bin="$PWD/.conda/bin/python3.12"
    elif [[ -x "${DEF_ERA_STORAGE_ROOT:-${SCRATCH:-/scratch/dmillard}/def-era}/.conda/bin/python3.12" ]]; then
        python_bin="${DEF_ERA_STORAGE_ROOT:-${SCRATCH:-/scratch/dmillard}/def-era}/.conda/bin/python3.12"
    else
        python_bin="python3"
    fi
fi

if (( begin_year > end_year )); then
    echo "BEGIN_YEAR must be <= END_YEAR." >&2
    exit 1
fi

if [[ ! -d "$raw_dir" ]]; then
    echo "Raw one-degree zarr not found at $raw_dir. Downloading ${begin_year}-${end_year}."
    mkdir -p "$(dirname "$raw_dir")"
    "$python_bin" scripts/download_weatherbench2_era5_subset.py \
        --output-dir "$raw_dir" \
        --start-date "${begin_year}-01-01T00:00:00" \
        --end-date "${end_year}-12-31T23:59:59" \
        --resolution 1.0
fi

echo "Using raw one-degree zarr at $raw_dir."
echo "Preprocessing $raw_dir -> $processed_dir for ${begin_year}-${end_year}."

chunk_years="${PREPROCESS_CHUNK_YEARS:-1}"
if (( chunk_years < 1 )); then
    echo "PREPROCESS_CHUNK_YEARS must be >= 1." >&2
    exit 1
fi

year="$begin_year"
while (( year <= end_year )); do
    chunk_end=$((year + chunk_years - 1))
    if (( chunk_end > end_year )); then
        chunk_end="$end_year"
    fi

    echo "Writing yearly data for ${year}-${chunk_end}."
    PYTHONPATH="$PWD/paradis/data:${PYTHONPATH:-}" "$python_bin" \
        paradis/scripts/preprocess_weatherbench_data.py \
        -i "$raw_dir" \
        -o "$processed_dir" \
        --remove-poles \
        --begin_year "$year" \
        --end_year "$chunk_end" \
        --skip-static \
        --skip-stats \
        --skip-existing-years

    year=$((chunk_end + 1))
done

echo "Writing constants for ${begin_year}-${end_year}."
PYTHONPATH="$PWD/paradis/data:${PYTHONPATH:-}" "$python_bin" \
    paradis/scripts/preprocess_weatherbench_data.py \
    -i "$raw_dir" \
    -o "$processed_dir" \
    --remove-poles \
    --begin_year "$begin_year" \
    --end_year "$end_year" \
    --skip-stack \
    --skip-stats

echo "Writing streaming statistics for ${begin_year}-${end_year}."
"$python_bin" scripts/compute_paradis_streaming_stats.py \
    --dataset-dir "$processed_dir" \
    --begin-year "$begin_year" \
    --end-year "$end_year"

cat <<EOF
Done.

Use with:
  accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py --config-name paradis_diffusion_1deg dataset.root_dir=$processed_dir
EOF
