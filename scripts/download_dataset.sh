#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Download the raw WeatherBench2 ERA5 zarr variables required by PARADIS.

Usage:
  bash scripts/download_dataset.sh OUTPUT_DIR

Environment:
  WEATHERBENCH2_ERA5_ZARR  Optional source zarr URI.

Default source:
  gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr

The downloaded raw grid is WeatherBench2 64x32, approximately 5.625 degrees.
PARADIS preprocessing later removes the poles and writes the processed 5.65deg
layout used by this repository.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -ne 1 ]]; then
    usage
    exit 0
fi

if command -v gsutil >/dev/null 2>&1; then
    gsutil_bin="$(command -v gsutil)"
elif [[ -x "$PWD/.conda/bin/gsutil" ]]; then
    gsutil_bin="$PWD/.conda/bin/gsutil"
else
    echo "gsutil is required to download WeatherBench2 data." >&2
    echo "Install project requirements or load a Google Cloud SDK environment." >&2
    exit 1
fi

base_path="${WEATHERBENCH2_ERA5_ZARR:-gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr}"
output_path="$1"

mkdir -p "${output_path}"

# Keep this list aligned with scripts/preprocess_paradis_weatherbench_data.py.
required_paths=(
    ".zattrs"
    ".zgroup"
    ".zmetadata"
    "10m_u_component_of_wind"
    "10m_v_component_of_wind"
    "2m_temperature"
    "mean_sea_level_pressure"
    "surface_pressure"
    "temperature"
    "land_sea_mask"
    "time"
    "u_component_of_wind"
    "v_component_of_wind"
    "vertical_velocity"
    "level"
    "specific_humidity"
    "geopotential"
    "latitude"
    "longitude"
    "geopotential_at_surface"
    "total_precipitation_6hr"
    "total_column_water"
    "standard_deviation_of_orography"
    "slope_of_sub_gridscale_orography"
)

sources=()
for path in "${required_paths[@]}"; do
    sources+=("${base_path}/${path}")
done

gsutil_flags=(-m)
if [[ "${GSUTIL_QUIET:-1}" == "1" ]]; then
    gsutil_flags+=(-q)
fi

"${gsutil_bin}" "${gsutil_flags[@]}" cp -r "${sources[@]}" "${output_path}"
