#!/usr/bin/env python3
"""Download a time/spatial subset of WeatherBench2 ERA5 to a local zarr store."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Iterable

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar


DEFAULT_SOURCE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

REQUIRED_DATA_VARS = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "temperature",
    "land_sea_mask",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "geopotential",
    "geopotential_at_surface",
    "total_precipitation_6hr",
    "total_column_water",
    "standard_deviation_of_orography",
    "slope_of_sub_gridscale_orography",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a WeatherBench2 ERA5 subset to a local zarr store. "
            "The default source is the public 0.25-degree WB13 ERA5 zarr; "
            "the default target grid is 1 degree without poles."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Local zarr directory to write.",
    )
    parser.add_argument(
        "--source",
        default=os.environ.get("WEATHERBENCH2_ERA5_ZARR", DEFAULT_SOURCE),
        help="Source zarr URI or path.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="First timestamp to include, e.g. 2015-01-01.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Last timestamp to include, e.g. 2015-12-31T18:00:00.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Convenience option equivalent to a full calendar year.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Latitude/longitude spacing in degrees.",
    )
    parser.add_argument(
        "--include-poles",
        action="store_true",
        help="Keep +/-90 degree latitude rows. PARADIS training should leave this disabled.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace OUTPUT_DIR if it already exists.",
    )
    return parser.parse_args()


def _dates_from_args(args: argparse.Namespace) -> tuple[str, str]:
    if args.year is not None:
        if args.start_date is not None or args.end_date is not None:
            raise ValueError("Use either --year or --start-date/--end-date, not both.")
        return f"{args.year}-01-01T00:00:00", f"{args.year}-12-31T23:59:59"

    if args.start_date is None or args.end_date is None:
        raise ValueError("Provide --year or both --start-date and --end-date.")

    start_date = args.start_date
    end_date = args.end_date
    if "T" not in start_date:
        start_date = f"{start_date}T00:00:00"
    if "T" not in end_date:
        end_date = f"{end_date}T23:59:59"
    return start_date, end_date


def _grid_values(resolution: float, include_poles: bool) -> tuple[np.ndarray, np.ndarray]:
    if resolution <= 0:
        raise ValueError("--resolution must be positive.")

    if include_poles:
        latitude_start = -90.0
        latitude_stop = 90.0 + resolution / 2.0
    else:
        latitude_start = -90.0 + resolution
        latitude_stop = 90.0

    latitudes = np.arange(latitude_start, latitude_stop, resolution, dtype=np.float64)
    longitudes = np.arange(0.0, 360.0, resolution, dtype=np.float64)
    return latitudes, longitudes


def _open_zarr(source: str) -> xr.Dataset:
    storage_options = {"token": "anon"} if source.startswith("gs://") else None
    try:
        return xr.open_zarr(source, consolidated=True, storage_options=storage_options)
    except ModuleNotFoundError as exc:
        if exc.name == "gcsfs":
            raise ModuleNotFoundError(
                "gcsfs is required for gs:// WeatherBench2 downloads. "
                "Install the project requirements or `pip install gcsfs`."
            ) from exc
        raise


def _require_variables(ds: xr.Dataset, variables: Iterable[str]) -> list[str]:
    missing = [name for name in variables if name not in ds.data_vars]
    if missing:
        raise KeyError(
            "Source dataset is missing required variables: " + ", ".join(missing)
        )
    return list(variables)


def main() -> None:
    args = _parse_args()
    start_date, end_date = _dates_from_args(args)
    latitudes, longitudes = _grid_values(args.resolution, args.include_poles)

    if os.path.exists(args.output_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    print(f"Opening source zarr: {args.source}")
    ds = _open_zarr(args.source)
    required_vars = _require_variables(ds, REQUIRED_DATA_VARS)

    print(
        "Selecting "
        f"{start_date} through {end_date}, "
        f"{latitudes.size} latitudes, {longitudes.size} longitudes."
    )
    subset = ds[required_vars].sel(time=slice(start_date, end_date))
    subset = subset.sel(
        latitude=latitudes,
        longitude=longitudes,
        method="nearest",
        tolerance=max(args.resolution * 1.0e-4, 1.0e-6),
    )
    subset = subset.chunk(
        {
            "time": 1,
            "latitude": latitudes.size,
            "longitude": longitudes.size,
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_dir)), exist_ok=True)
    with ProgressBar():
        subset.to_zarr(
            args.output_dir,
            mode="w",
            consolidated=True,
            zarr_format=2,
        )
    print(f"Wrote WeatherBench2 ERA5 subset to {args.output_dir}")


if __name__ == "__main__":
    main()
