#!/usr/bin/env python3
"""Compute PARADIS processed-zarr statistics with bounded memory."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

import dask
import numpy as np
import xarray as xr
from numcodecs import BitRound, Blosc


REPO_ROOT = Path(__file__).resolve().parents[1]
PARADIS_DATA = REPO_ROOT / "paradis" / "data"
if str(PARADIS_DATA) not in sys.path:
    sys.path.insert(0, str(PARADIS_DATA))

from forcings.toa_radiation import toa_radiation_stats  # noqa: E402


COMPRESSOR = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute stats and tendency_stats for a PARADIS processed zarr "
            "directory by streaming over time slices."
        )
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        required=True,
        help="PARADIS processed zarr root containing per-year zarr groups.",
    )
    parser.add_argument("--begin-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--delta-hours",
        type=int,
        nargs="+",
        default=[6],
        help="Tendency horizons to compute in hours.",
    )
    return parser.parse_args()


def _contains_partial_zarr_writes(output_dir: str) -> bool:
    if not os.path.exists(output_dir):
        return False
    for _, _, filenames in os.walk(output_dir):
        if any(filename.endswith(".partial") for filename in filenames):
            return True
    return False


def _year_group_complete(output_dir: str) -> bool:
    return (
        os.path.exists(os.path.join(output_dir, ".zmetadata"))
        and os.path.exists(os.path.join(output_dir, "data", ".zarray"))
        and not _contains_partial_zarr_writes(output_dir)
    )


def _year_paths(dataset_dir: str, begin_year: int, end_year: int) -> list[str]:
    missing_years = [
        year
        for year in range(begin_year, end_year + 1)
        if not _year_group_complete(os.path.join(dataset_dir, str(year)))
    ]
    if missing_years:
        raise FileNotFoundError(
            "Missing or incomplete years: "
            + ", ".join(str(year) for year in missing_years)
        )
    return [os.path.join(dataset_dir, str(year)) for year in range(begin_year, end_year + 1)]


def _open_processed_dataset(
    dataset_dir: str,
    begin_year: int,
    end_year: int,
) -> xr.Dataset:
    return xr.open_mfdataset(
        _year_paths(dataset_dir, begin_year, end_year),
        chunks={"time": 1},
        engine="zarr",
        join="exact",
    )


def _empty_accumulators(n_features: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(n_features, dtype=np.int64),
        "sum": np.zeros(n_features, dtype=np.float64),
        "sum_sq": np.zeros(n_features, dtype=np.float64),
        "min": np.full(n_features, np.inf, dtype=np.float64),
        "max": np.full(n_features, -np.inf, dtype=np.float64),
    }


def _update_accumulators(acc: dict[str, np.ndarray], values: np.ndarray) -> None:
    """Update feature-wise statistics from ``[features, latitude, longitude]``."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [features, latitude, longitude], got {values.shape}.")

    if np.isfinite(values).all():
        axes = (1, 2)
        acc["count"] += values.shape[1] * values.shape[2]
        acc["sum"] += values.sum(axis=axes)
        acc["sum_sq"] += np.square(values).sum(axis=axes)
        acc["min"] = np.minimum(acc["min"], values.min(axis=axes))
        acc["max"] = np.maximum(acc["max"], values.max(axis=axes))
        return

    for feature_idx in range(values.shape[0]):
        feature_values = values[feature_idx]
        finite_values = feature_values[np.isfinite(feature_values)]
        if finite_values.size == 0:
            continue
        acc["count"][feature_idx] += finite_values.size
        acc["sum"][feature_idx] += finite_values.sum()
        acc["sum_sq"][feature_idx] += np.square(finite_values).sum()
        acc["min"][feature_idx] = min(acc["min"][feature_idx], finite_values.min())
        acc["max"][feature_idx] = max(acc["max"][feature_idx], finite_values.max())


def _finalize_accumulators(acc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if np.any(acc["count"] == 0):
        bad = np.where(acc["count"] == 0)[0].tolist()
        raise ValueError(f"No finite values found for feature indices: {bad}")

    mean = acc["sum"] / acc["count"]
    variance = acc["sum_sq"] / acc["count"] - np.square(mean)
    std = np.sqrt(np.maximum(variance, 0.0))
    return {
        "mean": mean.astype("float32"),
        "std": std.astype("float32"),
        "max": acc["max"].astype("float32"),
        "min": acc["min"].astype("float32"),
    }


def _write_zarr(ds: xr.Dataset, output_path: str, encoding: dict) -> None:
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    ds.to_zarr(
        output_path,
        mode="w",
        consolidated=True,
        zarr_format=2,
        encoding=encoding,
    )


def _stats_encoding(names: Iterable[str]) -> dict:
    return {
        name: {
            "compressor": COMPRESSOR,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        }
        for name in names
    }


def compute_statistics(dataset_dir: str, begin_year: int, end_year: int) -> None:
    ds = _open_processed_dataset(dataset_dir, begin_year, end_year)
    data = ds["data"].transpose("time", "features", "latitude", "longitude")
    acc = _empty_accumulators(data.sizes["features"])

    for time_idx in range(data.sizes["time"]):
        (values,) = dask.compute(
            data.isel(time=time_idx).data,
            scheduler="synchronous",
            traverse=False,
        )
        _update_accumulators(acc, values)
        if (time_idx + 1) % 100 == 0 or time_idx + 1 == data.sizes["time"]:
            print(f"Stats: processed {time_idx + 1}/{data.sizes['time']} time slices.")

    stats = _finalize_accumulators(acc)
    toa_mean, toa_std = toa_radiation_stats(
        ds.indexes["time"].values,
        ds.latitude.values,
        ds.longitude.values,
    )

    result = xr.Dataset(
        {
            "mean": xr.DataArray(stats["mean"], dims=["features"], coords={"features": ds.features}),
            "std": xr.DataArray(stats["std"], dims=["features"], coords={"features": ds.features}),
            "max": xr.DataArray(stats["max"], dims=["features"], coords={"features": ds.features}),
            "min": xr.DataArray(stats["min"], dims=["features"], coords={"features": ds.features}),
        }
    )
    result.attrs["toa_radiation_mean"] = float(toa_mean)
    result.attrs["toa_radiation_std"] = float(toa_std)

    output_path = os.path.join(dataset_dir, "stats")
    _write_zarr(result, output_path, _stats_encoding(("mean", "std", "max", "min")))
    print(f"Wrote streaming stats to {output_path}")


def _native_dt_hours(time_values: np.ndarray) -> int:
    if len(time_values) < 2:
        raise ValueError("Need at least two time steps to compute tendencies.")
    native_dt = time_values[1] - time_values[0]
    return int(native_dt.astype("timedelta64[h]").astype(int))


def compute_tendency_statistics(
    dataset_dir: str,
    begin_year: int,
    end_year: int,
    delta_hours: Iterable[int],
) -> None:
    ds = _open_processed_dataset(dataset_dir, begin_year, end_year)
    data = ds["data"].transpose("time", "features", "latitude", "longitude")
    time_values = ds.indexes["time"].values
    native_dt_hours = _native_dt_hours(time_values)

    for delta_h in delta_hours:
        if delta_h % native_dt_hours != 0:
            raise ValueError(
                f"Requested tendency delta ({delta_h}h) is not a multiple of "
                f"the native time resolution ({native_dt_hours}h)."
            )
        stride = delta_h // native_dt_hours
        acc = _empty_accumulators(data.sizes["features"])

        (previous,) = dask.compute(
            data.isel(time=0).data,
            scheduler="synchronous",
            traverse=False,
        )
        for time_idx in range(stride, data.sizes["time"]):
            (current,) = dask.compute(
                data.isel(time=time_idx).data,
                scheduler="synchronous",
                traverse=False,
            )
            _update_accumulators(acc, np.asarray(current) - np.asarray(previous))
            previous = current
            completed = time_idx - stride + 1
            total = data.sizes["time"] - stride
            if completed % 100 == 0 or completed == total:
                print(
                    f"Tendency {delta_h}h: processed "
                    f"{completed}/{total} differences."
                )

        stats = _finalize_accumulators(acc)
        result = xr.Dataset(
            {
                "tendency_mean": xr.DataArray(
                    stats["mean"], dims=["features"], coords={"features": ds.features}
                ),
                "tendency_std": xr.DataArray(
                    stats["std"], dims=["features"], coords={"features": ds.features}
                ),
                "tendency_max": xr.DataArray(
                    stats["max"], dims=["features"], coords={"features": ds.features}
                ),
                "tendency_min": xr.DataArray(
                    stats["min"], dims=["features"], coords={"features": ds.features}
                ),
            }
        )
        result.attrs["delta_hours"] = int(delta_h)
        result.attrs["native_dt_hours"] = int(native_dt_hours)
        result.attrs["stride"] = int(stride)
        result.attrs["begin_year"] = int(begin_year)
        result.attrs["end_year"] = int(end_year)
        result.attrs["n_samples"] = int(data.sizes["time"] - stride)

        output_path = os.path.join(dataset_dir, f"tendency_stats_{delta_h}h")
        _write_zarr(
            result,
            output_path,
            _stats_encoding(
                ("tendency_mean", "tendency_std", "tendency_max", "tendency_min")
            ),
        )
        print(f"Wrote streaming {delta_h}h tendency stats to {output_path}")


def main() -> None:
    args = _parse_args()
    compute_statistics(args.dataset_dir, args.begin_year, args.end_year)
    compute_tendency_statistics(
        args.dataset_dir,
        args.begin_year,
        args.end_year,
        args.delta_hours,
    )


if __name__ == "__main__":
    main()
