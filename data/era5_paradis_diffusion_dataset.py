"""ERA5 dataset variant for PARADIS-backed diffusion training."""

import dask
import numpy
import torch
from omegaconf import DictConfig

from data.era5_dataset import ERA5Dataset


class ERA5ParadisDiffusionDataset(ERA5Dataset):
    """Return normalized clean states for PARADIS diffusion training.

    The diffusion model should not consume WeatherBench dynamic forcings. To
    keep data loading close to upstream PARADIS, workers only load and
    normalize ERA5 state channels. Static constants are exposed once through
    ``static_constants`` and diffusion noise/timesteps are generated on-device
    inside the training step.
    """

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        timesteps: int,
        dtype=torch.float32,
        cfg: DictConfig = {},
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            start_date=start_date,
            end_date=end_date,
            forecast_steps=0,
            dtype=dtype,
            cfg=cfg,
        )
        self.timesteps = timesteps
        self.validate_nan = bool(cfg.get("dataset", {}).get("validate_nan", False))
        self._set_paradis_constants(cfg)

    def __len__(self):
        return self.length

    def _set_paradis_constants(self, cfg: DictConfig) -> None:
        """Build static channels in the same order requested by the config."""

        constants = []
        normalize_const_vars = {
            "geopotential_at_surface",
            "slope_of_sub_gridscale_orography",
            "standard_deviation_of_orography",
        }

        lat_rad = torch.from_numpy(numpy.deg2rad(self.lat)).to(self.dtype)
        lon_rad = torch.from_numpy(numpy.deg2rad(self.lon)).to(self.dtype)
        lat_grid, lon_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")

        dlon = torch.diff(lon_rad)[0]
        radius_km = 6371.0
        lon_spacing = 1.0 / (
            2
            * torch.arcsin(torch.cos(lat_grid) ** 2 * torch.sin(dlon / 2))
            * radius_km
        )
        lon_spacing = (lon_spacing - lon_spacing.mean()) / lon_spacing.std()

        for var in cfg.features.input.constants:
            if var in normalize_const_vars:
                array = (
                    torch.from_numpy(self.ds_constants[var].data).to(self.dtype)
                    - self.ds_constants[var].attrs["mean"]
                ) / self.ds_constants[var].attrs["std"]
            elif var == "land_sea_mask":
                array = torch.from_numpy(self.ds_constants[var].data).to(self.dtype)
            elif var == "lon_spacing":
                array = lon_spacing
            elif var == "cos_latitude":
                array = torch.cos(lat_grid)
            elif var == "cos_longitude":
                array = torch.cos(lon_grid)
            elif var == "sin_longitude":
                array = torch.sin(lon_grid)
            elif var == "latitude":
                array = lat_grid
            elif var == "longitude":
                array = lon_grid
            else:
                raise ValueError(f"Unsupported PARADIS constant: {var}")

            constants.append(array)

        self.constant_data = (
            torch.stack(constants)
            .permute(1, 2, 0)
            .reshape(self.lat_size, self.lon_size, -1)
            .unsqueeze(0)
        )
        self._static_constants = self.constant_data.permute(0, 3, 1, 2).squeeze(0)

    @property
    def static_constants(self) -> torch.Tensor:
        """Static PARADIS channels in ``[channels, latitude, longitude]`` order."""

        return self._static_constants

    def _normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize all state channels using dataset statistics."""

        if self.norm_precip_in.numel() > 0:
            x[..., self.norm_precip_in] = self._normalize_precipitation(
                x[..., self.norm_precip_in]
            )
        if self.norm_humidity_in.numel() > 0:
            x[..., self.norm_humidity_in] = self._normalize_humidity(
                x[..., self.norm_humidity_in]
            )
        if self.norm_zscore_in.numel() > 0:
            x[..., self.norm_zscore_in] = self._normalize_standard(
                x[..., self.norm_zscore_in],
                self.input_mean,
                self.input_std,
            )

        return x

    def _load_clean_states(self, indices: list[int]) -> torch.Tensor:
        """Load and normalize a batch of clean states with one dask graph."""

        time_indices = (
            numpy.asarray([int(index) for index in indices], dtype=numpy.int64)
            * self.interval_steps
            + self._time_index_offset
        )
        input_data = self.ds_input.isel(time=time_indices)

        (input_data,) = dask.compute(
            input_data,
            scheduler="synchronous",
            traverse=False,
        )

        if self.validate_nan and numpy.isnan(input_data.data).any():
            raise ValueError("NaN values detected in input data")

        x = torch.as_tensor(input_data.data, dtype=self.dtype)
        x = self._normalize_state(x)

        return x.permute(0, 3, 1, 2).float()

    def __getitem__(self, ind: int):
        return self._load_clean_states([int(ind)]).squeeze(0)

    def __getitems__(self, indices: list[int]):
        states = self._load_clean_states([int(index) for index in indices])
        return list(states)
