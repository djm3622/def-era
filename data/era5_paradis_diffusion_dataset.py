"""ERA5 dataset variant for PARADIS-backed diffusion training."""

import dask
import numpy
import torch
from omegaconf import DictConfig

from data.era5_dataset import ERA5Dataset


class ERA5ParadisDiffusionDataset(ERA5Dataset):
    """Return normalized clean states plus PARADIS static channels.

    The diffusion model should not consume WeatherBench dynamic forcings. This
    dataset therefore returns only normalized state channels, static constants,
    sampled Gaussian noise, and sampled diffusion timesteps.
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

    def __getitem__(self, ind: int):
        input_data = self.ds_input.isel(time=slice(ind, ind + 1))

        with dask.config.set(scheduler="single-threaded"):
            input_data = dask.compute(input_data)[0]

        if numpy.isnan(input_data.data).any():
            raise ValueError("NaN values detected in input data")

        x = torch.as_tensor(input_data.data, dtype=self.dtype)
        x = self._normalize_state(x)
        constants_grid = self.constant_data.permute(0, 3, 1, 2).squeeze(0)
        x_grid = x.permute(0, 3, 1, 2).squeeze(0)

        noisy_states = torch.randn_like(x_grid)
        rand_timesteps = torch.randint(0, self.timesteps, (1,))

        return x_grid, constants_grid, noisy_states, rand_timesteps
