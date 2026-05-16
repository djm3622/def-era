"""ERA5 dataset variant for PARADIS-backed diffusion training."""

import dask
import numpy
import torch
from omegaconf import DictConfig

from data.era5_dataset import ERA5Dataset


class ERA5ParadisDiffusionDataset(ERA5Dataset):
    """Return clean states plus separated forcing/static channels.

    The existing diffusion dataset concatenates forcings and constants into one
    tensor because the current U-Net denoiser only consumes the clean/noisy state
    pair. PARADIS has an explicit static encoder, so this dataset keeps dynamic
    forcings and constants split.
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

    def __len__(self):
        return self.length

    def __getitem__(self, ind: int):
        input_data = self.ds_input.isel(time=slice(ind, ind + 1))

        with dask.config.set(scheduler="single-threaded"):
            input_data = dask.compute(input_data)[0]

        if numpy.isnan(input_data.data).any():
            raise ValueError("NaN values detected in input data")

        x = torch.tensor(input_data.data, dtype=self.dtype)
        x = self._standardize(x)

        forcings = self._compute_forcings(input_data)
        if forcings is None:
            forcings_grid = torch.empty(
                0, self.lat_size, self.lon_size, dtype=self.dtype
            )
        else:
            forcings_grid = forcings.permute(0, 3, 1, 2).squeeze(0)

        constants_grid = self.constant_data.permute(0, 3, 1, 2).squeeze(0)
        x_grid = x.permute(0, 3, 1, 2).squeeze(0)

        noisy_states = torch.randn_like(x_grid)
        rand_timesteps = torch.randint(0, self.timesteps, (1,))

        return x_grid, forcings_grid, constants_grid, noisy_states, rand_timesteps
