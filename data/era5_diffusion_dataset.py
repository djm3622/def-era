from datetime import datetime, timedelta
import os
import re

import dask
import numpy
from omegaconf import DictConfig
import torch
import xarray

from data.forcings import time_forcings, toa_radiation
from data.era5_dataset import ERA5Dataset


class ERA5DiffusionDataset(ERA5Dataset):
    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        timesteps: int,
        dtype = torch.float32,
        cfg: DictConfig = {},
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            start_date=start_date,
            end_date=end_date,
            forecast_steps=0,  # no need to load in anything extra
            dtype=dtype,
            cfg=cfg,
        )
        self.timesteps = timesteps

    def __len__(self):
        return self.length

    def __getitem__(self, ind: int):
        
        # Extract values from the requested indices
        input_data = self.ds_input.isel(time=slice(ind, ind + 1))

        # Load arrays into CPU memory
        with dask.config.set(scheduler="single-threaded"):
            input_data = dask.compute(input_data)[0]
                                      
        # # Add checks for invalid values
        if numpy.isnan(input_data.data).any():
            raise ValueError("NaN values detected in input/output data")

        # Convert to tensors - data comes in [time, lat, lon, features]
        x = torch.tensor(input_data.data, dtype=self.dtype)
        
        # Apply normalizations : removed for easier autoregression [self._apply_normalization(x, y)]
        x = self._standardize(x)
        
         # Compute forcings
        forcings = self._compute_forcings(input_data)

        # get forcings and constants for deterministic
        force_constants = torch.cat([forcings, self.constant_data], dim=-1)
        force_constants = force_constants.permute(0, 3, 1, 2).squeeze(0)

        # Permute to [time, channels, latitude, longitude] format
        x_grid = x.permute(0, 3, 1, 2).squeeze(0)
        
        # get noisy states
        noisy_states = torch.randn_like(x_grid)
        
        # Sample random timesteps
        rand_timesteps = torch.randint(0, self.timesteps, (1,))

        return x_grid, force_constants, noisy_states, rand_timesteps