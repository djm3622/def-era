"""PARADIS architecture adapter for conditional diffusion denoising."""

from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from .external import load_paradis_class


def _build_paradis_cfg(cfg: DictConfig) -> DictConfig:
    """Build the minimal config shape expected by upstream PARADIS."""

    return OmegaConf.create(
        {
            "model": OmegaConf.to_container(cfg.paradis.model, resolve=True),
            "compute": OmegaConf.to_container(cfg.paradis.compute, resolve=True),
            "features": OmegaConf.to_container(cfg.features, resolve=True),
            "dataset": {"n_time_inputs": 1},
        }
    )


def _latlon_grids(lat: np.ndarray, lon: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    lat_rad = torch.from_numpy(np.deg2rad(lat)).float()
    lon_rad = torch.from_numpy(np.deg2rad(lon)).float()
    return torch.meshgrid(lat_rad, lon_rad, indexing="ij")


class ParadisDiffusionDenoiser(nn.Module):
    """Use PARADIS as a diffusion noise predictor.

    PARADIS forecasts by adding a residual from a slice of its input channels.
    For denoising, the wrapper prepends a zero residual base, so the upstream
    residual connection contributes zero and the model output is the predicted
    diffusion noise.
    """

    def __init__(
        self,
        state_channels: int,
        static_channels: int,
        lat: np.ndarray,
        lon: np.ndarray,
        cfg: DictConfig,
        static_constants: Optional[torch.Tensor] = None,
        paradis_root: str = "paradis",
    ) -> None:
        super().__init__()
        self.state_channels = state_channels
        self.static_channels = static_channels
        self.timesteps = int(cfg.dataset.timestep)

        lat_grid, lon_grid = _latlon_grids(lat, lon)
        paradis_cfg = _build_paradis_cfg(cfg)
        Paradis = load_paradis_class(paradis_root)

        dyn_channels = state_channels * 3 + 1
        dataset_spec = SimpleNamespace(
            num_in_dyn_features=dyn_channels,
            num_in_static_features=static_channels,
            n_time_inputs=1,
        )
        datamodule_spec = SimpleNamespace(
            dataset=dataset_spec,
            num_common_features=state_channels,
            num_out_features=state_channels,
        )

        self.paradis = Paradis(datamodule_spec, paradis_cfg, lat_grid, lon_grid)
        if static_constants is not None:
            if static_constants.ndim != 3:
                raise ValueError(
                    "static_constants must have shape "
                    "[channels, latitude, longitude]."
                )
            if static_constants.shape[0] != static_channels:
                raise ValueError(
                    "static_constants channel count does not match "
                    f"static_channels "
                    f"({static_constants.shape[0]} != {static_channels})."
                )
            self.register_buffer(
                "static_constants",
                static_constants.detach().clone().float().unsqueeze(0),
                persistent=False,
            )
        else:
            self.register_buffer("static_constants", None, persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _time_channel(
        self, timesteps: torch.Tensor, shape: Tuple[int, int, int, int]
    ) -> torch.Tensor:
        batch, _, height, width = shape
        denom = max(self.timesteps - 1, 1)
        t = timesteps.float().view(batch, 1, 1, 1) / denom
        return t.expand(batch, 1, height, width)

    def forward(
        self,
        condition: torch.Tensor,
        noisy_state: torch.Tensor,
        timesteps: torch.Tensor,
        constants: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        if constants is None:
            if self.static_constants is not None:
                constants = self.static_constants.to(
                    device=condition.device,
                    dtype=condition.dtype,
                ).expand(
                    condition.shape[0],
                    -1,
                    -1,
                    -1,
                )
            else:
                constants = condition.new_empty(
                    condition.shape[0], 0, condition.shape[-2], condition.shape[-1]
                )
        else:
            constants = constants.to(device=condition.device, dtype=condition.dtype)

        zero_residual = torch.zeros_like(noisy_state)
        timestep_channel = self._time_channel(timesteps, noisy_state.shape)
        fields = torch.cat(
            [
                zero_residual,
                condition,
                noisy_state,
                timestep_channel,
                constants,
            ],
            dim=1,
        )

        sample = self.paradis(fields)
        if return_dict:
            return {"sample": sample}
        return (sample,)


def get_paradis_diffusion_model(
    state_channels: int,
    static_channels: int,
    lat: np.ndarray,
    lon: np.ndarray,
    cfg: DictConfig,
    static_constants: Optional[torch.Tensor] = None,
) -> ParadisDiffusionDenoiser:
    return ParadisDiffusionDenoiser(
        state_channels=state_channels,
        static_channels=static_channels,
        lat=lat,
        lon=lon,
        cfg=cfg,
        static_constants=static_constants,
    )
