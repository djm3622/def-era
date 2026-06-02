# PARADIS Diffusion Data Loading Speed

This note records the data-loading differences between upstream PARADIS time-integration training and the PARADIS diffusion training path in this repository.

## Upstream PARADIS Baseline

The fast reference path is `paradis/train.py` with `paradis/data/datamodule.py` and `paradis/data/era5_dataset.py`.

- Lightning owns device placement. Dataloader batches are moved to the selected accelerator before `training_step`.
- The dataloader uses `pin_memory=True` and `persistent_workers=True`.
- Dataset workers load ERA5 inputs/targets, forcings, and static constants, but do not generate stochastic training targets.
- Static constants are precomputed once in the dataset and reused.
- Per-step random work is minimal; the expensive model-side tensor construction happens after Lightning has moved the batch to GPU.

## Previous Diffusion Behavior

The old `ERA5ParadisDiffusionDataset.__getitem__` returned:

```python
clean_state, constants, noise, timestep
```

That made the dataloader responsible for work that is not actually data loading:

- `torch.randn_like(clean_state)` generated a full state-sized noise tensor on CPU for every sample.
- `torch.randint(...)` generated diffusion timesteps in workers.
- Static constants were included in every sample and collated/transferred every batch even though they never change.
- Every sample performed a full-array NaN scan.
- The training step then consumed those CPU-generated tensors after Accelerate moved the whole batch to GPU.

Accelerate was already moving batches to GPU after `accelerator.prepare(...)`; the bottleneck was the amount of CPU work and host-to-device payload created before that transfer.

## Current Diffusion Behavior

The PARADIS diffusion path now follows the upstream PARADIS separation more closely:

- `ERA5ParadisDiffusionDataset.__getitem__` returns only `clean_state`.
- `dataset.static_constants` exposes static channels once in `[channels, latitude, longitude]` order.
- `ParadisDiffusionDenoiser` registers static constants as a non-persistent buffer, so Accelerate moves them with the model.
- `_diffusion_step` samples Gaussian noise and diffusion timesteps on `clean_state.device`.
- NaN validation is opt-in through `dataset.validate_nan`; it is disabled by default for training throughput.
- Dask materialization uses the synchronous scheduler with `traverse=False`, matching the upstream PARADIS dataset style.
- `ERA5ParadisDiffusionDataset.__getitems__` loads a PyTorch batch with one xarray/dask graph instead of one graph per sample.

The expected speedup comes from removing CPU RNG for state-sized noise, eliminating repeated static-constant batch transfer, and shrinking each dataloader batch to the actual data read from zarr.

## ERA5/xarray Port

The base `ERA5Dataset` now also carries the upstream zarr access pattern while preserving its local return contract:

- ERA5 zarr stores are opened with `chunks={"time": 1}`, rounded float64 latitude/longitude coordinates, `join="exact"`, and the zarr engine.
- Descending latitude or longitude coordinates are sorted once at dataset construction, matching upstream PARADIS.
- Static constants are computed once, sorted to the same grid orientation, and reused by samples.
- PARADIS geometric constants (`lon_spacing`, trigonometric latitude/longitude channels, and radian latitude/longitude) are supported in the same grouped order as upstream.
- Requested input times are tracked separately from the lazy backing dataset, so targets can be read past the final requested input time without shortening the logical epoch.
- `ERA5Dataset.__getitems__` uses xarray advanced indexing to materialize a whole dataloader batch with one dask compute call. Subclasses with their own `__getitem__` keep their existing behavior unless they define a specialized `__getitems__`.

The base ERA5 class still returns `(x_grid.squeeze(0), y_grid.squeeze(0), torch.ones(1), (mu, sigma))`, and the PARADIS diffusion class still returns only the normalized clean state.
