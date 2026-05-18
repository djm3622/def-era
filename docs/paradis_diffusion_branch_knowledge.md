# PARADIS Diffusion Branch Knowledge Base

This note documents how the current `paradis-diffusion-5deg` branch is wired. It is intended as an operational map for continuing development on a new server, not as a final method description for a paper.

## Branch Purpose

The branch adds a PARADIS-backed conditional diffusion training path for WeatherBench/ERA5 data at the PARADIS 5-degree processed resolution. The top-level diffusion code in this repository remains in place, while this branch adds a separate PARADIS adapter, dataset variant, trainer, config, and data-preparation wrapper.

The main training entrypoint is:

```bash
accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py
```

The Hydra config for this path is `_config/paradis_diffusion.yaml`.

## Repository Areas

- `paradis/`: checked-out PARADIS submodule. The branch uses its model implementation and preprocessing scripts.
- `data/era5_paradis_diffusion_dataset.py`: dataset adapter that returns clean normalized states, static constants, sampled diffusion noise, and sampled timesteps.
- `model/paradis_diffusion/model.py`: wrapper that adapts upstream PARADIS into a diffusion noise predictor.
- `model/paradis_diffusion/train.py`: diffusion training loop for the PARADIS wrapper.
- `model/paradis_diffusion/external.py`: import shim that loads the PARADIS submodule despite both repositories using a top-level `model` package name.
- `paradis_diffusion_trainer.py`: Hydra/Accelerate entrypoint that constructs datasets, dataloaders, model, optimizer, scheduler, loss, and training loop.
- `scripts/prepare_paradis_5deg.sh`: manual preprocessing wrapper for raw WeatherBench2 zarr data.
- `docs/paradis_diffusion_5deg.md`: earlier scaffold note for this branch.

## Data Contract

The trainer expects `dataset.root_dir` to point at a PARADIS-compatible processed zarr directory. The default is:

```yaml
dataset:
  root_dir: "${oc.env:SCRATCH,/scratch/dmillard}/def-era/ERA5/5.65deg"
```

The processed directory must contain the data, stats, and constants layout expected by `data/era5_dataset.py`, including:

- data zarr shards matched by `xarray.open_mfdataset(os.path.join(root_dir, "*"), engine="zarr")`
- `stats` zarr with feature-level `mean`, `std`, `min`, `max`
- `constants` zarr with PARADIS static fields

The PARADIS diffusion config uses these dynamic state variables:

- Atmospheric variables at all configured pressure levels: `geopotential`, `wind_x`, `wind_y`, `wind_z`, `specific_humidity`, `temperature`, `vertical_velocity`
- Surface variables: `wind_x_10m`, `wind_y_10m`, `wind_z_10m`, `2m_temperature`, `mean_sea_level_pressure`, `surface_pressure`, `total_column_water`

The config intentionally sets dynamic forcings to an empty list:

```yaml
features:
  input:
    forcings: []
```

Static constants passed to PARADIS are:

- `geopotential_at_surface`
- `land_sea_mask`
- `slope_of_sub_gridscale_orography`
- `standard_deviation_of_orography`
- `lon_spacing`
- `cos_latitude`
- `cos_longitude`
- `sin_longitude`
- `latitude`
- `longitude`

## Data Preparation

The preparation wrapper downloads the raw PARADIS WeatherBench2 inputs when the
raw directory is missing, then preprocesses them on scratch:

```bash
bash scripts/prepare_paradis_5deg.sh "$SCRATCH/def-era/ERA5/5.625deg_wb2" "$SCRATCH/def-era/ERA5/5.65deg" 2010 2011
```

That script:

1. Checks whether the raw zarr directory exists.
2. Calls `scripts/download_dataset.sh` if the raw zarr directory is missing.
3. Runs `paradis/scripts/preprocess_weatherbench_data.py`.
4. Uses `--remove-poles`.
5. Writes a PARADIS-compatible processed zarr directory.

The script sets `PYTHONPATH="$PWD/paradis/data:${PYTHONPATH:-}"` so the PARADIS preprocessor can resolve its local data imports.

## Dataset Behavior

`ERA5ParadisDiffusionDataset` subclasses `ERA5Dataset` with `forecast_steps=0`. Each item returns:

```python
clean_state, constants, noise, timestep
```

with tensor layout:

- `clean_state`: `[state_channels, latitude, longitude]`
- `constants`: `[static_channels, latitude, longitude]`
- `noise`: same shape as `clean_state`
- `timestep`: shape `[1]`, sampled uniformly from `[0, cfg.dataset.timestep)`

The clean state is selected from one time index, normalized, and permuted from xarray's `[time, lat, lon, features]` layout into PyTorch channel-first grid layout.

Normalization is inherited from the base ERA5 dataset:

- precipitation channels use the repository's logarithmic precipitation transform
- specific humidity uses the repository's logarithmic humidity transform
- remaining variables use feature-level z-score statistics from the processed `stats` zarr

The PARADIS diffusion dataset rebuilds static constants in the exact order requested by `cfg.features.input.constants`. It normalizes or derives them as follows:

- z-score normalization for `geopotential_at_surface`, `slope_of_sub_gridscale_orography`, and `standard_deviation_of_orography`
- raw values for `land_sea_mask`
- derived spherical geometry channels for longitude spacing, trigonometric latitude/longitude, latitude, and longitude

## Model Adapter

`ParadisDiffusionDenoiser` wraps the upstream `Paradis` class from the `paradis/` submodule.

The wrapper presents PARADIS with a synthetic dynamic input made by concatenating:

```text
zero_residual | condition | noisy_state | normalized_timestep | constants
```

where:

- `zero_residual` has the same shape as the clean/noisy state
- `condition` is the clean state, optionally dropped to zeros for classifier-free training
- `noisy_state` is the forward-diffused state at the sampled timestep
- `normalized_timestep` is a scalar timestep channel expanded over the grid
- `constants` are the PARADIS static channels

The upstream PARADIS model returns:

```python
residual_base + output_projection(hidden)
```

where `residual_base` is sliced from the first `num_common_features` input channels. This branch intentionally prepends zeros in that position, so the PARADIS residual forecast connection contributes zero. The wrapper therefore interprets the PARADIS output directly as predicted diffusion noise.

The adapter constructs a minimal PARADIS datamodule/config shape:

- `num_in_dyn_features = state_channels * 3 + 1`
- `num_in_static_features = static_channels`
- `num_common_features = state_channels`
- `num_out_features = state_channels`
- `n_time_inputs = 1`

## Import Shim

The main repository and the PARADIS submodule both use a top-level package named `model`. Directly importing PARADIS would collide with this repository's `model` package.

`model/paradis_diffusion/external.py` avoids that collision by temporarily installing a shim `model` module whose `__path__` points at `paradis/model`, importing `model.paradis`, returning the `Paradis` class, and then restoring the original `sys.modules` entries.

If the submodule is missing, it raises an error asking for:

```bash
git submodule update --init paradis
```

## Training Loop

`paradis_diffusion_trainer.py` performs the high-level orchestration:

1. Loads `_config/paradis_diffusion.yaml` with Hydra.
2. Creates an `Accelerator` with configured gradient accumulation and mixed precision.
3. Seeds random number generators through `utils.utility.set_random_seeds`.
4. Creates train and validation `ERA5ParadisDiffusionDataset` instances.
5. Inspects the first sample to infer state and static channel counts.
6. Builds the PARADIS diffusion denoiser.
7. Optionally loads model weights from `experiment.from_checkpoint`.
8. Creates AdamW, dataloaders, OneCycleLR, and MSE diffusion loss.
9. Optionally resumes full training state from `experiment.from_state`.
10. Calls `model.paradis_diffusion.train.training_loop`.

For each batch, `_diffusion_step` does standard DDPM-style noise prediction training:

```python
noisy_state = sqrt(alpha_bar_t) * clean_state + sqrt(1 - alpha_bar_t) * noise
noise_pred = model(condition, noisy_state, timestep, constants)
loss = MSE(noise_pred, noise)
```

The noise schedule is a linear beta schedule from `dataset.beta_start` to `dataset.beta_end`, with cumulative products used as `alpha_bar`.

Classifier-free conditioning dropout is implemented by replacing the clean condition with zeros for a random subset of batch elements. Validation forces dropout to `0.0`.

After each epoch, the loop writes Accelerate training state under:

```text
<experiment.save_path>/states
```

The default save path is:

```yaml
experiment:
  save_path: "${oc.env:SCRATCH,/scratch/dmillard}/def-era/outputs/paradis-diffusion-5deg-smoke/"
```

## Default Smoke Configuration

The current config is sized as a smoke-training path rather than a paper-scale run:

- 5 epochs
- train window: `2010-01-01` to `2010-01-31`
- validation window: `2011-01-01` to `2011-01-07`
- total batch size 2
- validation limited to 2 batches
- `mixed_precision: "fp16"`
- PARADIS latent size 128
- PARADIS `num_layers: 2`
- `condition_dropout: 0.1`
- diffusion timesteps: 1000

Useful smoke override:

```bash
accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py \
  dataset.root_dir="$SCRATCH/def-era/ERA5/5.65deg" \
  training_info.epochs=1 \
  distributed_training.total_batch_size=1 \
  distributed_training.workers=0
```

## Environment Notes

The repo includes a project-local environment helper:

```bash
bash scripts/create_project_env.sh .conda
conda activate .conda
```

On BlueHive-style systems, the helper attempts to load `mambaforge3/22.11.1-2` if `conda` is not already on `PATH`.

The requirements pin `numpy==2.2.6`, `pandas==2.2.3`, `wandb==0.18.7`, and use `zarr<3.0`. They also include `layerquantizer` from GitHub for PARADIS-related compression extras.

## Current Assumptions and Limitations

- This is a scaffold for validating the PARADIS diffusion path.
- There is no sampling/evaluation script yet for the PARADIS diffusion model.
- Dynamic WeatherBench forcings are not passed into the diffusion model.
- The PARADIS residual forecast pathway is disabled by construction through zero-valued residual input channels.
- The code assumes the PARADIS submodule is present at `paradis/`.
- The config and docs target 5-degree processed data at `$SCRATCH/def-era/ERA5/5.65deg`.
- Validation reports a limited MSE noise-prediction loss, not forecast skill.
- The training loop saves state every epoch but does not currently implement early stopping despite `training_info.patience` being present in the config.
