# PARADIS Diffusion 5-Degree Scaffold

This branch adds the first PARADIS-backed diffusion training path.

## Branches Stubbed

- `add-graphcast-submodule`
- `add-paradis-submodule`
- `graphcast-diffusion-5deg`
- `paradis-diffusion-5deg`
- `paradis-diffusion-1deg-smoke`
- `graphcast-diffusion-1deg-smoke`

## What This Branch Does

- Adds `paradis` as a git submodule.
- Adds `ERA5ParadisDiffusionDataset`, which returns:
  - normalized clean atmospheric/surface state
  - static constants
  - sampled Gaussian noise
  - sampled diffusion timestep
- Adds a PARADIS adapter that uses the upstream PARADIS model as a conditional diffusion denoiser.
- Adds `paradis_diffusion_trainer.py` and `_config/paradis_diffusion.yaml`.

The trainer targets a PARADIS-compatible processed zarr at
`$SCRATCH/def-era/ERA5/5.65deg`.

## Data

If `$SCRATCH/def-era/ERA5/5.65deg` already exists and contains the PARADIS
variable set, run the trainer against it directly.

To download raw WeatherBench2 data when needed and prepare the 5-degree data:

```bash
bash scripts/prepare_paradis_5deg.sh "$SCRATCH/def-era/ERA5/5.625deg_wb2" "$SCRATCH/def-era/ERA5/5.65deg" 2010 2011
```

That wrapper downloads the raw WeatherBench2 zarr variables required by PARADIS
if `$SCRATCH/def-era/ERA5/5.625deg_wb2` is missing, then uses the PARADIS
preprocessor to produce fields such as `vertical_velocity` and `wind_z_10m`
plus PARADIS static geometry channels.

## Smoke Training

```bash
accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py
```

Useful overrides:

```bash
accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py \
  dataset.root_dir="$SCRATCH/def-era/ERA5/5.65deg" \
  training_info.epochs=1 \
  distributed_training.total_batch_size=1 \
  distributed_training.workers=0
```

## Current Modeling Assumption

The adapter prepends zero-valued dynamic channels so PARADIS's built-in residual
forecast connection contributes zero. The PARADIS output is therefore interpreted
as the predicted diffusion noise. The conditioning input is:

- clean state, with classifier-free dropout
- noisy state at the sampled diffusion timestep
- normalized timestep channel
- static constants

Dynamic WeatherBench forcings are intentionally not passed into the diffusion
model on this branch.

## Environment

Create a project-local conda environment with Python 3.12 and upgraded pip
dependencies:

```bash
bash scripts/create_project_env.sh .conda
conda activate .conda
```

On BlueHive, `conda` may come from the `mambaforge3/22.11.1-2` module. The setup
script attempts to load that module if `conda` is not already on `PATH`.

This is a scaffold for validating the training path, not yet the final paper-scale
architecture or evaluation protocol.
