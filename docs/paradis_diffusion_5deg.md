# PARADIS Diffusion 5-Degree Scaffold

This branch adds the first PARADIS-backed diffusion training path.

## Branches Stubbed

- `codex/add-graphcast-submodule`
- `codex/add-paradis-submodule`
- `codex/graphcast-diffusion-5deg`
- `codex/paradis-diffusion-5deg`
- `codex/paradis-diffusion-1deg-smoke`
- `codex/graphcast-diffusion-1deg-smoke`

## What This Branch Does

- Adds `paradis` as a git submodule.
- Adds `ERA5ParadisDiffusionDataset`, which returns:
  - clean atmospheric/surface state
  - dynamic forcings
  - static constants
  - sampled Gaussian noise
  - sampled diffusion timestep
- Adds a PARADIS adapter that uses the upstream PARADIS model as a conditional diffusion denoiser.
- Adds `paradis_diffusion_trainer.py` and `_config/paradis_diffusion.yaml`.

The trainer targets the existing repo data layout at `ERA5/5.65deg`.

## Data

No data is downloaded by this branch setup.

If `ERA5/5.65deg` already exists, run the trainer against it directly.

To prepare the 5-degree data manually:

```bash
bash scripts/prepare_paradis_5deg.sh ERA5/5.625deg_wb2 ERA5/5.65deg
```

That wrapper downloads the WeatherBench2 5.625-degree zarr only if the raw zarr
directory is missing, then preprocesses it into this repo's stacked zarr format.

## Smoke Training

```bash
accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py
```

Useful overrides:

```bash
accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py \
  dataset.root_dir=/path/to/ERA5/5.65deg \
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
- WeatherBench forcings
- static constants

This is a scaffold for validating the training path, not yet the final paper-scale
architecture or evaluation protocol.
