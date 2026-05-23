# PARADIS Diffusion 5-Degree Training

This branch trains a PARADIS-backed diffusion denoiser on the processed
WeatherBench2/ERA5 5-degree layout.

- Adds `paradis` as a git submodule.
- Adds `ERA5ParadisDiffusionDataset`, which returns:
  - normalized clean atmospheric/surface state
  - static constants
  - sampled Gaussian noise
  - sampled diffusion timestep
- Adds a PARADIS adapter that uses the upstream PARADIS model as a conditional diffusion denoiser.
- Adds `paradis_diffusion_trainer.py` and `_config/paradis_diffusion.yaml`.

## Entrypoints And Configs

- Data preparation: `scripts/prepare_paradis_5deg.sh`
- A100 Slurm training: `scripts/train_paradis_diffusion_5deg.sh`
- H100 Slurm training: `scripts/train_paradis_diffusion_5deg_h100.sh`
- Trainer entrypoint: `paradis_diffusion_trainer.py`
- Training config: `_config/paradis_diffusion.yaml`
- Accelerate config: `_config/accelerator.yaml`

The trainer targets a PARADIS-compatible processed zarr at:

```text
$SCRATCH/def-era/ERA5/5.65deg
```

## Data

If `$SCRATCH/def-era/ERA5/5.65deg` already exists and contains the PARADIS
variable set for all requested years, run the trainer against it directly.

To download raw WeatherBench2 data when needed and prepare the full 5-degree
data range:

```bash
bash scripts/prepare_paradis_5deg.sh "$SCRATCH/def-era/ERA5/5.625deg_wb2" "$SCRATCH/def-era/ERA5/5.65deg" 1959 2023
```

That wrapper downloads the raw WeatherBench2 zarr variables required by PARADIS
if `$SCRATCH/def-era/ERA5/5.625deg_wb2` is missing, then uses the PARADIS
preprocessor to produce fields such as `vertical_velocity` and `wind_z_10m`
plus PARADIS static geometry channels.

## Full Training

Submit one of the Slurm wrappers:

```bash
sbatch scripts/train_paradis_diffusion_5deg.sh
sbatch scripts/train_paradis_diffusion_5deg_h100.sh
```

The wrappers run Weights & Biases offline unless `WANDB_API_KEY` or
`WANDB_MODE` is already exported.

Both wrappers use:

```bash
accelerate launch --config_file _config/accelerator.yaml paradis_diffusion_trainer.py
```

The default full-run split in `_config/paradis_diffusion.yaml` is:

- training: `1960-01-01` through `2014-12-31`
- validation: `2015-01-01` through `2023-01-10`
- epochs: `300`
- validation loss batches per epoch: `64`

Useful direct launch:

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

## Training Outputs

The trainer writes under `experiment.save_path`, which defaults to:

```text
$SCRATCH/def-era/outputs/paradis-diffusion-5deg/
```

Artifacts:

- `states/checkpoint_epoch_<n>.pt`: Accelerate training state each epoch.
- `samples/epoch_<n>.pt`: saved validation examples at epoch 1, every
  `sampling.interval` epochs, and the final epoch.
- `config_mod.yaml`: resolved run config.
- `arch.txt`: model architecture dump.

Each sample file contains:

- `samples`: generated denoised fields, tensor shape `[N, C, latitude, longitude]`
- `condition`: validation condition fields used for classifier-free guidance
- `target`: validation clean fields for direct comparison
- `feature_names`: channel names aligned with `C`
- `sampling`: DDIM sampling settings used for the artifact

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
