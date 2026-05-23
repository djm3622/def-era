# ECCC Test Run

## 1. Clone

```bash
git clone --recurse-submodules git@github.com:djm3622/def-era.git
cd def-era
git submodule update --init --recursive
```

## 2. Create Environment

```bash
bash scripts/create_project_env.sh .conda
conda activate ./.conda
```

If `conda activate ./.conda` is not available, use:

```bash
export PATH="$PWD/.conda/bin:$PATH"
```

## 3. Download And Prepare Data

This requires `gsutil`. The raw WeatherBench2 download is large; the year range
below only limits preprocessing and training.

Choose a storage location with enough space:

```bash
export DEF_ERA_STORAGE_ROOT=/path/to/def-era-data
export DATA_ROOT="$DEF_ERA_STORAGE_ROOT/ERA5"
```

For a full run, use `1959 2023`.

```bash
bash scripts/prepare_paradis_5deg.sh \
  "$DATA_ROOT/5.625deg_wb2" \
  "$DATA_ROOT/5.65deg" \
  1959 2023
```

## 4. Run A Small Test

```bash
export WANDB_MODE=offline

accelerate launch \
  --config_file _config/accelerator.yaml \
  --num_processes 1 \
  paradis_diffusion_trainer.py \
  dataset.root_dir="$DATA_ROOT/5.65deg" \
  experiment.save_path="$DEF_ERA_STORAGE_ROOT/outputs/test-run/" \
  experiment.experiment_name=test-run \
  training.dataset.start_date=2014-01-01 \
  training.dataset.end_date=2014-01-07 \
  training.validation_dataset.start_date=2015-01-01 \
  training.validation_dataset.end_date=2015-01-07 \
  training_info.epochs=1 \
  training_info.validation_batches=1 \
  distributed_training.total_batch_size=1 \
  distributed_training.workers=0 \
  sampling.enabled=false
```

Outputs will be written to:

```text
$DEF_ERA_STORAGE_ROOT/outputs/test-run/
```
