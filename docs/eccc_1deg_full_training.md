# ECCC 1-Degree PARADIS Diffusion Full Training

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

The 1-degree downloader requires `gcsfs`, which is included in
`requirements.txt`.

## 3. Choose Storage

Use a high-capacity filesystem. The 2015 1-degree test produced about `26G` of
raw zarr and `17G` of processed zarr, so the full 1959-2023 range can require
terabytes.

```bash
export DEF_ERA_STORAGE_ROOT=/path/to/def-era-data
export DATA_ROOT="$DEF_ERA_STORAGE_ROOT/ERA5"
```

## 4. Prepare Full 1-Degree Data

The 1-degree path downloads a one-degree subset from the public 0.25-degree
WeatherBench2 WB13 source, removes poles, preprocesses with the PARADIS
submodule, and computes streaming stats to avoid the high-memory reduction used
by the upstream preprocessor.

```bash
bash scripts/prepare_paradis_1deg.sh \
  "$DATA_ROOT/1.0deg_wb2_1959_2023" \
  "$DATA_ROOT/1.0deg" \
  1959 2023
```

Expected processed layout:

```text
$DATA_ROOT/1.0deg/
  1959/
  ...
  2023/
  constants/
  stats/
  tendency_stats_6h/
```

For a one-year smoke dataset, use:

```bash
bash scripts/prepare_paradis_1deg.sh \
  "$DATA_ROOT/1.0deg_wb2_2015" \
  "$DATA_ROOT/1.0deg_2015" \
  2015 2015
```

## 5. Run The Preempt Benchmark

After the 2015 smoke data exists, submit the 1-epoch preempt benchmark:

```bash
sbatch scripts/train_paradis_diffusion_1deg_preempt_bench_1ep.sh
```

That job uses:

- `--partition=preempt`
- `--gres=gpu:2`
- one epoch
- validation and sampling disabled
- per-process dataloader batch size `8`
- processed data at `$DATA_ROOT/1.0deg_2015` unless `PROCESSED_DATA_DIR` is set

## 6. Full 20-GPU Training

The `_config/paradis_diffusion_1deg.yaml` file is a 2015 smoke config. For full
training, override the data root, dates, epoch count, validation batches, and
process count explicitly.

```bash
export WANDB_PROJECT=DEF
export RUN_NAME=paradis-diffusion-1deg-full
export RUN_OUTPUT_DIR="$DEF_ERA_STORAGE_ROOT/outputs/$RUN_NAME"
mkdir -p "$RUN_OUTPUT_DIR"

accelerate launch \
  --config_file _config/accelerator.yaml \
  --num_processes 20 \
  paradis_diffusion_trainer.py \
  --config-name paradis_diffusion_1deg \
  dataset.root_dir="$DATA_ROOT/1.0deg" \
  experiment.experiment_name="$RUN_NAME" \
  experiment.save_path="$RUN_OUTPUT_DIR/" \
  training.dataset.start_date=1960-01-01 \
  training.dataset.end_date=2014-12-31 \
  training.validation_dataset.start_date=2015-01-01 \
  training.validation_dataset.end_date=2023-01-10 \
  training_info.epochs=300 \
  training_info.validation_batches=64 \
  sampling.enabled=true \
  distributed_training.total_batch_size=8 \
  distributed_training.workers=2 \
  distributed_training.prefetch_factor=1 \
  distributed_training.mixed_precision=bf16
```

`distributed_training.total_batch_size` is the dataloader batch size passed to
each process in the current trainer. On 20 GPUs with the command above, the
effective global batch is `20 * 8` before gradient accumulation.

## 7. Outputs

Outputs are written under:

```text
$DEF_ERA_STORAGE_ROOT/outputs/paradis-diffusion-1deg-full/
```

Important artifacts:

- `states/checkpoint_epoch_<n>.pt`: Accelerate training state.
- `samples/epoch_<n>.pt`: DDIM validation samples when `sampling.enabled=true`.
- `config_mod.yaml`: resolved Hydra config.
- `arch.txt`: model architecture.
