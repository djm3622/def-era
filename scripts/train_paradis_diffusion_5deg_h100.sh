#!/bin/bash
#SBATCH --job-name=paradis-diffusion-5deg-preempt-bench-1ep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=preempt
#SBATCH --nodelist=bhg0075,bhg0078,bhgrb4x0081,bhgrb4x0082,bhgrb4x0083,bhgrb4x0084,bhgrb4x0085,bhgrb8x0080
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/dmillard/def-era/logs/%x-%j.out
#SBATCH --error=/scratch/dmillard/def-era/logs/%x-%j.err

set -euo pipefail

cd /home/dmillard/Projects/def-era

export SCRATCH="${SCRATCH:-/scratch/dmillard}"
export DEF_ERA_STORAGE_ROOT="${DEF_ERA_STORAGE_ROOT:-$SCRATCH/def-era}"

RUN_NAME="paradis-diffusion-5deg-l40s-preempt-benchmark-1epoch"
RUN_OUTPUT_DIR="$DEF_ERA_STORAGE_ROOT/outputs/$RUN_NAME"

mkdir -p "$DEF_ERA_STORAGE_ROOT/logs" "$RUN_OUTPUT_DIR"

export WANDB_PROJECT="${WANDB_PROJECT:-DEF}"
if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then
    export WANDB_MODE=offline
fi

env_dir=""
if [[ -d "$PWD/.conda" ]]; then
    env_dir="$PWD/.conda"
elif [[ -d "$DEF_ERA_STORAGE_ROOT/.conda" ]]; then
    env_dir="$DEF_ERA_STORAGE_ROOT/.conda"
fi

if [[ -n "$env_dir" ]]; then
    if command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$env_dir"
    else
        export PATH="$env_dir/bin:$PATH"
    fi
fi

if ! command -v accelerate >/dev/null 2>&1; then
    echo "accelerate is not available. Create or activate the project .conda environment before submitting." >&2
    exit 1
fi

export PYTHONPATH="$PWD/paradis/data:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

DATA_ROOT="${DEF_ERA_DATA_ROOT:-$DEF_ERA_STORAGE_ROOT/ERA5}"

echo "Starting $RUN_NAME at $(date)"
echo "Training split: 1960-01-01 to 2014-12-31"
echo "Validation and sample generation are disabled for epoch-throughput benchmarking."

accelerate launch \
    --config_file _config/accelerator.yaml \
    --num_processes "${SLURM_GPUS_ON_NODE:-1}" \
    paradis_diffusion_trainer.py \
    dataset.root_dir="$DATA_ROOT/5.65deg" \
    experiment.experiment_name="$RUN_NAME" \
    experiment.save_path="$RUN_OUTPUT_DIR/" \
    training_info.epochs=1 \
    training_info.validation_batches=0 \
    sampling.enabled=false \
    distributed_training.prefetch_factor=1 \
    distributed_training.mixed_precision=bf16

echo "Finished $RUN_NAME at $(date)"
