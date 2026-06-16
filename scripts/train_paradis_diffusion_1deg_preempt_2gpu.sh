#!/bin/bash
#SBATCH --job-name=paradis-diffusion-1deg-preempt-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=preempt
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/dmillard/def-era/logs/%x-%j.out
#SBATCH --error=/scratch/dmillard/def-era/logs/%x-%j.err

set -euo pipefail

cd /home/dmillard/Projects/def-era

export SCRATCH="${SCRATCH:-/scratch/dmillard}"
export DEF_ERA_STORAGE_ROOT="${DEF_ERA_STORAGE_ROOT:-$SCRATCH/def-era}"

RUN_NAME="${RUN_NAME:-paradis-diffusion-1deg-2015-preempt-2gpu-smoke}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-$DEF_ERA_STORAGE_ROOT/outputs/$RUN_NAME}"

DATA_ROOT="${DEF_ERA_DATA_ROOT:-$DEF_ERA_STORAGE_ROOT/ERA5}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$DATA_ROOT/1.0deg_wb2_2015}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-$DATA_ROOT/1.0deg_2015}"
DATA_YEAR="${DATA_YEAR:-2015}"

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
cpus_per_task="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((cpus_per_task / 2))}"

if [[ ! -d "$PROCESSED_DATA_DIR" ]]; then
    if [[ "${PREPARE_DATA:-0}" == "1" ]]; then
        bash scripts/prepare_paradis_1deg.sh \
            "$RAW_DATA_DIR" \
            "$PROCESSED_DATA_DIR" \
            "$DATA_YEAR" \
            "$DATA_YEAR"
    else
        echo "Processed data not found at $PROCESSED_DATA_DIR." >&2
        echo "Prepare it with:" >&2
        echo "  bash scripts/prepare_paradis_1deg.sh \"$RAW_DATA_DIR\" \"$PROCESSED_DATA_DIR\" \"$DATA_YEAR\" \"$DATA_YEAR\"" >&2
        echo "Or submit with PREPARE_DATA=1 to prepare inside this job." >&2
        exit 1
    fi
fi

num_processes="${SLURM_GPUS_ON_NODE:-2}"
echo "Starting $RUN_NAME at $(date)"
echo "Using $num_processes GPU processes"
echo "Dataset: $PROCESSED_DATA_DIR"

accelerate launch \
    --config_file _config/accelerator.yaml \
    --num_processes "$num_processes" \
    paradis_diffusion_trainer.py \
    --config-name paradis_diffusion_1deg \
    dataset.root_dir="$PROCESSED_DATA_DIR" \
    experiment.experiment_name="$RUN_NAME" \
    experiment.save_path="$RUN_OUTPUT_DIR/" \
    training_info.epochs="${EPOCHS:-1}" \
    training_info.validation_batches="${VALIDATION_BATCHES:-2}" \
    sampling.enabled=false \
    distributed_training.total_batch_size="${BATCH_SIZE_PER_GPU:-1}" \
    distributed_training.workers="${WORKERS_PER_GPU:-2}" \
    distributed_training.prefetch_factor="${PREFETCH_FACTOR:-1}" \
    distributed_training.mixed_precision="${MIXED_PRECISION:-bf16}"

echo "Finished $RUN_NAME at $(date)"
