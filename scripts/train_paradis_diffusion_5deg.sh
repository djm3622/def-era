#!/bin/bash
#SBATCH --job-name=paradis-diffusion-5deg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/dmillard/def-era/logs/%x-%j.out
#SBATCH --error=/scratch/dmillard/def-era/logs/%x-%j.err

set -euo pipefail

cd /home/dmillard/Projects/def-era

export SCRATCH="${SCRATCH:-/scratch/dmillard}"
export DEF_ERA_STORAGE_ROOT="${DEF_ERA_STORAGE_ROOT:-$SCRATCH/def-era}"

mkdir -p "$DEF_ERA_STORAGE_ROOT/logs" "$DEF_ERA_STORAGE_ROOT/outputs/paradis-diffusion-5deg"

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

accelerate launch \
    --config_file _config/accelerator.yaml \
    --num_processes "${SLURM_GPUS_ON_NODE:-2}" \
    paradis_diffusion_trainer.py \
    dataset.root_dir="$DATA_ROOT/5.65deg" \
    experiment.experiment_name=paradis-diffusion-5deg \
    experiment.save_path="$DEF_ERA_STORAGE_ROOT/outputs/paradis-diffusion-5deg/"
