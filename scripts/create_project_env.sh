#!/usr/bin/env bash
set -euo pipefail

env_dir="${1:-.conda}"

if ! command -v conda >/dev/null 2>&1; then
    if [[ -n "${CONDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
        module load "$CONDA_MODULE"
    fi
fi

if [[ ! -d "$env_dir" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is not available. Load a conda/mambaforge module first, or set CONDA_MODULE to the site-specific module name." >&2
        exit 1
    fi

    conda create -p "$env_dir" python=3.12 pip setuptools wheel -y
else
    echo "Using existing conda environment at $env_dir"
fi

if command -v conda >/dev/null 2>&1; then
    conda install -p "$env_dir" -c conda-forge numpy=2.2.6 pandas=2.2.3 -y
fi

clean_env=(env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 PATH="$env_dir/bin:/usr/local/bin:/usr/bin:/bin")
"${clean_env[@]}" "$env_dir/bin/python" -m pip install --upgrade pip setuptools wheel
"${clean_env[@]}" "$env_dir/bin/python" -m pip install --upgrade --prefer-binary -r requirements.txt

cat <<EOF
Environment ready at $env_dir

Activate with:
  conda activate $env_dir

Or run commands with:
  $env_dir/bin/python -m pip --version
EOF
