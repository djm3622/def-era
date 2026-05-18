#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STORAGE_ROOT="${DEF_ERA_STORAGE_ROOT:-${SCRATCH:-/scratch/dmillard}/def-era}"

CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/_config/diffusion.yaml}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_ROOT/diffusion_eval.py}"

# Define parameter variations
SAMPLERS=("DPM++" "DDIM")
WALKS=(1 4 8)
GUIDANCE_SCALES=(0.3 0.5 0.7 1.0)
ETAS=(0.01 0.1)
SOLVER_ORDERS=(1 2 3)
ALGORITHM_TYPES=("dpmsolver++")
NUM_STEPS=(10 25 50 100)

# Root directory for outputs
ROOT_DIR="${ROOT_DIR:-$STORAGE_ROOT/outputs/run02}"
TEMP_CONFIG="$STORAGE_ROOT/tmp/config_temp.yaml"
mkdir -p "$(dirname "$TEMP_CONFIG")"

# Function to update yaml value - always with evaluation indentation
update_yaml() {
    local key=$1
    local value=$2
    local file=$3
    perl -i -pe "s|^([[:space:]]*)$key:.*|  $key: $value|g if /^evaluation:/../^[^\s]/" "$file"
}

# Make a copy of the original config
cp "$CONFIG_PATH" "$TEMP_CONFIG"

# Nested loops for parameter combinations
for sampler in "${SAMPLERS[@]}"; do
    for walk in "${WALKS[@]}"; do
        for guidance in "${GUIDANCE_SCALES[@]}"; do
            for num_step in "${NUM_STEPS[@]}"; do
                case "$sampler" in
                    "DDIM")
                        for eta in "${ETAS[@]}"; do
                            # Updated OUTPUT_DIR to include walks
                            OUTPUT_DIR="${ROOT_DIR}/${sampler}/walks_${walk}/guidance_${guidance}/eta_${eta}/steps_${num_step}"
                            mkdir -p "$OUTPUT_DIR"
                            
                            # Update config file
                            update_yaml "sampler" "\"$sampler\"" "$TEMP_CONFIG"
                            update_yaml "walks" "$walk" "$TEMP_CONFIG"
                            update_yaml "guidance_scale" "$guidance" "$TEMP_CONFIG"
                            update_yaml "eta" "$eta" "$TEMP_CONFIG"
                            update_yaml "num_steps" "$num_step" "$TEMP_CONFIG"
                            update_yaml "solver_order" "1" "$TEMP_CONFIG"
                            update_yaml "algorithm_type" "\"dpmsolver\"" "$TEMP_CONFIG"
                            update_yaml "save_path" "'${OUTPUT_DIR}/'" "$TEMP_CONFIG"
                            
                            # Copy current config to output directory
                            cp "$TEMP_CONFIG" "${OUTPUT_DIR}/config.yaml"
                            
                            echo "Running evaluation with parameters:"
                            echo "Sampler: $sampler"
                            echo "Walks: $walk"
                            echo "Guidance Scale: $guidance"
                            echo "Eta: $eta"
                            echo "Num Steps: $num_step"
                            echo "Output Directory: $OUTPUT_DIR"
                            echo "----------------------------------------"
                            
                            # Modified command for Hydra
                            CUDA_VISIBLE_DEVICES=0 python "$EVAL_SCRIPT" --config-dir="$(dirname "$CONFIG_PATH")" --config-name="$(basename "$CONFIG_PATH" .yaml)" \
                                evaluation.sampler="$sampler" \
                                evaluation.walks=$walk \
                                evaluation.guidance_scale=$guidance \
                                evaluation.eta=$eta \
                                evaluation.num_steps=$num_step \
                                evaluation.solver_order=1 \
                                evaluation.algorithm_type="dpmsolver" \
                                evaluation.save_path="${OUTPUT_DIR}/"
                        done
                        ;;
                    
                    "DPM++")
                        for solver_order in "${SOLVER_ORDERS[@]}"; do
                            for algorithm_type in "${ALGORITHM_TYPES[@]}"; do
                                # Updated OUTPUT_DIR to include walks
                                OUTPUT_DIR="${ROOT_DIR}/${sampler}/walks_${walk}/guidance_${guidance}/solver_${solver_order}/algo_${algorithm_type}/steps_${num_step}"
                                mkdir -p "$OUTPUT_DIR"
                                
                                # Update config file
                                update_yaml "sampler" "\"$sampler\"" "$TEMP_CONFIG"
                                update_yaml "walks" "$walk" "$TEMP_CONFIG"
                                update_yaml "guidance_scale" "$guidance" "$TEMP_CONFIG"
                                update_yaml "eta" "0.0" "$TEMP_CONFIG"
                                update_yaml "num_steps" "$num_step" "$TEMP_CONFIG"
                                update_yaml "solver_order" "$solver_order" "$TEMP_CONFIG"
                                update_yaml "algorithm_type" "\"$algorithm_type\"" "$TEMP_CONFIG"
                                update_yaml "save_path" "'${OUTPUT_DIR}/'" "$TEMP_CONFIG"
                                
                                # Copy current config to output directory
                                cp "$TEMP_CONFIG" "${OUTPUT_DIR}/config.yaml"
                                
                                echo "Running evaluation with parameters:"
                                echo "Sampler: $sampler"
                                echo "Walks: $walk"
                                echo "Guidance Scale: $guidance"
                                echo "Solver Order: $solver_order"
                                echo "Algorithm Type: $algorithm_type"
                                echo "Num Steps: $num_step"
                                echo "Output Directory: $OUTPUT_DIR"
                                echo "----------------------------------------"
                                
                                # Modified command for Hydra
                                CUDA_VISIBLE_DEVICES=0 python "$EVAL_SCRIPT" --config-dir="$(dirname "$CONFIG_PATH")" --config-name="$(basename "$CONFIG_PATH" .yaml)" \
                                    evaluation.sampler="$sampler" \
                                    evaluation.walks=$walk \
                                    evaluation.guidance_scale=$guidance \
                                    evaluation.eta=0.0 \
                                    evaluation.num_steps=$num_step \
                                    evaluation.solver_order=$solver_order \
                                    evaluation.algorithm_type="$algorithm_type" \
                                    evaluation.save_path="${OUTPUT_DIR}/"
                            done
                        done
                        ;;
                esac
            done
        done
    done
done

# Clean up
rm "$TEMP_CONFIG"
