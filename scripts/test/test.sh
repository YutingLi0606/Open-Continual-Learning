#!/bin/bash

METHOD="${1:-finetune}"

declare -A configs=(
    ["finetune"]="ckpt/exp_finetune scripts/test/test_ft.sh"
    ["lwf"]="ckpt/exp_lwf_fair_large scripts/test/test_lwf.sh" 
    ["lwf_vr"]="ckpt/exp_lwf_vr_fair scripts/test/test_lwf_vr.sh"
    ["wiseft"]="ckpt/exp_wiseft scripts/test/test_wiseft.sh"
)

config=${configs[$METHOD]}
if [[ -z "$config" ]]; then
    echo "Usage: $0 [finetune|lwf|lwf_vr|wiseft]"
    exit 1
fi

read -r SAVE_PATH TEST_SH <<< "$config"

source ~/.bashrc
conda activate moe

python test/zscl_matrix_evaluation.py \
    --test-script "$TEST_SH" \
    --experiment-name "${METHOD}_eval" \
    --matrix-csv "accuracy_matrix.csv" \
    --metrics-csv "${METHOD}_metrics.csv" \
    --checkpoint-dir "$SAVE_PATH"

METRICS_FILE="$SAVE_PATH/${METHOD}_metrics.csv"
if [[ -f "$METRICS_FILE" ]]; then
    awk -F',' 'NR>1 {printf "%s: %s\n", $1, $NF}' "$METRICS_FILE"
fi