#!/bin/bash
source ~/.bashrc
conda activate imageflownet
cd /nfs/roberts/project/pi_sk2433/sa2556/DCD_state_space

SEEDS=(42 123 456 789 888)
BASE_DIR="results_new_model/bf21"

for seed in "${SEEDS[@]}"; do
    echo "================================================="
    echo "Running pipeline for seed: $seed"
    echo "================================================="
    
    OUT_DIR="${BASE_DIR}/${seed}/classification_target1dim4"
    mkdir -p "$OUT_DIR"
    
    echo "Training..."
    python train_full.py --config config_yale_bf.yaml --seed $seed --out_dir "$OUT_DIR"
    
    echo "Evaluating..."
    python evaluate_transfer.py --config config_yale_bf.yaml --test_config config_yale_af.yaml --out_dir "$OUT_DIR"
done
