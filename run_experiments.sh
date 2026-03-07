#!/bin/bash
# run_experiments.sh
# Runs two 15-epoch trainings (weighted / unweighted) sequentially,
# then evaluates both models on the held-out test set.
# Logs are saved separately for each run.

PYTHON=/home/sa2556/.conda/envs/imageflownet/bin/python3
DIR=/nfs/roberts/project/pi_sk2433/sa2556/DCD_state_space

cd "$DIR"
mkdir -p logs results

echo "========================================"
echo " Experiment 1: WITH class weights"
echo "========================================"
$PYTHON train.py --config config_weighted.yaml 2>&1 | tee logs/weighted_train.log

echo ""
echo "========================================"
echo " Experiment 2: WITHOUT class weights"
echo "========================================"
$PYTHON train.py --config config_unweighted.yaml 2>&1 | tee logs/unweighted_train.log

echo ""
echo "========================================"
echo " Evaluation: weighted model"
echo "========================================"
$PYTHON evaluate.py --config config_weighted.yaml 2>&1 | tee logs/weighted_eval.log

echo ""
echo "========================================"
echo " Evaluation: unweighted model"
echo "========================================"
$PYTHON evaluate.py --config config_unweighted.yaml 2>&1 | tee logs/unweighted_eval.log

echo ""
echo "All done. Results saved to results/"
