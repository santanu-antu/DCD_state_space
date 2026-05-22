# Project Guidelines

## Code Style & General
- **Configuration-Driven**: All routing and hyperparameters are defined in YAML config files without hardcoding. See [config_yale_bf.yaml](config_yale_bf.yaml) for an example.
- **Python-focused**: The repository contains research code utilizing PyTorch, focusing on state-space models and ODE integration.

## Architecture
The main model is **DualStreamSSM** ([model/dual_stream_ssm.py](model/dual_stream_ssm.py)), which combines three independent data streams:
1. **Static Stream** ([model/static_encoder.py](model/static_encoder.py)): Projects static features (sex, weight, BMI, age) to initialize hidden states.
2. **Intervention Stream** ([model/intervention_mamba.py](model/intervention_mamba.py)): A time-aware SSM that evolves hidden states based on sparse medication events.
3. **Physiological Stream** ([model/irregular_gru.py](model/irregular_gru.py) or [model/ode_rnn_dynamic.py](model/ode_rnn_dynamic.py)): Handles physiological features using either an IrregularGRU (exponential decay between observations) or an ODERNNDynamic module.
4. **Readout** ([model/readout.py](model/readout.py)): Resamples trajectories onto a uniform time grid and applies a single-head temporal attention.

## Build and Test Execution
There are no standard `npm` or `make` test sets. Instead, agents should use the following commands to execute scripts and run model experiments:

- **Training**: `python train.py --config config_yale_bf.yaml --seed 42 --out_dir checkpoints/`
- **Evaluation (supports transfer)**: `python evaluate.py --config config_yale_bf.yaml --test_config config_yale_af.yaml --out_dir checkpoints/`
- **Hyperparameter Search**: `python grid_search.py --n_samples 16`
- **Sweep Orchestration**: `./sweep_yale_bf_to_af.sh --train-config config_yale_bf.yaml --test-config config_yale_af.yaml --num-runs 48`
- **SLURM Execution**: `sbatch run_yale_sweep.slurm`

## Conventions
- **Timestamps**: Timestamps are intentionally kept as raw hours (not normalized) to allow for physically meaningful Δt within ODE integration and exponential decay modes.
- **Output Directories**: Checkpoints operate under the `checkpoints/` directory storing `best_model.pt` and `test_metrics.json`. Sweeps output multiple runs to `sweeps/` (producing `runs.jsonl`, `summary.csv`, and `best_run.json`).
- **Data Padding**: Variable-length sequences (dynamic / medications data) are padded to the batch max. Optional `max_seq_len` truncates data by keeping the *last* N observations (those closest to extubation). See [preprocessing/collate.py](preprocessing/collate.py).
- **Split Modes**: Config splits typically use `mode: "split"` to operate on standard train/val/test splits via predefined seeds, or `mode: "full"` to train on the entire cohort downstream.
