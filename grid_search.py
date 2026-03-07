"""
grid_search.py
--------------
Random hyperparameter search over:
  - learning rate
  - model size (d_h, d_z)
  - weight decay
  - dropout
  - use_class_weights

Each configuration is trained for 10 epochs.
The best configuration (lowest val loss at epoch 10) is reported.

Usage
-----
python grid_search.py               # 16 random samples (default)
python grid_search.py --n_samples 8 # fewer samples
"""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from itertools import product

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "lr":                [1e-4, 5e-4, 1e-3],
    "model_size":        ["small", "medium", "large"],   # maps to (d_h, d_z, d_u)
    "weight_decay":      [1e-5, 1e-4, 1e-3],
    "dropout":           [0.1, 0.2, 0.3],
    "use_class_weights": [True, False],
}

MODEL_SIZES = {
    "small":  {"d_h": 32,  "d_z": 64,  "d_u": 16},
    "medium": {"d_h": 64,  "d_z": 128, "d_u": 32},
    "large":  {"d_h": 128, "d_z": 256, "d_u": 64},
}

# ─────────────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "data": {
        "static_csv":   "data/static_target.csv",
        "dyn_csv":      "data/dynamic_variables.csv",
        "mask_csv":     "data/missing_variables.csv",
        "int_csv":      "data/medications.csv",
        "scalers_path": "data_preprocessed/scalers.pkl",
        "static_cols":  ["SEX", "WEIGHT_IN_KG", "BMI", "AGE", "has_dialysisorders"],
        "static_continuous_cols": ["WEIGHT_IN_KG", "BMI", "AGE"],
        "static_binary_cols":     ["SEX", "has_dialysisorders"],
        "target_col":  "time_extub_to_death_hours",
        "label_col":   "time_range",
        "time_col":    "time_to_extube_hours",
        "pid_col":     "PAT_ID",
    },
    "split": {"train_frac": 0.70, "val_frac": 0.15, "seed": 42},
    "model": {
        "n_static": 5, "n_dyn": 20, "n_int": 5, "n_classes": 4,
        "n_read": 10,
    },
    "training": {
        "batch_size":   16,
        "epochs":       10,
        "grad_clip":    1.0,
        "lr_scheduler": "plateau",
        "lr_patience":  3,
        "lr_factor":    0.5,
    },
    "paths": {
        "checkpoint_dir": "checkpoints/",
        "log_dir":        "logs/grid_search/",
    },
}

PYTHON = sys.executable


# ─────────────────────────────────────────────────────────────────────────────
def make_config(run_id: int, params: dict) -> dict:
    cfg = copy.deepcopy(BASE_CONFIG)

    # learning rate and weight decay
    cfg["training"]["lr"]                = params["lr"]
    cfg["training"]["weight_decay"]      = params["weight_decay"]
    cfg["training"]["use_class_weights"] = params["use_class_weights"]

    # model size
    sz = MODEL_SIZES[params["model_size"]]
    cfg["model"]["d_h"]      = sz["d_h"]
    cfg["model"]["d_z"]      = sz["d_z"]
    cfg["model"]["d_u"]      = sz["d_u"]
    cfg["model"]["dropout"]  = params["dropout"]

    # paths unique per run
    tag = f"run{run_id:03d}"
    cfg["paths"]["best_model"] = f"checkpoints/gs_{tag}.pt"
    cfg["paths"]["results"]    = f"results/gs_{tag}.json"

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
def run_training(cfg: dict, cfg_path: str) -> float | None:
    """Run training and return best val loss. Returns None on failure."""
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    result = subprocess.run(
        [PYTHON, "train.py", "--config", cfg_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  [ERROR] Training failed:\n{result.stderr[-500:]}")
        return None

    # Extract best val loss from stdout
    best_val_loss = None
    for line in result.stdout.splitlines():
        if "Saved best model" in line:
            try:
                best_val_loss = float(line.split("val_loss=")[1].rstrip(")"))
            except (IndexError, ValueError):
                pass

    # Print epoch lines to show progress
    for line in result.stdout.splitlines():
        if line.startswith("["):
            print(f"  {line}", flush=True)

    return best_val_loss


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    # Build all combinations and sample randomly
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    all_combos = [dict(zip(keys, v)) for v in product(*values)]
    sampled    = random.sample(all_combos, min(args.n_samples, len(all_combos)))

    os.makedirs("checkpoints",        exist_ok=True)
    os.makedirs("logs/grid_search",   exist_ok=True)
    os.makedirs("results",            exist_ok=True)

    results = []
    print(f"\nGrid search: {len(sampled)} random configurations, 10 epochs each")
    print(f"Total estimated time: ~{len(sampled) * 40 / 60:.1f} hours\n")

    for i, params in enumerate(sampled, 1):
        print(f"\n[{i:02d}/{len(sampled)}] {params}")
        cfg      = make_config(i, params)
        cfg_path = f"logs/grid_search/config_run{i:03d}.yaml"

        t0 = time.time()
        val_loss = run_training(cfg, cfg_path)
        elapsed  = time.time() - t0

        record = {
            "run_id":    i,
            "params":    params,
            "val_loss":  val_loss,
            "elapsed_s": round(elapsed),
        }
        results.append(record)

        status = f"val_loss={val_loss:.4f}" if val_loss is not None else "FAILED"
        print(f"  → {status}  ({elapsed/60:.1f} min)")

        # Save incremental results after every run
        with open("logs/grid_search/results.json", "w") as f:
            json.dump(sorted(results, key=lambda r: r["val_loss"] or 9999), f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in results if r["val_loss"] is not None]
    if not valid:
        print("\nAll runs failed.")
        return

    valid.sort(key=lambda r: r["val_loss"])
    best = valid[0]

    print("\n" + "=" * 60)
    print(f"  Best val loss: {best['val_loss']:.4f}")
    print(f"  Best params:   {best['params']}")
    print(f"  Config saved:  logs/grid_search/config_run{best['run_id']:03d}.yaml")
    print("=" * 60)

    print("\nTop 5 configurations:")
    print(f"  {'rank':>5}  {'val_loss':>10}  params")
    for rank, r in enumerate(valid[:5], 1):
        print(f"  {rank:>5}  {r['val_loss']:>10.4f}  {r['params']}")

    with open("logs/grid_search/results.json", "w") as f:
        json.dump(valid, f, indent=2)
    print("\nFull results → logs/grid_search/results.json")


if __name__ == "__main__":
    main()
