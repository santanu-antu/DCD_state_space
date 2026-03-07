"""
Fit and persist StandardScaler objects for static, dynamic, and medication
features.  Only observed dynamic values (mask == 1) are used to fit the
dynamic scalers, so carry-forward-imputed values don't distort the statistics.

Usage:
python -m preprocessing.normalize --config path/to/config.yaml   # uses config.yaml by default
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


def fit_scalers(cfg: dict) -> dict:
    data_cfg = cfg["data"]

    print("Loading CSVs...")
    static_df = pd.read_csv(data_cfg["static_csv"])
    dyn_df    = pd.read_csv(data_cfg["dyn_csv"])
    mask_df   = pd.read_csv(data_cfg["mask_csv"])
    int_df    = pd.read_csv(data_cfg["int_csv"])

    static_cols            = data_cfg["static_cols"]
    static_continuous_cols = data_cfg["static_continuous_cols"]
    dyn_cols               = [c for c in dyn_df.columns if c.startswith("dynamic_")]
    int_cols               = [c for c in int_df.columns if c.startswith("medication_")]

    # Static scaler (continuous columns only)
    # SEX and has_dialysisorders are binary (0/1). So skipping them.
    print(f"Fitting static scaler on continuous cols: {static_continuous_cols}...")
    static_scaler = StandardScaler()
    static_scaler.fit(static_df[static_continuous_cols].astype(float).values)

    # Dynamic scaler (fit only on observed cells)
    print("Fitting dynamic scaler (observed-only)...")
    dyn_vals  = dyn_df[dyn_cols].values.astype(float)
    mask_vals = mask_df[dyn_cols].values.astype(float)

    dyn_scaler = StandardScaler()
    means = np.zeros(len(dyn_cols))
    stds  = np.ones(len(dyn_cols))
    for i in range(len(dyn_cols)):
        observed = dyn_vals[:, i][mask_vals[:, i] == 1]
        observed = observed[~np.isnan(observed)]
        if len(observed) > 1:
            means[i] = observed.mean()
            stds[i]  = observed.std()
            if stds[i] == 0:
                stds[i] = 1.0
    dyn_scaler.mean_  = means
    dyn_scaler.scale_ = stds
    dyn_scaler.var_   = stds ** 2
    dyn_scaler.n_features_in_ = len(dyn_cols)

    # Medications: binary (1.0 or NaN). no scaler needed
    # NaN -> 0 (not given) is handled in dataset.py; 1.0 stays as is
    print("Medications are binary — skipping medication scaler.")

    scalers = {
        "static": static_scaler,
        "dynamic": dyn_scaler,
        "static_cols": static_cols,
        "static_continuous_cols": static_continuous_cols,
        "dyn_cols": dyn_cols,
        "int_cols": int_cols,
    }
    return scalers


def save_scalers(scalers: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved → {path}")


def load_scalers(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    scalers = fit_scalers(cfg)
    save_scalers(scalers, cfg["data"]["scalers_path"])


if __name__ == "__main__":
    main()
