"""
Dataset for the baseline ODE-RNN model (prev_model.py).

Unlike the dual-stream model, vitals and medications are combined into a single
flat feature vector X of shape (T, n_feat=25).  Missingness mask M has the same
shape.  Time t is (T, 1) in raw hours relative to each patient's first
observation in the sequence.

Padding is applied at the FRONT (zero-rows prepended) so every sample in a
batch has the same T_max.  The model's _step skips timesteps where
Mi.max() <= 0.5, which correctly ignores front-padded rows.

max_seq_len caps each patient at the last N observations (closest to
extubation) before padding, keeping T_max tractable.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


STATIC_COLS  = ["SEX", "WEIGHT_IN_KG", "BMI", "AGE", "has_dialysisorders"]
STATIC_CONT  = ["WEIGHT_IN_KG", "BMI", "AGE"]   # cols to StandardScale
TARGET_COL   = "time_extub_to_death_hours"
LABEL_COL    = "time_range"
TIME_COL     = "time_to_extube_hours"
PID_COL      = "PAT_ID"

# class edges for 4-class label scheme (identical to the main model)
LABEL_EDGES  = [30, 60, 90]


class BaselineDataset(Dataset):
    """
    Parameters
    ----------
    dyn_csv, miss_csv, static_csv : paths to CSVs
    max_seq_len : keep last N timesteps per patient (None = keep all)
    scalers     : dict with keys "mean_dyn", "std_dyn", "mean_stat", "std_stat"
                  (numpy arrays).  If None, raw (un-normalised) values are returned.
    """

    def __init__(
        self,
        dyn_csv:     str,
        miss_csv:    str,
        static_csv:  str,
        max_seq_len: int | None = 300,
        scalers:     dict | None = None,
    ):
        self.dyn_df  = pd.read_csv(dyn_csv)
        self.miss_df = pd.read_csv(miss_csv)
        self.stat_df = pd.read_csv(static_csv).set_index(PID_COL)

        self.feat_cols = [c for c in self.dyn_df.columns
                          if c not in (PID_COL, TIME_COL)]  # 25 features

        self.dyn_g  = self.dyn_df.groupby(PID_COL, sort=False)
        self.miss_g = self.miss_df.groupby(PID_COL, sort=False)

        self.pat_ids = [pid for pid in self.stat_df.index
                        if pid in self.dyn_g.groups]

        # class labels
        hours = self.stat_df.loc[self.pat_ids][TARGET_COL].values.astype(float)
        self.labels = np.digitize(hours, LABEL_EDGES).astype(int)

        self.max_seq_len = max_seq_len
        self.scalers     = scalers

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pat_ids)

    def __getitem__(self, idx):
        pid = self.pat_ids[idx]

        # ── Dynamic ─────────────────────────────────────────────────────
        dyn  = self.dyn_g.get_group(pid).sort_values(TIME_COL)
        miss = self.miss_g.get_group(pid).sort_values(TIME_COL)

        t_raw = dyn[TIME_COL].to_numpy(dtype=np.float32)   # (T,)
        X_raw = dyn[self.feat_cols].to_numpy(dtype=np.float32)   # (T, F)
        M_raw = miss[self.feat_cols].to_numpy(dtype=np.float32)  # (T, F)
        X_raw = np.nan_to_num(X_raw, nan=0.0)

        # cap to last max_seq_len observations
        if self.max_seq_len is not None and len(t_raw) > self.max_seq_len:
            t_raw = t_raw[-self.max_seq_len:]
            X_raw = X_raw[-self.max_seq_len:]
            M_raw = M_raw[-self.max_seq_len:]

        # ── Normalise X ──────────────────────────────────────────────────
        if self.scalers is not None:
            X_raw = (X_raw - self.scalers["mean_dyn"]) / self.scalers["std_dyn"]

        # time relative to first observation in window, shape (T, 1)
        t_rel = (t_raw - t_raw[0]).reshape(-1, 1)

        # ── Static ──────────────────────────────────────────────────────
        s_row  = self.stat_df.loc[pid]
        s_vals = s_row[STATIC_COLS].to_numpy(dtype=np.float32).copy()
        if self.scalers is not None:
            cont_idx = [STATIC_COLS.index(c) for c in STATIC_CONT]
            s_vals[cont_idx] = ((s_vals[cont_idx] - self.scalers["mean_stat"])
                                / self.scalers["std_stat"])

        return dict(
            pid   = pid,
            X     = torch.tensor(X_raw,  dtype=torch.float32),   # (T, F)
            M     = torch.tensor(M_raw,  dtype=torch.float32),   # (T, F)
            t     = torch.tensor(t_rel,  dtype=torch.float32),   # (T, 1)
            s     = torch.tensor(s_vals, dtype=torch.float32),   # (n_static,)
            y_cls = torch.tensor(int(self.labels[idx]), dtype=torch.long),
        )


# ──────────────────────────────────────────────────────────────────────────────
def collate_baseline(batch: list[dict]) -> dict:
    """
    Pad sequences at the FRONT (prepend zero rows) so every sample reaches
    the same T_max.  Zero-padded rows have M==0, so _step will skip them.
    """
    T_max = max(b["X"].shape[0] for b in batch)
    B     = len(batch)
    F     = batch[0]["X"].shape[1]

    X_pad = torch.zeros(B, T_max, F)
    M_pad = torch.zeros(B, T_max, F)
    t_pad = torch.zeros(B, T_max, 1)
    lens  = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        T = b["X"].shape[0]
        start = T_max - T          # front-pad offset
        X_pad[i, start:, :] = b["X"]
        M_pad[i, start:, :] = b["M"]
        t_pad[i, start:, :] = b["t"]
        lens[i] = T

    return dict(
        pid   = [b["pid"]   for b in batch],
        X     = X_pad,                           # (B, T_max, F)
        M     = M_pad,                           # (B, T_max, F)
        t     = t_pad,                           # (B, T_max, 1)
        s     = torch.stack([b["s"]     for b in batch]),  # (B, n_static)
        y_cls = torch.stack([b["y_cls"] for b in batch]),  # (B,)
        lens  = lens,                            # (B,)
    )


# ──────────────────────────────────────────────────────────────────────────────
def fit_scalers(dataset: "BaselineDataset", train_idx: list[int]) -> dict:
    """
    Fit mean/std on training patients only (no leakage from val/test).
    Dynamic: ALL cells including imputed/fill-forward, matching the original
    get_mean_std() behaviour in prev_model_train.py.
    Static: training rows, continuous cols only.
    """
    train_pids = set(dataset.pat_ids[i] for i in train_idx)

    # ── Dynamic scaler ───────────────────────────────────────────────────
    dyn_train  = dataset.dyn_df[dataset.dyn_df[PID_COL].isin(train_pids)]
    miss_train = dataset.miss_df[dataset.miss_df[PID_COL].isin(train_pids)]

    X_all = dyn_train[dataset.feat_cols].to_numpy(dtype=np.float32)
    M_all = miss_train[dataset.feat_cols].to_numpy(dtype=np.float32)

    n_feat = len(dataset.feat_cols)
    mean_dyn = np.zeros(n_feat, dtype=np.float32)
    std_dyn  = np.ones(n_feat,  dtype=np.float32)
    for i in range(n_feat):
        # Use ALL cells (including imputed), matching original get_mean_std() behaviour.
        all_vals = X_all[:, i]
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) > 1:
            mean_dyn[i] = all_vals.mean()
            std_dyn[i]  = all_vals.std() or 1.0

    # ── Static scaler (continuous cols only) ─────────────────────────────
    cont_idx   = [STATIC_COLS.index(c) for c in STATIC_CONT]
    stat_train = dataset.stat_df.loc[list(train_pids)]
    s_vals     = stat_train[STATIC_COLS].to_numpy(dtype=np.float32)
    s_cont     = s_vals[:, cont_idx]
    mean_stat  = s_cont.mean(axis=0)
    std_stat   = s_cont.std(axis=0)
    std_stat[std_stat == 0] = 1.0

    return dict(mean_dyn=mean_dyn, std_dyn=std_dyn,
                mean_stat=mean_stat, std_stat=std_stat)
