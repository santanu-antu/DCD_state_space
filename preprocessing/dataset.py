import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# LABEL SCHEMES
# Each scheme maps from the continuous target (hours) to integer class labels
# using np.digitize(hours, edges):  class k = number of edges that hours exceeds.
LABEL_SCHEMES: dict[str, dict] = {
    "4class": {
        "edges": [30, 60, 90],           # bins: [0,30), [30,60), [60,90), [90,∞)
        "names": ["<30h", "30-59h", "60-89h", ">=90h"],
    },
    "6class": {
        "edges": [30, 60, 120, 180, 240],  # bins: [0,30), [30,60), [60,120), [120,180), [180,240), [240,∞)
        "names": ["<30h", "30-60h", "60-120h", "120-180h", "180-240h", ">240h"],
    },
}


class ICUStreamsDataset(Dataset):
    def __init__(
        self,
        static_csv: str,
        dyn_csv: str,
        dyn_mask_csv: str,
        int_csv: str,
        static_cols=("SEX", "WEIGHT_IN_KG", "BMI", "AGE", "has_dialysisorders"),
        target_col="time_extub_to_death_hours",
        label_col="time_range",
        time_col="time_to_extube_hours",
        pid_col="PAT_ID",
        normalize: bool = False,
        scalers_path: str | None = None,
        task: str = "cls",        # "cls" | "reg" | "both"
        label_scheme: str | None = None,  # "4class" | "6class" | None → use label_col from CSV
    ):
        self.static_df = pd.read_csv(static_csv)
        self.dyn_df    = pd.read_csv(dyn_csv)
        self.mask_df   = pd.read_csv(dyn_mask_csv)
        self.int_df    = pd.read_csv(int_csv)

        # columns
        self.pid_col     = pid_col
        self.time_col    = time_col
        self.static_cols = list(static_cols)
        self.target_col  = target_col
        self.label_col   = label_col
        self.task        = task

        self.dyn_cols = [c for c in self.dyn_df.columns if c.startswith("dynamic_")]
        self.int_cols = [c for c in self.int_df.columns if c.startswith("medication_")]

        # Scalers 
        self.normalize = normalize
        self.scalers   = None
        if normalize:
            if scalers_path is None:
                raise ValueError("scalers_path must be provided when normalize=True")
            with open(scalers_path, "rb") as f:
                self.scalers = pickle.load(f)
            # Indices of continuous static cols within static_cols (for selective scaling)
            cont_cols = self.scalers.get("static_continuous_cols", list(static_cols))
            self._static_cont_idx = [list(static_cols).index(c) for c in cont_cols]

        # group for fast lookup
        self.dyn_g  = self.dyn_df.groupby(pid_col, sort=False)
        self.mask_g = self.mask_df.groupby(pid_col, sort=False)
        self.int_g  = self.int_df.groupby(pid_col, sort=False)

        self.pat_ids = []
        for _, row in self.static_df.iterrows():
            pid = row[pid_col]
            if pid in self.dyn_g.groups and pid in self.mask_g.groups:
                self.pat_ids.append(pid)

        # make static row lookup
        self.static_df = self.static_df.set_index(pid_col)

        # ── Precompute integer labels for all patients ──────────────────────
        # If label_scheme is set, derive labels from the continuous target_col;
        # otherwise fall back to the precomputed label_col column in the CSV.
        self.label_scheme = label_scheme
        if label_scheme is not None:
            if label_scheme not in LABEL_SCHEMES:
                raise ValueError(
                    f"Unknown label_scheme {label_scheme!r}. "
                    f"Choose from: {list(LABEL_SCHEMES)}"
                )
            scheme = LABEL_SCHEMES[label_scheme]
            self.label_names: list[str] = scheme["names"]
            hours = self.static_df.loc[self.pat_ids][target_col].values.astype(float)
            self.labels: np.ndarray = np.digitize(hours, scheme["edges"]).astype(int)
        else:
            self.label_names = None
            self.labels = self.static_df.loc[self.pat_ids][label_col].values.astype(int)

    def __len__(self):
        return len(self.pat_ids)

    def __getitem__(self, idx):
        pid = self.pat_ids[idx]

        # Static 
        s_row  = self.static_df.loc[pid]
        s_vals = s_row[self.static_cols].astype(float).values.copy()  # (n_static,)
        if self.normalize and self.scalers is not None:
            # Scale only continuous columns; binary columns (SEX, has_dialysisorders) stay as 0/1
            cont_idx = self._static_cont_idx
            cont_vals = s_vals[cont_idx].reshape(1, -1)
            s_vals[cont_idx] = self.scalers["static"].transform(cont_vals).squeeze(0)
        S = torch.tensor(s_vals, dtype=torch.float32)

        y     = torch.tensor(float(s_row[self.target_col]), dtype=torch.float32)
        y_cls = torch.tensor(int(self.labels[idx]),          dtype=torch.long)

        # Dynamic + mask 
        d = self.dyn_g.get_group(pid).copy()
        m = self.mask_g.get_group(pid).copy()

        d = d.sort_values(self.time_col).reset_index(drop=True)
        m = m.sort_values(self.time_col).reset_index(drop=True)

        if len(d) != len(m):
            raise ValueError(f"{pid}: dyn and mask length mismatch {len(d)} vs {len(m)}")
        td = d[self.time_col].to_numpy()
        tm = m[self.time_col].to_numpy()
        if not np.allclose(td, tm):
            raise ValueError(f"{pid}: dyn and mask time mismatch")

        t_dyn  = torch.tensor(td, dtype=torch.float32)
        Y_dyn  = d[self.dyn_cols].to_numpy().astype(float)
        M_dyn  = m[self.dyn_cols].to_numpy().astype(float)

        if self.normalize and self.scalers is not None:
            Y_dyn = self.scalers["dynamic"].transform(Y_dyn)

        Y_dyn = torch.tensor(Y_dyn, dtype=torch.float32)
        M_dyn = torch.tensor(M_dyn, dtype=torch.float32)   # 1=observed, 0=imputed
        Y_dyn = torch.nan_to_num(Y_dyn, nan=0.0)

        # Interventions 
        if pid in self.int_g.groups:
            u     = self.int_g.get_group(pid).copy().sort_values(self.time_col).reset_index(drop=True)
            t_int = torch.tensor(u[self.time_col].to_numpy(), dtype=torch.float32)
            U_int = u[self.int_cols].to_numpy().astype(float)
            # Medications are binary (1.0 = given, NaN = not given -> 0.0). no scaling.
            U_int = torch.tensor(U_int, dtype=torch.float32)
            U_int = torch.nan_to_num(U_int, nan=0.0)
        else:
            t_int = torch.empty(0, dtype=torch.float32)
            U_int = torch.empty((0, len(self.int_cols)), dtype=torch.float32)

        return {
            "pid":   pid,
            "S":     S,                    # [n_static]
            "t_dyn": t_dyn,               # [T]
            "Y_dyn": Y_dyn,               # [T, n_dyn]
            "M_dyn": M_dyn,               # [T, n_dyn]  1=observed
            "t_int": t_int,               # [K]
            "U_int": U_int,               # [K, n_int]
            "y":     y,                    # scalar regression target
            "y_cls": y_cls,               # int class label {0,1,2,3}
        }