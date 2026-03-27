import pandas as pd
import torch
import sys
from pathlib import Path
from model.dataset import YaleDatasetWithMissingnessInfo

def pad_or_truncate(tensor, target_rows, pad_with_zeros=False):
    k = tensor.size(0)
    if k < target_rows:
        if pad_with_zeros:
            padding = torch.zeros(target_rows - k, tensor.size(1), dtype=tensor.dtype, device=tensor.device)
        else:
            padding = tensor[0].unsqueeze(0).repeat(target_rows - k, 1)
        return torch.cat((padding, tensor), dim=0)
    elif k > target_rows:
        return tensor[-target_rows:]
    else:
        return tensor

def _to_float_tensor(df_or_series, cols, what):
    x = df_or_series[cols]
    x_num = pd.to_numeric(x, errors="coerce") if isinstance(x, pd.Series) else x.apply(pd.to_numeric, errors="coerce")
    if isinstance(x_num, pd.Series):
        if x_num.isna().any():
            bad = [c for c in cols if pd.isna(x_num[c])]
            raise ValueError(f"{what}: non-numeric or missing values in columns: {bad}")
        arr = x_num.values.astype("float32")
    else:
        bad_cols = [c for c in cols if x_num[c].isna().any()]
        if bad_cols:
            raise ValueError(f"{what}: non-numeric or missing values found in columns: {bad_cols}")
        arr = x_num.values.astype("float32")
    return torch.tensor(arr, dtype=torch.float32)

def build_dataset(dynamic_csv, missing_csv, static_target_csv, output_pt, medications_csv=None, id_col="PAT_ID", time_col="time_to_extube_hours", max_len=100):
    dyn = pd.read_csv(dynamic_csv)
    mis = pd.read_csv(missing_csv)
    sta = pd.read_csv(static_target_csv)

    if medications_csv:
        med = pd.read_csv(medications_csv)
        # Merge dynamic features and medications
        # Outer merge to include potential timestamps only present in medications
        dyn = pd.merge(dyn, med, on=[id_col, time_col], how='outer')
        
        # Merge the truth arrays to handle missingness of the new meds
        # 1 means observed, 0 means unobserved
        med_mask = med.copy()
        med_feat_cols = [c for c in med.columns if c not in [id_col, time_col]]
        for col in med_feat_cols:
            # If the value exists in meds, it is considered observed (1.0). If empty, unobserved (0.0).
            med_mask[col] = med_mask[col].notna().astype(float)
        
        mis = pd.merge(mis, med_mask, on=[id_col, time_col], how='outer')
        
        # Now fill NaN values appropriately:
        # In `dyn`, unobserved meds are typically represented as 1.0 (as found in yale_bf21.pt)
        for col in med_feat_cols:
            dyn[col] = dyn[col].fillna(1.0)
            mis[col] = mis[col].fillna(0.0) # Unobserved

        # Also, because of outer merge, there could be NaN in standard dynamic columns.
        # We fill missing mask for standard variables with 0.0 (unobserved)
        std_dyn_cols = [c for c in dyn.columns if c not in med_feat_cols and c not in [id_col, time_col]]
        for col in std_dyn_cols:
            mis[col] = mis[col].fillna(0.0)
            
        # We should logically carry forward (ffill) the dynamic variables, 
        # or fill them with 0.0, to match whatever was done. Let's just ffill then fillna.
        dyn[std_dyn_cols] = dyn.groupby(id_col)[std_dyn_cols].ffill()
        dyn[std_dyn_cols] = dyn[std_dyn_cols].fillna(0.0)

    dyn_feat_cols = [c for c in dyn.columns if c not in [id_col, time_col]]
    mis_feat_cols = [c for c in mis.columns if c not in [id_col, time_col]]

    exclude_static = {id_col, "time_extub_to_death_hours", "<30", "30-59", "60-89", ">=90", "time_range"}
    static_cols = [c for c in sta.columns if c not in exclude_static]
    y_cols = ["time_extub_to_death_hours", "time_range"]

    # We instantiate a mostly empty instance directly, then override its internal self.data
    dataset = YaleDatasetWithMissingnessInfo.__new__(YaleDatasetWithMissingnessInfo)
    samples = []
    
    common_ids = sorted(set(dyn[id_col]).intersection(set(mis[id_col])).intersection(set(sta[id_col])))

    for pid in common_ids:
        d = dyn[dyn[id_col] == pid].sort_values(time_col).reset_index(drop=True)
        m = mis[mis[id_col] == pid].sort_values(time_col).reset_index(drop=True)
        srow = sta[sta[id_col] == pid].iloc[0]

        n = min(len(d), len(m))
        d, m = d.iloc[:n], m.iloc[:n]

        X = _to_float_tensor(d, dyn_feat_cols, f"PAT_ID={pid} dynamic")
        M = _to_float_tensor(m, mis_feat_cols, f"PAT_ID={pid} missingness")
        t = _to_float_tensor(d, [time_col], f"PAT_ID={pid} time")
        s = _to_float_tensor(srow, static_cols, f"PAT_ID={pid} static")
        y = _to_float_tensor(srow, y_cols, f"PAT_ID={pid} target")

        if t.ndim == 1:
            t = t.unsqueeze(-1)
            
        X = pad_or_truncate(X, max_len, pad_with_zeros=False)
        M = pad_or_truncate(M, max_len, pad_with_zeros=True)
        t = pad_or_truncate(t, max_len, pad_with_zeros=False)

        samples.append((X, M, t, s, y))

    dataset.data = samples
    
    Path(output_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_pt)
    print(f"Saved dataset with {len(dataset)} patients to {output_pt}")

if __name__ == "__main__":
    import os
    
    # Using relative paths to the actual_baseline directory mapping to project root structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    build_dataset(
        dynamic_csv=os.path.join(base_dir, "data/dynamic_variables.csv"),
        missing_csv=os.path.join(base_dir, "data/missing_variables.csv"),
        static_target_csv=os.path.join(base_dir, "data/static_target.csv"),
        medications_csv=os.path.join(base_dir, "data/medications.csv"),
        output_pt=os.path.join(base_dir, "actual_baseline/data/yale_custom.pt"),
    )
