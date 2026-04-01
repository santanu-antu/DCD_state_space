"""
train.py
--------
Unified training script capable of:
  1) Train/Val/Test splitting (mode: split) - default
  2) Full cohort training for transfer  (mode: full)

Usage
-----
python train.py --config path/to/cfg.yaml [--mode split]
python train.py --config path/to/cfg.yaml --mode full --out_dir ./checkpoints/full_model/
"""

import argparse
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset

from preprocessing.collate import collate_fn
from functools import partial
from preprocessing.dataset import ICUStreamsDataset
from model.dual_stream_ssm import DualStreamSSM

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_splits(dataset: ICUStreamsDataset, cfg: dict) -> tuple:
    sp = cfg["split"]
    seed = sp["seed"]
    labels = dataset.labels.tolist()
    indices = list(range(len(dataset)))
    tr_val_idx, te_idx, tr_val_lbl, _ = train_test_split(
        indices, labels,
        test_size=1.0 - sp["train_frac"] - sp["val_frac"],
        stratify=labels,
        random_state=seed,
    )
    val_frac_corrected = sp["val_frac"] / (sp["train_frac"] + sp["val_frac"])
    tr_idx, va_idx = train_test_split(
        tr_val_idx,
        test_size=val_frac_corrected,
        stratify=tr_val_lbl,
        random_state=seed,
    )
    return tr_idx, va_idx, te_idx

def _fit_scalers_on_train(dataset: ICUStreamsDataset, tr_idx: list[int], cfg: dict) -> dict:
    d_cfg = cfg["data"]
    pid_col = dataset.pid_col
    dyn_cols = dataset.dyn_cols
    int_cols = dataset.int_cols
    static_continuous_cols = d_cfg["static_continuous_cols"]

    train_pids = [dataset.pat_ids[i] for i in tr_idx]
    train_pid_set = set(train_pids)

    train_static = dataset.static_df.loc[train_pids]
    static_scaler = StandardScaler()
    static_scaler.fit(train_static[static_continuous_cols].astype(float).values)

    train_dyn = dataset.dyn_df[dataset.dyn_df[pid_col].isin(train_pid_set)]
    train_mask = dataset.mask_df[dataset.mask_df[pid_col].isin(train_pid_set)]

    dyn_vals = train_dyn[dyn_cols].values.astype(float)
    mask_vals = train_mask[dyn_cols].values.astype(float)

    means = np.zeros(len(dyn_cols))
    stds = np.ones(len(dyn_cols))
    for i in range(len(dyn_cols)):
        observed = dyn_vals[:, i][mask_vals[:, i] == 1]
        observed = observed[~np.isnan(observed)]
        if len(observed) > 1:
            means[i] = observed.mean()
            stds[i] = observed.std()
            if stds[i] == 0:
                stds[i] = 1.0

    dyn_scaler = StandardScaler()
    dyn_scaler.mean_ = means
    dyn_scaler.scale_ = stds
    dyn_scaler.var_ = stds ** 2
    dyn_scaler.n_features_in_ = len(dyn_cols)

    return {
        "static": static_scaler,
        "dynamic": dyn_scaler,
        "static_cols": d_cfg["static_cols"],
        "static_continuous_cols": static_continuous_cols,
        "dyn_cols": dyn_cols,
        "int_cols": int_cols,
    }

def run_epoch(model, loader, criterion, optimizer, device, grad_clip, train):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            S = batch["S"].to(device)
            t_dyn = batch["t_dyn"].to(device)
            Y_dyn = batch["Y_dyn"].to(device)
            M_dyn = batch["M_dyn"].to(device)
            t_int = batch["t_int"].to(device)
            U_int = batch["U_int"].to(device)
            dyn_lens = batch["dyn_lens"].to(device)
            int_lens = batch["int_lens"].to(device)
            y_cls = batch["y_cls"].to(device)

            logits = model(S, t_dyn, Y_dyn, M_dyn, t_int, U_int, dyn_lens, int_lens)
            loss = criterion(logits, y_cls)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            B = y_cls.shape[0]
            total_loss += loss.item() * B
            total_correct += (logits.argmax(1) == y_cls).sum().item()
            total_n += B
    return total_loss / total_n, total_correct / total_n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["split", "full"], default="split", help="split: train/val/test splits, full: train on full data")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override split seed")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save artifacts. If unused, falls back to config `paths` keys.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    seed = args.seed if args.seed is not None else cfg.get("split", {}).get("seed", 42)
    if "split" not in cfg:
        cfg["split"] = {}
    cfg["split"]["seed"] = seed
    
    if args.out_dir is not None:
        # Append the seed as a separate folder
        actual_out_dir = os.path.join(args.out_dir, str(seed))
        os.makedirs(actual_out_dir, exist_ok=True)
        cfg["paths"]["checkpoint_dir"] = actual_out_dir
        cfg["paths"]["log_dir"] = actual_out_dir
        cfg["paths"]["best_model"] = os.path.join(actual_out_dir, "best_model.pt")
        cfg["data"]["scalers_path"] = os.path.join(actual_out_dir, "scalers.pkl")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode.upper()}")

    d_cfg = cfg["data"]
    dataset = ICUStreamsDataset(
        static_csv=d_cfg["static_csv"], dyn_csv=d_cfg["dyn_csv"], dyn_mask_csv=d_cfg["mask_csv"], int_csv=d_cfg["int_csv"],
        static_cols=d_cfg.get("static_cols", []), target_col=d_cfg["target_col"], label_col=d_cfg["label_col"],
        time_col=d_cfg["time_col"], pid_col=d_cfg["pid_col"], normalize=False, task="cls", label_scheme=d_cfg.get("label_scheme", None)
    )
    print(f"Total patients: {len(dataset)}")

    if args.mode == "split":
        tr_idx, va_idx, te_idx = build_splits(dataset, cfg)
        print(f"Split — train: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)}")
        os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)
        torch.save({"tr_idx": tr_idx, "va_idx": va_idx, "te_idx": te_idx}, os.path.join(cfg["paths"]["checkpoint_dir"], "splits.pt"))
    else:
        tr_idx = list(range(len(dataset)))
        va_idx = []
        print(f"Full dataset training: {len(tr_idx)} patients")

    scalers = _fit_scalers_on_train(dataset, tr_idx, cfg)
    dataset.scalers = scalers
    dataset.normalize = True
    cont_cols = scalers["static_continuous_cols"]
    dataset._static_cont_idx = [list(dataset.static_cols).index(c) for c in cont_cols]

    scalers_path = d_cfg.get("scalers_path", "scalers.pkl")
    os.makedirs(os.path.dirname(scalers_path) or ".", exist_ok=True)
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Scalers fitted on {len(tr_idx)} patients -> {scalers_path}")

    tr_cfg = cfg["training"]
    max_seq_len = d_cfg.get("max_seq_len", None)
    _collate = partial(collate_fn, max_seq_len=max_seq_len) if max_seq_len else collate_fn

    train_loader = DataLoader(
        Subset(dataset, tr_idx), batch_size=tr_cfg["batch_size"], shuffle=True,
        collate_fn=_collate, num_workers=2, pin_memory=(device.type == "cuda")
    )
    
    val_loader = None
    if args.mode == "split":
        val_loader = DataLoader(
            Subset(dataset, va_idx), batch_size=tr_cfg["batch_size"], shuffle=False,
            collate_fn=_collate, num_workers=2, pin_memory=(device.type == "cuda")
        )

    tr_labels = [int(dataset.labels[i]) for i in tr_idx]
    if tr_cfg.get("use_class_weights", False):
        classes = np.unique(tr_labels)
        cw = compute_class_weight("balanced", classes=classes, y=tr_labels)
        # Handle cases where some classes are missing in full data (extremely rare but secure)
        cw_full = np.ones(cfg["model"]["n_classes"])
        for c_idx, c in enumerate(classes):
            cw_full[int(c)] = cw[c_idx]
        w = torch.tensor(cw_full, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    model = DualStreamSSM.from_config(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg.get("weight_decay", 1e-4))
    sched_name = tr_cfg.get("lr_scheduler", "none").lower()
    scheduler = None
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr_cfg["epochs"])
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=tr_cfg.get("lr_patience", 3), factor=tr_cfg.get("lr_factor", 0.5))

    best_val_loss = math.inf
    history = []
    
    for epoch in range(1, tr_cfg["epochs"] + 1):
        t0 = time.time()
        run_epoch(model, train_loader, criterion, optimizer, device, tr_cfg.get("grad_clip"), train=True)
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, None, device, tr_cfg.get("grad_clip"), train=False)
        
        va_loss, va_acc = 0.0, 0.0
        if val_loader:
            va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device, tr_cfg.get("grad_clip"), train=False)
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(va_loss)
                else: scheduler.step()
        else:
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        elapsed = time.time() - t0
        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc, va_loss=va_loss, va_acc=va_acc, elapsed=elapsed))
        
        if args.mode == "split":
            print(f"[{epoch:03d}/{tr_cfg['epochs']}] loss: {tr_loss:.4f}/{va_loss:.4f} acc: {tr_acc:.3f}/{va_acc:.3f} ({elapsed:.1f}s)")
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_loss": va_loss, "val_acc": va_acc}, cfg.get("paths", {}).get("best_model", "best_model.pt"))
                print(f"  ✓ Saved best model (val_loss={va_loss:.4f})")
        else:
            print(f"[{epoch:03d}/{tr_cfg['epochs']}] tr_loss: {tr_loss:.4f} tr_acc: {tr_acc:.3f} ({elapsed:.1f}s)")
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "train_loss": tr_loss}, cfg.get("paths", {}).get("best_model", "best_model.pt"))

    log_path = os.path.join(cfg.get("paths", {}).get("log_dir", "."), "log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Model -> {cfg.get('paths', {}).get('best_model', 'best_model.pt')}")

if __name__ == "__main__":
    main()
