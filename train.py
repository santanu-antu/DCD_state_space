"""
train.py
--------
Stratified train / val / test split → class-weighted CrossEntropyLoss →
AdamW + CosineAnnealingLR.

Usage
-----
python train.py                          # uses config.yaml
python train.py --config path/to/cfg.yaml
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset

from preprocessing.collate  import collate_fn
from preprocessing.dataset  import ICUStreamsDataset
from model.dual_stream_ssm  import DualStreamSSM


# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
def build_splits(dataset: ICUStreamsDataset, cfg: dict) -> tuple:
    """Stratified 70 / 15 / 15 split by time_range label."""
    sp   = cfg["split"]
    seed = sp["seed"]

    labels = [int(dataset.static_df.loc[pid][cfg["data"]["label_col"]])
              for pid in dataset.pat_ids]
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


# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    grad_clip: float,
    train:     bool,
) -> tuple[float, float]:
    """Run one epoch.  Returns (mean_loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            S        = batch["S"].to(device)
            t_dyn    = batch["t_dyn"].to(device)
            Y_dyn    = batch["Y_dyn"].to(device)
            M_dyn    = batch["M_dyn"].to(device)
            t_int    = batch["t_int"].to(device)
            U_int    = batch["U_int"].to(device)
            dyn_lens = batch["dyn_lens"].to(device)
            int_lens = batch["int_lens"].to(device)
            y_cls    = batch["y_cls"].to(device)

            logits = model(S, t_dyn, Y_dyn, M_dyn, t_int, U_int,
                           dyn_lens, int_lens)

            loss = criterion(logits, y_cls)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            B = y_cls.shape[0]
            total_loss    += loss.item() * B
            total_correct += (logits.argmax(1) == y_cls).sum().item()
            total_n       += B

    return total_loss / total_n, total_correct / total_n


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["split"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    d_cfg = cfg["data"]
    dataset = ICUStreamsDataset(
        static_csv   = d_cfg["static_csv"],
        dyn_csv      = d_cfg["dyn_csv"],
        dyn_mask_csv = d_cfg["mask_csv"],
        int_csv      = d_cfg["int_csv"],
        static_cols  = d_cfg["static_cols"],
        target_col   = d_cfg["target_col"],
        label_col    = d_cfg["label_col"],
        time_col     = d_cfg["time_col"],
        pid_col      = d_cfg["pid_col"],
        normalize    = True,
        scalers_path = d_cfg["scalers_path"],
        task         = "cls",
    )
    print(f"Total patients: {len(dataset)}")

    # ── Splits ───────────────────────────────────────────────────────────────
    tr_idx, va_idx, te_idx = build_splits(dataset, cfg)
    print(f"Split — train: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)}")

    tr_cfg = cfg["training"]
    train_loader = DataLoader(
        Subset(dataset, tr_idx),
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, va_idx),
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Save test indices for evaluate.py
    os.makedirs(cfg["paths"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"],        exist_ok=True)
    torch.save({"tr_idx": tr_idx, "va_idx": va_idx, "te_idx": te_idx},
               os.path.join(cfg["paths"]["checkpoint_dir"], "splits.pt"))

    # ── Class weights ────────────────────────────────────────────────────────
    tr_labels = [int(dataset.static_df.loc[dataset.pat_ids[i]][d_cfg["label_col"]])
                 for i in tr_idx]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(cfg["model"]["n_classes"]),
        y=tr_labels,
    )
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=(w if tr_cfg["use_class_weights"] else None))

    # ── Model ────────────────────────────────────────────────────────────────
    model = DualStreamSSM.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
    )
    sched_name = tr_cfg.get("lr_scheduler", "cosine").lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tr_cfg["epochs"]
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=tr_cfg.get("lr_patience", 3),
            factor=tr_cfg.get("lr_factor", 0.5),
        )
    elif sched_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown lr_scheduler: {sched_name}")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = math.inf
    history = []
    log_stem = os.path.splitext(os.path.basename(cfg["paths"]["best_model"]))[0]
    log_path = os.path.join(cfg["paths"]["log_dir"], f"{log_stem}_log.json")

    for epoch in range(1, tr_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, tr_cfg["grad_clip"], train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion,
                                    None,      device, tr_cfg["grad_clip"], train=False)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_loss)
            else:
                scheduler.step()
        elapsed = time.time() - t0

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                   va_loss=va_loss, va_acc=va_acc, elapsed=elapsed)
        history.append(row)
        print(f"[{epoch:03d}/{tr_cfg['epochs']}] "
              f"loss: {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc: {tr_acc:.3f}/{va_acc:.3f}  "
              f"({elapsed:.1f}s)")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": va_loss, "val_acc": va_acc},
                       cfg["paths"]["best_model"])
            print(f"  ✓ Saved best model (val_loss={va_loss:.4f})")

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete.  Logs → {log_path}")
    print(f"Best model → {cfg['paths']['best_model']}")


if __name__ == "__main__":
    main()
