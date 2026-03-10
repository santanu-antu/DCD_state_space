"""
train_baseline.py
-----------------
Self-contained training + evaluation script for the baseline ODE-RNN model
(prev_model.py / RNNModel with model_type='ODERNN').

Key differences from the dual-stream model:
  - Vitals and medications are combined into one 25-feature input X
  - No separation of intervention stream
  - Standard GRU-based ODE-RNN without Mamba conditioning

Usage (from DCD_state_space/ root):
    python baseline_model/train_baseline.py [--epochs 15] [--hidden_dim 128]

All artefacts (checkpoints, logs, results) are written to baseline_model/.
"""

import argparse
import json
import math
import os
import random
import sys
import time as time_mod

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# ── Paths ────────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.dirname(HERE)
# Allow importing prev_model from baseline_model/
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from prev_model import RNNModel                    # noqa: E402
from baseline_dataset import (BaselineDataset,    # noqa: E402
                               collate_baseline,
                               fit_scalers)

# ──────────────────────────────────────────────────────────────────────────────
DATA = dict(
    dyn_csv    = os.path.join(HERE, "dynamic.csv"),
    miss_csv   = os.path.join(HERE, "missing.csv"),
    static_csv = os.path.join(HERE, "static_target.csv"),
)

SEED        = 42
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
# TEST_FRAC = 0.15 (remainder)

CLASS_NAMES = ["<30h", "30-59h", "60-89h", ">=90h"]

# ──────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_splits(labels, seed=SEED):
    n = len(labels)
    indices = list(range(n))
    tr_val, te, tr_val_lbl, _ = train_test_split(
        indices, labels,
        test_size=1.0 - TRAIN_FRAC - VAL_FRAC,
        stratify=labels, random_state=seed,
    )
    val_frac_corr = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    tr, va = train_test_split(
        tr_val, test_size=val_frac_corr,
        stratify=tr_val_lbl, random_state=seed,
    )
    return tr, va, te


def run_epoch(model, loader, criterion, optimizer, device, grad_clip, train):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            X     = batch["X"].to(device)       # (B, T, F)
            M     = batch["M"].to(device)       # (B, T, F)
            t     = batch["t"].to(device)       # (B, T, 1)
            s     = batch["s"].to(device)       # (B, n_static)
            y_cls = batch["y_cls"].to(device)   # (B,)

            logits = model(X, M, t, s)          # (B, n_classes)
            loss   = criterion(logits, y_cls)

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


# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, n_classes):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            X     = batch["X"].to(device)
            M     = batch["M"].to(device)
            t     = batch["t"].to(device)
            s     = batch["s"].to(device)
            y_cls = batch["y_cls"]
            logits = model(X, M, t, s).cpu()
            all_logits.append(logits)
            all_labels.append(y_cls)

    logits = torch.cat(all_logits)          # (N, C)
    labels = torch.cat(all_labels)          # (N,)
    probs  = torch.softmax(logits, dim=1).numpy()
    preds  = logits.argmax(1).numpy()
    y_true = labels.numpy()

    acc     = (preds == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, preds)
    auroc   = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    cm      = confusion_matrix(y_true, preds)
    report  = classification_report(y_true, preds,
                                     target_names=CLASS_NAMES, digits=4)
    return dict(acc=acc, bal_acc=bal_acc, auroc=auroc, cm=cm.tolist(),
                report=report)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--hidden_dim", type=int,   default=128)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--grad_clip",  type=float, default=1.0)
    parser.add_argument("--max_seq_len",type=int,   default=300)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset (no scalers yet — fitted after split) ─────────────────────
    dataset = BaselineDataset(
        dyn_csv    = DATA["dyn_csv"],
        miss_csv   = DATA["miss_csv"],
        static_csv = DATA["static_csv"],
        max_seq_len = args.max_seq_len,
        scalers     = None,
    )
    print(f"Total patients: {len(dataset)}, features: {len(dataset.feat_cols)}")

    # ── Split ────────────────────────────────────────────────────────────
    tr_idx, va_idx, te_idx = build_splits(dataset.labels.tolist())
    print(f"Split — train: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)}")

    # ── Fit scalers on training patients, apply to whole dataset ─────────
    scalers = fit_scalers(dataset, tr_idx)
    dataset.scalers = scalers
    print("Scalers fitted on training patients only.")

    # ── DataLoaders ──────────────────────────────────────────────────────
    make_loader = lambda idx, shuffle: DataLoader(
        Subset(dataset, idx),
        batch_size  = args.batch_size,
        shuffle     = shuffle,
        collate_fn  = collate_baseline,
        num_workers = 2,
        pin_memory  = (device.type == "cuda"),
    )
    train_loader = make_loader(tr_idx, shuffle=True)
    val_loader   = make_loader(va_idx, shuffle=False)
    test_loader  = make_loader(te_idx, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    n_feat    = len(dataset.feat_cols)   # 25
    n_static  = 5
    n_classes = 4

    model = RNNModel(
        s_dim      = n_static,
        X_feat_dim = n_feat,
        hidden_dim = args.hidden_dim,
        output_dim = n_classes,
        model_type = "ODERNN",
        rnn_type   = "GRU",
        # normalisation is handled in the dataset; pass no-ops to the model
        mean_dyn   = 0.0,
        std_dyn    = 1.0,
        mean_stat  = 0.0,
        std_stat   = 1.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()   # no class weights

    # ── Training loop ─────────────────────────────────────────────────────
    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"),     exist_ok=True)
    os.makedirs(os.path.join(HERE, "logs"),        exist_ok=True)

    ckpt_path = os.path.join(HERE, "checkpoints", "best_baseline.pt")
    log_path  = os.path.join(HERE, "logs", "train_baseline.json")

    best_val_loss = math.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time_mod.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion,
                                    optimizer, device, args.grad_clip, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion,
                                    None, device, args.grad_clip, train=False)
        scheduler.step()
        elapsed = time_mod.time() - t0

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc,
                   va_loss=va_loss, va_acc=va_acc, elapsed=elapsed)
        history.append(row)
        print(f"[{epoch:03d}/{args.epochs}] "
              f"loss: {tr_loss:.4f}/{va_loss:.4f}  "
              f"acc: {tr_acc:.3f}/{va_acc:.3f}  "
              f"({elapsed:.1f}s)")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_loss": va_loss, "val_acc": va_acc}, ckpt_path)
            print(f"  ✓ Saved best model (val_loss={va_loss:.4f})")

    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best checkpoint → {ckpt_path}")

    # ── Save split indices ────────────────────────────────────────────────
    torch.save({"tr_idx": tr_idx, "va_idx": va_idx, "te_idx": te_idx},
               os.path.join(HERE, "checkpoints", "splits.pt"))

    # ── Final evaluation on test set ──────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\nLoaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    metrics = evaluate(model, test_loader, device, n_classes)

    print(f"\n{'='*60}")
    print(f"  Test Accuracy        : {metrics['acc']:.4f}")
    print(f"  Balanced Accuracy    : {metrics['bal_acc']:.4f}")
    print(f"  Macro AUROC          : {metrics['auroc']:.4f}")
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    header = "  ".join(f"{n:>8}" for n in CLASS_NAMES)
    print(f"  {'':>8}  {header}")
    for i, row in enumerate(metrics["cm"]):
        row_str = "  ".join(f"{v:>8}" for v in row)
        print(f"  {CLASS_NAMES[i]:>8}  {row_str}")
    print(f"\n  Classification Report:\n{metrics['report']}")
    print(f"{'='*60}\n")

    # Save results
    results_path = os.path.join(HERE, "results", "test_metrics_baseline.json")
    save_metrics  = {k: v for k, v in metrics.items() if k != "cm"}
    save_metrics["cm"] = metrics["cm"]
    save_metrics["acc"]      = float(metrics["acc"])
    save_metrics["bal_acc"]  = float(metrics["bal_acc"])
    save_metrics["auroc"]    = float(metrics["auroc"])
    with open(results_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
