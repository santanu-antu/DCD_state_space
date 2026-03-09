"""
evaluate.py
-----------
Load the best checkpoint and evaluate on the held-out test set.

Metrics
~~~~~~~
  - Overall accuracy
  - Balanced accuracy
  - Macro-averaged AUROC (one-vs-rest, using softmax probabilities)
  - Per-class precision, recall, F1 (sklearn classification_report)
  - Confusion matrix

Usage
-----
python evaluate.py                          # uses config.yaml + checkpoints/best_model.pt
python evaluate.py --config path/to/cfg.yaml --checkpoint path/to/model.pt
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

from sklearn.utils.class_weight import compute_class_weight
from preprocessing.collate  import collate_fn
from functools import partial
from preprocessing.dataset  import ICUStreamsDataset
from model.dual_stream_ssm  import DualStreamSSM


# ─────────────────────────────────────────────────────────────────────────────
def collect_predictions(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            S        = batch["S"].to(device)
            t_dyn    = batch["t_dyn"].to(device)
            Y_dyn    = batch["Y_dyn"].to(device)
            M_dyn    = batch["M_dyn"].to(device)
            t_int    = batch["t_int"].to(device)
            U_int    = batch["U_int"].to(device)
            dyn_lens = batch["dyn_lens"].to(device)
            int_lens = batch["int_lens"].to(device)
            y_cls      = batch["y_cls"]

            logits = model(S, t_dyn, Y_dyn, M_dyn, t_int, U_int,
                           dyn_lens, int_lens)

            all_logits.append(logits.cpu())
            all_labels.append(y_cls)

    logits = torch.cat(all_logits, dim=0)    # (N, n_classes)
    labels = torch.cat(all_labels, dim=0)    # (N,)
    return logits, labels


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.checkpoint or cfg["paths"]["best_model"]
    print(f"Loading checkpoint: {ckpt_path}")

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
        label_scheme = d_cfg.get("label_scheme", None),
    )

    # Load the same split used during training
    splits_path = os.path.join(cfg["paths"]["checkpoint_dir"], "splits.pt")
    splits = torch.load(splits_path, weights_only=False)
    te_idx  = splits["te_idx"]
    tr_idx  = splits["tr_idx"]
    print(f"Test set size: {len(te_idx)}")

    # ── Class weights (always computed from training set) ─────────────────────
    from preprocessing.dataset import LABEL_SCHEMES
    n_classes   = cfg["model"]["n_classes"]
    class_names = (dataset.label_names
                   if dataset.label_names is not None
                   else [str(i) for i in range(n_classes)])
    all_labels  = [int(dataset.labels[i]) for i in range(len(dataset))]
    tr_labels   = [int(dataset.labels[i]) for i in tr_idx]
    te_labels   = [int(dataset.labels[i]) for i in te_idx]

    cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=tr_labels)
    w  = torch.tensor(cw, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)

    # ── Print class distributions ─────────────────────────────────────────────
    from collections import Counter
    total_counts = Counter(all_labels)
    test_counts  = Counter(te_labels)
    print("\nClass distribution (whole dataset vs test set):")
    print(f"  {'Class':<12} {'Dataset':>10} {'Dataset%':>10} {'Test':>8} {'Test%':>8}")
    for c, name in enumerate(class_names):
        tot = total_counts[c]; tot_pct = 100 * tot / len(all_labels)
        tst = test_counts[c];  tst_pct = 100 * tst / len(te_idx)
        print(f"  {name:<12} {tot:>10}  {tot_pct:>8.1f}%  {tst:>6}  {tst_pct:>8.1f}%")

    max_seq_len = cfg.get("data", {}).get("max_seq_len", None)
    _collate = partial(collate_fn, max_seq_len=max_seq_len) if max_seq_len else collate_fn
    test_loader = DataLoader(
        Subset(dataset, te_idx),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        collate_fn=_collate,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = DualStreamSSM.from_config(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',float('nan')):.4f}")

    # ── Predictions ──────────────────────────────────────────────────────────
    logits, labels = collect_predictions(model, test_loader, device)
    probs  = torch.softmax(logits, dim=-1).numpy()
    preds  = logits.argmax(dim=-1).numpy()
    labels = labels.numpy()

    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    logits_t = logits.to(device)
    weighted_loss   = nn.CrossEntropyLoss(weight=w)(logits_t, labels_t).item()
    unweighted_loss = nn.CrossEntropyLoss()(logits_t, labels_t).item()

    acc      = (preds == labels).mean()
    bal_acc  = balanced_accuracy_score(labels, preds)
    auroc    = roc_auc_score(labels, probs,
                             multi_class="ovr", average="macro")
    cm       = confusion_matrix(labels, preds)
    report   = classification_report(
        labels, preds, target_names=class_names, digits=4
    )

    print("\n" + "="*60)
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Balanced Accuracy  : {bal_acc:.4f}")
    print(f"  Macro AUROC        : {auroc:.4f}")
    print(f"  Weighted test loss : {weighted_loss:.4f}")
    print(f"  Unweighted test loss: {unweighted_loss:.4f}")
    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{n:>7}" for n in class_names))
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>7}  " + "  ".join(f"{v:>7}" for v in row))
    print("\n  Classification Report:")
    print(report)
    print("="*60)

    # ── Save ─────────────────────────────────────────────────────────────────
    results = {
        "accuracy":            float(acc),
        "balanced_accuracy":   float(bal_acc),
        "macro_auroc":         float(auroc),
        "weighted_test_loss":  float(weighted_loss),
        "unweighted_test_loss": float(unweighted_loss),
        "confusion_matrix":    cm.tolist(),
        "classification_report": classification_report(
            labels, preds, target_names=class_names, output_dict=True
        ),
    }
    os.makedirs(os.path.dirname(cfg["paths"]["results"]) or ".", exist_ok=True)
    with open(cfg["paths"]["results"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {cfg['paths']['results']}")


if __name__ == "__main__":
    main()
