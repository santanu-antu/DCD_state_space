"""
evaluate.py
-----------
Unified evaluation script for:
  1) Test fraction evaluation (mode: split) - default
  2) Full cohort evaluation/transfer (mode: full)

Usage
-----
python evaluate.py --config config.yaml [--mode split]
python evaluate.py --config train_cfg.yaml --test_config test_cfg.yaml --out_dir ./checkpoints/ --mode full
"""

import argparse
import json
import os
import torch
import numpy as np
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, \
    balanced_accuracy_score, roc_auc_score, accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from functools import partial

from model.dual_stream_ssm import DualStreamSSM
from preprocessing.dataset import ICUStreamsDataset
from preprocessing.collate import collate_fn

def collect_predictions(model, loader, device, criterion=None):
    model.eval()
    all_logits, all_labels = [], []
    all_losses = []

    with torch.no_grad():
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
            
            if criterion is not None:
                loss_vals = criterion(logits, y_cls)
                all_losses.extend(loss_vals.cpu().tolist())

            all_logits.append(logits.cpu())
            all_labels.append(y_cls.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return logits, labels, all_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Config used for training")
    parser.add_argument("--test_data_dir", default=None, help="Directory containing test CSVs (overrides config data paths)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (overrides config)")
    parser.add_argument("--mode", choices=["split", "full"], default="split", help="split: evaluate on test_idx, full: evaluate on all data")
    parser.add_argument("--seed", type=int, default=None, help="Override split seed (for loading correct folder)")
    parser.add_argument("--out_dir", type=str, default=None, help="Optional directory to load scalers and checkpoint and save results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.test_data_dir:
        for k in ["static_csv", "dyn_csv", "mask_csv", "int_csv"]:
            if k in cfg["data"]:
                filename = os.path.basename(cfg["data"][k])
                cfg["data"][k] = os.path.join(args.test_data_dir, filename)

    tcfg = cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed if args.seed is not None else cfg.get("split", {}).get("seed", 42)
    
    ckpt_path = args.checkpoint or cfg.get("paths", {}).get("best_model", "best_model.pt")
    if args.out_dir and args.checkpoint is None:
        # Load from the seed folder inside out_dir
        actual_out_dir = os.path.join(args.out_dir, str(seed))
        ckpt_path = os.path.join(actual_out_dir, "best_model.pt")
    print(f"Loading checkpoint: {ckpt_path}")

    d_cfg = tcfg["data"]
    scalers_path = d_cfg.get("scalers_path", "scalers.pkl")
    if args.out_dir:
        actual_out_dir = os.path.join(args.out_dir, str(seed))
        scalers_path = os.path.join(actual_out_dir, "scalers.pkl")

    dataset = ICUStreamsDataset(
        static_csv=d_cfg["static_csv"], dyn_csv=d_cfg["dyn_csv"], dyn_mask_csv=d_cfg["mask_csv"], int_csv=d_cfg["int_csv"],
        static_cols=d_cfg.get("static_cols", []), target_col=d_cfg["target_col"], label_col=d_cfg["label_col"],
        time_col=d_cfg["time_col"], pid_col=d_cfg["pid_col"], normalize=True, scalers_path=scalers_path,
        task="cls", label_scheme=d_cfg.get("label_scheme", None)
    )

    if args.mode == "split":
        splits_path = os.path.join(cfg.get("paths", {}).get("checkpoint_dir", "."), "splits.pt")
        splits = torch.load(splits_path, weights_only=False)
        te_idx = splits["te_idx"]
    else:
        te_idx = list(range(len(dataset)))
        
    print(f"Evaluation mode: {args.mode.upper()} | Test set size: {len(te_idx)}")

    max_seq_len = tcfg.get("data", {}).get("max_seq_len", None)
    _collate = partial(collate_fn, max_seq_len=max_seq_len) if max_seq_len else collate_fn
    test_loader = DataLoader(
        Subset(dataset, te_idx), batch_size=tcfg["training"]["batch_size"], shuffle=False,
        collate_fn=_collate, num_workers=2, pin_memory=(device.type == "cuda")
    )

    model = DualStreamSSM.from_config(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    criterion = nn.CrossEntropyLoss(reduction="none")
    logits, labels, unweighted_losses = collect_predictions(model, test_loader, device, criterion)
    
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    labels = labels.numpy()
    mean_loss = float(np.mean(unweighted_losses)) if unweighted_losses else 0.0

    n_classes = cfg["model"]["n_classes"]
    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    f1_mac = classification_report(labels, preds, output_dict=True, zero_division=0).get("macro avg", {}).get("f1-score", 0.0)

    try:
        auroc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auroc = float("nan")

    try:
        labels_one_hot = np.zeros((labels.size, n_classes))
        labels_one_hot[np.arange(labels.size), labels] = 1
        auprc = float(average_precision_score(labels_one_hot, probs, average="macro"))
    except ValueError:
        auprc = float("nan")

    cm = confusion_matrix(labels, preds, labels=range(n_classes))
    class_names = dataset.label_names if dataset.label_names else [str(x) for x in range(n_classes)]
    
    report_dict = classification_report(labels, preds, target_names=class_names, labels=range(n_classes), output_dict=True, zero_division=0)
    report_str = classification_report(labels, preds, target_names=class_names, labels=range(n_classes), zero_division=0)

    print("\n" + "="*60)
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  Balanced Accuracy  : {bacc:.4f}")
    print(f"  F1 Macro           : {f1_mac:.4f}")
    print(f"  Macro AUROC        : {auroc:.4f}")
    print(f"  Macro AUPRC        : {auprc:.4f}")
    print(f"  Unweighted Loss    : {mean_loss:.4f}")
    print("\n  Confusion Matrix (rows=true, cols=pred):")
    print("  " + "  ".join(f"{n:>7}" for n in class_names))
    for i, row in enumerate(cm):
        print(f"  {class_names[i]:>7}  " + "  ".join(f"{v:>7}" for v in row))
    print("\n  Classification Report:")
    print(report_str)
    print("="*60)

    results = {
        "metrics": {
            "loss": mean_loss,
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "f1_macro": float(f1_mac),
            "auroc": float(auroc),
            "auprc": float(auprc)
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    
    res_path = cfg.get("paths", {}).get("results", "results.json")
    if args.out_dir:
        actual_out_dir = os.path.join(args.out_dir, str(seed))
        res_path = os.path.join(actual_out_dir, "test_metrics.json")
    
    os.makedirs(os.path.dirname(res_path) or ".", exist_ok=True)

    import re
    
    # Serialize with standard indentation
    json_str = json.dumps(results, indent=2)
    
    # regex to collapse 1D arrays of numbers into a single line (for confusion matrix)
    def collapse_array(match):
        return re.sub(r'\s+', ' ', match.group(0)).replace('[ ', '[').replace(' ]', ']')
        
    json_str = re.sub(r'\[\s*(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*,\s*)*-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*\]', collapse_array, json_str)
    
    with open(res_path, "w") as f:
        f.write(json_str)

    print(f"\nResults saved -> {res_path}")

if __name__ == "__main__":
    main()
