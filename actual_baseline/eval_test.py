import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)
import argparse
import os

from model.dataset import YaleDatasetWithMissingnessInfo  # noqa: F401

class CompactConfusionMatrixEncoder(json.JSONEncoder):
    """A custom JSON encoder to format confusion matrices cleanly."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def iterencode(self, o, _one_shot=False):
        list_lvl = 0
        for s in super().iterencode(o, _one_shot=_one_shot):
            if s.startswith('['):
                list_lvl += 1
                yield s
            elif s.endswith(']'):
                list_lvl -= 1
                yield s
            elif isinstance(o, dict) and 'confusion_matrix' in o and list_lvl == 2:
                yield s.replace('\n', '').replace(' ', '')
                if s == ',':
                    yield ' '
            else:
                yield s

def format_custom_json(data):
    """Manually format the JSON to keep the confusion matrix compact."""
    import re
    json_str = json.dumps(data, indent=2)
    
    # We want to find the big expanded array for confusion matrix and flatten the inner lists
    def replacer(match):
        # Flatten by removing newlines and extensive spaces 
        content = match.group(0)
        content = re.sub(r'\s+', ' ', content).replace('[ ', '[').replace(' ]', ']')
        return content
        
    # Match an array of numbers like: [\n      307,\n      5,\n      0,\n      0\n    ]
    formatted_str = re.sub(r'\[\n\s+\d+[\s\d,]*\n\s+\]', replacer, json_str)
    return formatted_str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results_local/bf21/42/classification_target1dim4_ODERNNGRU_trainonly/model.pt", help="Path to the saved model")
    parser.add_argument("--test_data", type=str, default="data/dataset/yale_af21.pt", help="Path to test dataset (.pt)")
    parser.add_argument("--target_index", type=int, default=1, help="Index of the target in y tensor")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--out_json", type=str, default="test_metrics.json", help="Output JSON metrics file")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model}")
    print(f"Loading test data from {args.test_data}")

    model = torch.load(args.model, map_location=DEVICE, weights_only=False)
    test_ds = torch.load(args.test_data, map_location="cpu", weights_only=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model.to(DEVICE).eval()

    all_y, all_pred, all_prob = [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for X, M, t, s, y in loader:
            if y.ndim == 1:
                y = y.unsqueeze(1)

            X, M, t, s, y = X.to(DEVICE), M.to(DEVICE), t.to(DEVICE), s.to(DEVICE), y.to(DEVICE)
            target = y[:, args.target_index].long()

            logits = model(X, M, t, s)
            loss = F.cross_entropy(logits, target)
            total_loss += loss.item()
            n_batches += 1

            prob = F.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)

            all_y.extend(target.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_prob.extend(prob.cpu().numpy().tolist())

    all_y = np.array(all_y)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    metrics = {}
    metrics["loss"] = total_loss / max(n_batches, 1)
    metrics["accuracy"] = float(accuracy_score(all_y, all_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(all_y, all_pred))
    metrics["f1_macro"] = float(f1_score(all_y, all_pred, average="macro"))
    metrics["f1_micro"] = float(f1_score(all_y, all_pred, average="micro"))

    try:
        metrics["auroc"] = float(roc_auc_score(all_y, all_prob, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auroc"] = None

    try:
        # AUPRC macro: average over classes, assuming all_y contains correct target classes
        labels_one_hot = np.zeros((all_y.size, all_prob.shape[1]))
        labels_one_hot[np.arange(all_y.size), all_y] = 1
        metrics["auprc"] = float(average_precision_score(labels_one_hot, all_prob, average="macro"))
    except Exception:
        metrics["auprc"] = None

    cm = confusion_matrix(all_y, all_pred)
    report = classification_report(all_y, all_pred, output_dict=True, zero_division=0)

    print(json.dumps(metrics, indent=2))
    print("Confusion matrix:\n", cm)

    model_dir = os.path.dirname(args.model)
    out_json_path = os.path.join(model_dir, args.out_json)

    final_data = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(out_json_path, "w") as f:
        f.write(format_custom_json(final_data))

    print(f"Saved metrics to {out_json_path}")

if __name__ == "__main__":
    main()
