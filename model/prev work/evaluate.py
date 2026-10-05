"""
model/evaluate.py
═════════════════
تقييم شامل للموديل بعد التدريب.

Metrics المحسوبة:

  Exercise Classification:
    • Accuracy (overall)
    • F1-score per class + macro average
    • Confusion matrix

  Rep State Classification:
    • Accuracy
    • F1 per state (ready/down/up)
    • Confusion matrix

  Form Score Regression:
    • MAE  (Mean Absolute Error)
    • RMSE (Root Mean Square Error)
    • Pearson correlation coefficient

  Per-exercise breakdown:
    • State accuracy per exercise
    • Form MAE per exercise

Usage:
    python model/evaluate.py --ckpt model/checkpoints/best.pth --data-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
#  Core evaluation function
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_split(
    model:   torch.nn.Module,
    loader:  DataLoader,
    device:  str,
    split:   str = "val",
    verbose: bool = True,
) -> Dict:
    """
    Run the model on a DataLoader split and compute all metrics.

    Returns dict with all computed metrics.
    """
    from exercises import Exercise

    model.eval()
    all_ex_pred,    all_ex_true    = [], []
    all_state_pred, all_state_true = [], []
    all_form_pred,  all_form_true  = [], []

    with torch.no_grad():
        for X, y_ex, y_state, y_form in loader:
            X       = X.to(device)
            y_ex    = y_ex.to(device)
            y_state = y_state.to(device)
            y_form  = y_form.to(device)

            ex_logits, st_logits, form_sc = model(X)

            all_ex_pred.extend(ex_logits.argmax(1).cpu().numpy())
            all_ex_true.extend(y_ex.cpu().numpy())
            all_state_pred.extend(st_logits.argmax(1).cpu().numpy())
            all_state_true.extend(y_state.cpu().numpy())
            all_form_pred.extend(form_sc.cpu().numpy())
            all_form_true.extend(y_form.cpu().numpy())

    ex_pred    = np.array(all_ex_pred)
    ex_true    = np.array(all_ex_true)
    st_pred    = np.array(all_state_pred)
    st_true    = np.array(all_state_true)
    form_pred  = np.array(all_form_pred)
    form_true  = np.array(all_form_true)

    N = len(ex_pred)
    metrics = {"split": split, "n_samples": N}

    # ── Exercise metrics ──────────────────────────────────────────────────────
    ex_acc = float((ex_pred == ex_true).mean())
    metrics["exercise_accuracy"] = ex_acc

    ex_names  = [e.value for e in Exercise]
    n_classes = len(ex_names)

    # Confusion matrix
    ex_cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(ex_true, ex_pred):
        ex_cm[t, p] += 1

    # Per-class F1
    ex_f1 = []
    for i in range(n_classes):
        tp = ex_cm[i, i]
        fp = ex_cm[:, i].sum() - tp
        fn = ex_cm[i, :].sum() - tp
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        ex_f1.append(f1)

    metrics["exercise_f1_macro"] = float(np.mean(ex_f1))
    metrics["exercise_f1_per_class"] = {ex_names[i]: round(ex_f1[i], 4) for i in range(n_classes)}
    metrics["exercise_confusion_matrix"] = ex_cm.tolist()

    # ── Rep state metrics ─────────────────────────────────────────────────────
    st_acc   = float((st_pred == st_true).mean())
    st_names = ["ready", "down", "up"]
    st_cm    = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(st_true, st_pred):
        st_cm[t, p] += 1

    st_f1 = []
    for i in range(3):
        tp = st_cm[i, i]
        fp = st_cm[:, i].sum() - tp
        fn = st_cm[i, :].sum() - tp
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        st_f1.append(f1)

    metrics["state_accuracy"]          = st_acc
    metrics["state_f1_macro"]          = float(np.mean(st_f1))
    metrics["state_f1_per_class"]      = {st_names[i]: round(st_f1[i], 4) for i in range(3)}
    metrics["state_confusion_matrix"]  = st_cm.tolist()

    # ── Form score metrics ────────────────────────────────────────────────────
    mae  = float(np.abs(form_pred - form_true).mean())
    rmse = float(np.sqrt(((form_pred - form_true) ** 2).mean()))

    # Pearson correlation
    if form_true.std() > 1e-6 and form_pred.std() > 1e-6:
        corr = float(np.corrcoef(form_pred, form_true)[0, 1])
    else:
        corr = 0.0

    metrics["form_mae"]         = mae
    metrics["form_rmse"]        = rmse
    metrics["form_correlation"] = corr

    # ── Per-exercise state accuracy ───────────────────────────────────────────
    per_ex_state_acc = {}
    for i, name in enumerate(ex_names):
        mask = ex_true == i
        if mask.sum() > 0:
            acc  = float((st_pred[mask] == st_true[mask]).mean())
            fmae = float(np.abs(form_pred[mask] - form_true[mask]).mean())
            per_ex_state_acc[name] = {"state_acc": round(acc,4), "form_mae": round(fmae,4)}

    metrics["per_exercise"] = per_ex_state_acc

    # ── Print report ─────────────────────────────────────────────────────────
    if verbose:
        _print_report(metrics, ex_names, ex_cm, st_names, st_cm)

    return metrics


def _print_report(metrics, ex_names, ex_cm, st_names, st_cm) -> None:
    sep = "═" * 62

    print(f"\n{sep}")
    print(f"  EVALUATION REPORT  —  split={metrics['split']}  N={metrics['n_samples']}")
    print(f"{sep}")

    print(f"\n  ── Exercise Classification ──")
    print(f"    Accuracy  : {metrics['exercise_accuracy']*100:.2f}%")
    print(f"    F1 Macro  : {metrics['exercise_f1_macro']:.4f}")
    print(f"    Per class:")
    for name, f1 in metrics["exercise_f1_per_class"].items():
        bar = "█" * int(f1 * 20)
        print(f"      {name:<20s}: F1={f1:.4f}  {bar}")

    print(f"\n  ── Rep State ──")
    print(f"    Accuracy  : {metrics['state_accuracy']*100:.2f}%")
    print(f"    F1 Macro  : {metrics['state_f1_macro']:.4f}")
    for name, f1 in metrics["state_f1_per_class"].items():
        print(f"      {name:<10s}: F1={f1:.4f}")

    print(f"\n  ── Form Score ──")
    print(f"    MAE       : {metrics['form_mae']:.4f}")
    print(f"    RMSE      : {metrics['form_rmse']:.4f}")
    print(f"    Pearson r : {metrics['form_correlation']:.4f}")

    print(f"\n  ── Per Exercise ──")
    for ex, stats in metrics["per_exercise"].items():
        print(f"    {ex:<22s}: state_acc={stats['state_acc']:.3f}  form_mae={stats['form_mae']:.3f}")

    print(f"\n{sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI — evaluate a saved checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FitnessModel checkpoint")
    parser.add_argument("--ckpt",     required=True,          help="Path to best.pth")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split",    default="test",         choices=["train","val","test"])
    parser.add_argument("--batch",    type=int, default=128)
    parser.add_argument("--workers",  type=int, default=4)
    parser.add_argument("--save",     default=None,           help="Save metrics JSON to path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] Device: {device}")

    # Load model
    from model.architecture import build_model
    ckpt = torch.load(args.ckpt, map_location=device)
    train_args = ckpt.get("args", {})

    model = build_model(
        hidden_size = train_args.get("hidden", 128),
        n_layers    = train_args.get("layers", 2),
        dropout     = 0.0,     # no dropout in eval
        device      = device,
    )
    model.load_state_dict(ckpt["model_state"])
    print(f"[Eval] Loaded checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch','?')})")

    # Load data
    from model.dataset import make_dataloaders
    loaders = make_dataloaders(args.data_dir, args.batch, args.workers, balanced=False)

    if args.split not in loaders:
        print(f"[ERROR] Split '{args.split}' not found"); sys.exit(1)

    metrics = evaluate_split(model, loaders[args.split], device, split=args.split)

    if args.save:
        with open(args.save, "w") as f:
            # Convert numpy arrays to lists for JSON
            def _clean(obj):
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, dict): return {k: _clean(v) for k,v in obj.items()}
                if isinstance(obj, list): return [_clean(v) for v in obj]
                return obj
            json.dump(_clean(metrics), f, indent=2)
        print(f"[Eval] Metrics saved → {args.save}")


if __name__ == "__main__":
    main()
