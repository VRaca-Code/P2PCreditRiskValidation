"""Discrimination, calibration and threshold metrics.

What this file does
-------------------
Computes every number used to judge a credit model from true labels and predicted
default probabilities (PDs):

* ranking quality - ROC-AUC, PR-AUC (preferred for rare defaults) and the KS statistic
* probability quality - Brier score and Expected Calibration Error (``expected_calibration_error``)
* decision quality at a cut-off - accuracy, precision, recall, F1 (``threshold_metrics``) and a
  whole table across thresholds with the share of loans approved (``threshold_table``)
* ``evaluate`` bundles the above for one model; ``model_metrics`` adds train ROC-AUC and the
  train-test generalization gap.

Used by: experiments.py, calibration.py, tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true, probabilities) -> float:
    """Kolmogorov-Smirnov separation between defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Equal-width ECE: bin-weighted mean |observed rate - mean predicted PD|."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, edges[1:-1], right=True)  # 0 .. n_bins-1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - p[mask].mean())
    return float(ece)


def threshold_metrics(y_true, probabilities, threshold: float) -> dict:
    p = np.asarray(probabilities)
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def threshold_table(y_true, probabilities, thresholds=None) -> pd.DataFrame:
    """Metrics across decision thresholds; `approval_rate` = share predicted to repay."""
    p = np.asarray(probabilities)
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for t in thresholds:
        row = threshold_metrics(y_true, p, t)
        row["approval_rate"] = float(np.mean(p < t))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(y_true, probabilities, threshold: float = 0.5) -> dict:
    """Threshold-free ranking and calibration metrics plus threshold metrics."""
    p = np.asarray(probabilities)
    out = {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "ks": ks_statistic(y_true, p),
        "brier": brier_score_loss(y_true, p),
        "ece": expected_calibration_error(y_true, p),
    }
    out.update(threshold_metrics(y_true, p, threshold))
    return out


def model_metrics(
    name: str,
    y_train, train_prob,
    y_test, test_prob,
    train_time: float = float("nan"),
    predict_time: float = float("nan"),
    threshold: float = 0.5,
) -> dict:
    """One results row: test metrics, train ROC-AUC and the generalization gap."""
    test = evaluate(y_test, test_prob, threshold)
    train_auc = roc_auc_score(y_train, train_prob)
    row = {"Model": name, "Train_ROC_AUC": train_auc, "Test_ROC_AUC": test["roc_auc"],
           "Generalization_Gap": train_auc - test["roc_auc"]}
    row.update({
        "PR_AUC": test["pr_auc"], "KS": test["ks"], "Brier": test["brier"], "ECE": test["ece"],
        "Accuracy": test["accuracy"], "Balanced_Accuracy": test["balanced_accuracy"],
        "Precision": test["precision"], "Recall": test["recall"], "F1": test["f1"],
        "Threshold": test["threshold"],
        "Train_Time_Seconds": train_time, "Predict_Time_Seconds": predict_time,
    })
    return row
