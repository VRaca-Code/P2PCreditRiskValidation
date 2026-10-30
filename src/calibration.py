"""Calibration tables and post-model recalibration.

What this file does
-------------------
A credit model is calibrated when a predicted 10% PD really corresponds to about 10% of such
loans defaulting. This file:

* ``calibration_table`` bins predictions and returns mean predicted PD vs observed default rate
  (the data behind a reliability curve);
* ``recalibrate`` fits a sigmoid (Platt) or isotonic calibrator around a whole pipeline using
  internal cross-validation, refitting the preprocessor in each fold so nothing leaks;
* ``compare_calibration`` reports ROC-AUC, Brier and ECE for each model before and after
  recalibration, which is how the effect of class re-weighting is measured and repaired.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from .metrics import expected_calibration_error


def calibration_table(y_true, probabilities, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy=strategy)
    return pd.DataFrame({"mean_predicted_pd": predicted, "observed_default_rate": observed})


def recalibrate(pipeline, X_train, y_train, method: str = "isotonic", cv: int = 3):
    """Fit a calibrated copy of the whole pipeline.

    The preprocessor is refit inside every internal fold, so calibration never sees
    held-out rows during preprocessing.
    """
    calibrated = CalibratedClassifierCV(estimator=clone(pipeline), method=method, cv=cv)
    return calibrated.fit(X_train, y_train)


def compare_calibration(pipelines: dict, X_train, y_train, X_test, y_test,
                        methods=("sigmoid", "isotonic"), cv: int = 3) -> pd.DataFrame:
    """Test-set ROC-AUC / Brier / ECE for each model before ("none") and after recalibration."""
    rows = []
    for name, pipe in pipelines.items():
        variants = {"none": pipe.predict_proba(X_test)[:, 1]}
        for method in methods:
            variants[method] = recalibrate(pipe, X_train, y_train, method, cv).predict_proba(X_test)[:, 1]
        for method, p in variants.items():
            rows.append({
                "Model": name, "Method": method,
                "ROC_AUC": roc_auc_score(y_test, p),
                "Brier": brier_score_loss(y_test, p),
                "ECE": expected_calibration_error(y_test, p),
            })
    return pd.DataFrame(rows)
