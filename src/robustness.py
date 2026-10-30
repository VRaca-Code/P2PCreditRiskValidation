"""Stability and stress checks: PSI, cross-validation, sensitivity, stress, chronological backtest.

What this file does
-------------------
Tests whether a model keeps working when conditions change:

* ``calculate_psi`` / ``feature_psi`` - Population Stability Index: how far each feature's
  distribution moved between training and test (handles binary / low-cardinality features)
* ``cv_scores`` - stratified cross-validation on training data only
* ``sensitivity_test`` - mean predicted PD when one feature is scaled up or down
* ``stress_test`` - mean PD under combined shocks (lower income, higher DTI and utilisation);
  economically sensible models should predict higher PD as stress grows
* ``chronological_backtest`` - expanding-window test: train on years before Y, test on year Y
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import DATE_COL, RANDOM_STATE, TARGET


def calculate_psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index of `actual` against quantile bins of `expected`."""
    expected = pd.Series(expected).replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.Series(actual).replace([np.inf, -np.inf], np.nan).dropna()
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    if expected.nunique() <= bins:
        # Binary / low-cardinality features have no usable quantile cuts: compare value shares.
        values = expected.unique()
        e = expected.value_counts(normalize=True).reindex(values, fill_value=0.0)
        a = actual.value_counts(normalize=True).reindex(values, fill_value=0.0)
        unseen = 1.0 - a.sum()  # actual values never seen in `expected` count as one extra bin
        e, a = np.append(e.to_numpy(), 0.0), np.append(a.to_numpy(), max(unseen, 0.0))
    else:
        cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        cuts[0], cuts[-1] = -np.inf, np.inf
        e = pd.cut(expected, bins=cuts, include_lowest=True).value_counts(sort=False, normalize=True).to_numpy()
        a = pd.cut(actual, bins=cuts, include_lowest=True).value_counts(sort=False, normalize=True).to_numpy()
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """PSI per numeric feature. With a chronological split this measures real drift."""
    cols = X_train.select_dtypes(include="number").columns
    rows = [{"Feature": c, "PSI": calculate_psi(X_train[c], X_test[c])} for c in cols]
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def cv_scores(pipeline, X, y, n_splits: int = 3, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Stratified CV on TRAINING data only; the preprocessor is refit in every fold."""
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    rows = []
    for fold, (tr, va) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y), 1):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        rows.append({"Fold": fold, "ROC_AUC": roc_auc_score(y.iloc[va], p),
                     "PR_AUC": average_precision_score(y.iloc[va], p)})
    return pd.DataFrame(rows)


def sensitivity_test(predict_fn: Callable[[pd.DataFrame], np.ndarray], records: pd.DataFrame,
                     feature: str, multipliers=(0.8, 0.9, 1.1, 1.2)) -> pd.DataFrame:
    """Mean predicted PD when one numeric feature is scaled; `predict_fn(records) -> PDs`."""
    baseline = float(np.mean(predict_fn(records)))
    rows = []
    for m in multipliers:
        shocked = records.copy()
        shocked[feature] = shocked[feature] * m
        pd_m = float(np.mean(predict_fn(shocked)))
        rows.append({"Feature": feature, "Multiplier": m, "Baseline_PD": baseline,
                     "Scenario_PD": pd_m, "Absolute_Change": pd_m - baseline,
                     "Relative_Change": pd_m / baseline - 1 if baseline > 0 else np.nan})
    return pd.DataFrame(rows)


DEFAULT_STRESS = {
    "Baseline": {},
    "Moderate stress": {"annual_inc": 0.90, "dti": 1.15, "revol_util": 1.10},
    "Severe stress": {"annual_inc": 0.80, "dti": 1.25, "revol_util": 1.20},
}


def stress_test(predict_fn: Callable[[pd.DataFrame], np.ndarray], records: pd.DataFrame,
                scenarios: dict | None = None) -> pd.DataFrame:
    """Mean PD under multiplicative shocks. Scenarios are illustrative, not macro-calibrated."""
    rows = []
    for name, shocks in (scenarios or DEFAULT_STRESS).items():
        data = records.copy()
        for feature, m in shocks.items():
            data[feature] = data[feature] * m
        rows.append({"Scenario": name, "Average_PD": float(np.mean(predict_fn(data)))})
    return pd.DataFrame(rows)


def chronological_backtest(make_pipeline_fn: Callable[[pd.DataFrame, pd.Series], object],
                           frame: pd.DataFrame, first_test_year: int = 2014,
                           min_train: int = 3_000, min_test: int = 500,
                           max_train: int | None = 100_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Expanding-window backtest: for each year Y train on years < Y, test on year Y.

    `make_pipeline_fn(X_train, y_train)` returns an UNFITTED pipeline.
    """
    years = frame[DATE_COL].dt.year
    rows = []
    for year in sorted(years.unique()):
        if year < first_test_year:
            continue
        train, test = frame[years < year], frame[years == year]
        if len(train) < min_train or len(test) < min_test or test[TARGET].nunique() < 2:
            continue
        if max_train and len(train) > max_train:
            train = train.sample(max_train, random_state=seed)
        Xtr, ytr = train.drop(columns=[TARGET, DATE_COL]), train[TARGET]
        Xte, yte = test.drop(columns=[TARGET, DATE_COL]), test[TARGET]
        p = make_pipeline_fn(Xtr, ytr).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        rows.append({"Test_Year": int(year), "Observations": len(yte),
                     "Observed_Default_Rate": float(yte.mean()), "Average_Predicted_PD": float(p.mean()),
                     "ROC_AUC": roc_auc_score(yte, p), "PR_AUC": average_precision_score(yte, p),
                     "Brier": brier_score_loss(yte, p)})
    return pd.DataFrame(rows)
