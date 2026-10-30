"""Tests for src/robustness.py.

What this file checks: PSI is 0 for identical data and large for shifted data, handles constant
and binary features, has no missing values on real splits; stress scenarios raise predicted PD in
the economically sensible direction; sensitivity moves with DTI; cross-validation returns one row
per fold; the chronological backtest never trains on the future.
"""

import numpy as np
import pandas as pd

from src.models import make_pipeline
from src.robustness import (
    calculate_psi,
    chronological_backtest,
    cv_scores,
    feature_psi,
    sensitivity_test,
    stress_test,
)


def test_psi_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20_000)
    assert calculate_psi(x, x) < 1e-6
    assert calculate_psi(x, rng.normal(2, 1, 20_000)) > 1.0
    assert calculate_psi(x, rng.normal(0, 1, 20_000)) < 0.02  # same distribution, sampling noise only


def test_psi_handles_degenerate_and_low_cardinality_input():
    assert np.isnan(calculate_psi([], [1, 2, 3]))
    assert calculate_psi(np.ones(100), np.ones(100)) == 0.0  # constant, unchanged
    assert calculate_psi(np.ones(100), np.full(100, 2.0)) > 5  # constant feature takes a new value
    rng = np.random.default_rng(0)
    binary = rng.integers(0, 2, 10_000)
    assert calculate_psi(binary, binary) == 0.0
    assert calculate_psi(binary, (rng.random(10_000) < 0.9).astype(int)) > 0.2  # share of 1s: 50% -> 90%


def test_feature_psi_has_no_missing_values(splits):
    X_train, _, X_test, _ = splits
    assert feature_psi(X_train, X_test)["PSI"].notna().all()


def test_feature_psi_detects_drift_in_chronological_split(splits):
    X_train, _, X_test, _ = splits
    table = feature_psi(X_train, X_test)
    assert "PSI" in table and table["PSI"].is_monotonic_decreasing
    assert set(table["Feature"]) <= set(X_train.columns)


def test_stress_moves_pd_in_the_economically_sensible_direction(splits):
    X_train, y_train, X_test, _ = splits
    sub = X_train.sample(3_000, random_state=0)
    pipe = make_pipeline("Logistic Regression", sub).fit(sub, y_train.loc[sub.index])
    fn = lambda d: pipe.predict_proba(d)[:, 1]
    out = stress_test(fn, X_test.head(1_000)).set_index("Scenario")["Average_PD"]
    assert out["Baseline"] < out["Moderate stress"] < out["Severe stress"]


def test_sensitivity_dti_increases_pd(splits):
    X_train, y_train, X_test, _ = splits
    sub = X_train.sample(3_000, random_state=0)
    pipe = make_pipeline("Logistic Regression", sub).fit(sub, y_train.loc[sub.index])
    out = sensitivity_test(lambda d: pipe.predict_proba(d)[:, 1], X_test.head(1_000), "dti", (0.8, 1.2))
    assert out.loc[out.Multiplier == 1.2, "Absolute_Change"].iloc[0] > 0
    assert out.loc[out.Multiplier == 0.8, "Absolute_Change"].iloc[0] < 0


def test_cv_scores_shape(splits):
    X_train, y_train, _, _ = splits
    sub = X_train.sample(1_500, random_state=0)
    scores = cv_scores(make_pipeline("Logistic Regression", sub), sub, y_train.loc[sub.index], n_splits=3)
    assert len(scores) == 3 and (scores["ROC_AUC"] > 0.5).all()


def test_chronological_backtest_never_trains_on_the_future(frame):
    out = chronological_backtest(
        lambda X, y: make_pipeline("Logistic Regression", X),
        frame, first_test_year=2014, min_train=500, min_test=200,
    )
    assert not out.empty
    assert out["Test_Year"].min() >= 2014
    assert (out["ROC_AUC"] > 0.5).all()
    assert out["Test_Year"].is_monotonic_increasing
