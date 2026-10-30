"""Tests for src/calibration.py.

What this file checks: class re-weighting visibly distorts predicted probabilities (high ECE),
isotonic recalibration repairs most of it, an un-weighted model is better calibrated than a
re-weighted one, and the calibration table has the expected shape.
"""

import pytest

from src.calibration import calibration_table, compare_calibration
from src.metrics import expected_calibration_error
from src.models import make_pipeline


@pytest.fixture(scope="module")
def reweighted_lr(splits):
    X_train, y_train, _, _ = splits
    pipe = make_pipeline("Logistic Regression", X_train, seed=0, class_weight="balanced")
    return pipe.fit(X_train, y_train)


def test_balanced_weighting_distorts_calibration_and_recalibration_repairs_it(reweighted_lr, splits):
    X_train, y_train, X_test, y_test = splits
    table = compare_calibration({"LR balanced": reweighted_lr}, X_train, y_train, X_test, y_test,
                                methods=("isotonic",))
    ece = table.set_index("Method")["ECE"]
    assert ece["none"] > 0.10  # re-weighting inflates predicted PDs well above the base rate
    assert ece["isotonic"] < ece["none"] / 2


def test_calibration_table_shape(reweighted_lr, splits):
    _, _, X_test, y_test = splits
    t = calibration_table(y_test, reweighted_lr.predict_proba(X_test)[:, 1], n_bins=5)
    assert list(t.columns) == ["mean_predicted_pd", "observed_default_rate"]
    assert 1 < len(t) <= 5


def test_unweighted_model_is_better_calibrated_than_weighted(splits, reweighted_lr):
    X_train, y_train, X_test, y_test = splits
    plain = make_pipeline("Logistic Regression", X_train, seed=0).fit(X_train, y_train)
    e_plain = expected_calibration_error(y_test, plain.predict_proba(X_test)[:, 1])
    e_weighted = expected_calibration_error(y_test, reweighted_lr.predict_proba(X_test)[:, 1])
    assert e_plain < e_weighted
