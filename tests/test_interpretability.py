"""Tests for src/interpretability.py.

What this file checks: SHAP values of one-hot dummies are summed with their sign before taking
absolute values, linear SHAP is exactly additive in log-odds, permutation importance finds the true
drivers, logistic SHAP importances are normalised over original features, the method label records
what really ran, coefficients are sorted, and (if shap is installed) Tree SHAP works on a forest.
"""

import numpy as np
import pandas as pd
import pytest

from src.interpretability import (
    aggregate_to_source,
    global_importance,
    importance_from_values,
    linear_shap_values,
    logistic_coefficients,
    permutation_importance_original,
)
from src.models import make_pipeline


def test_aggregation_sums_signed_values_before_abs():
    # Two dummies of one feature with opposite signs cancel for that row (+1 and -1 -> 0).
    values = np.array([[1.0, -1.0, 2.0],
                       [1.0, 1.0, 2.0]])
    names = ["cat", "cat", "num"]
    agg = aggregate_to_source(values, names)
    assert list(agg.columns) == ["cat", "num"]
    np.testing.assert_allclose(agg["cat"], [0.0, 2.0])
    imp = importance_from_values(values, names)
    np.testing.assert_allclose(imp["cat"], 1.0 / (1.0 + 2.0))  # mean|[0,2]| = 1 ; num = 2
    assert np.isclose(imp.sum(), 1.0)


def test_linear_shap_is_additive_in_log_odds():
    rng = np.random.default_rng(0)
    coef = rng.normal(size=6)
    X, bg = rng.normal(size=(20, 6)), rng.normal(size=(50, 6))
    phi = linear_shap_values(coef, X, bg)
    # sum_j phi_j(x) == f(x) - mean_bg f(bg)  with f(x) = coef . x  (intercept cancels)
    np.testing.assert_allclose(phi.sum(axis=1), X @ coef - (bg @ coef).mean())


@pytest.fixture(scope="module")
def fitted_lr(splits):
    X_train, y_train, _, _ = splits
    sub = X_train.sample(2_500, random_state=0)
    return make_pipeline("Logistic Regression", sub).fit(sub, y_train.loc[sub.index])


def test_permutation_importance_finds_the_driver(fitted_lr, splits):
    _, _, X_test, y_test = splits
    imp = permutation_importance_original(fitted_lr, X_test.head(1_500), y_test.head(1_500), n_repeats=2)
    assert np.isclose(imp.sum(), 1.0)
    assert imp.idxmax() in {"int_rate", "sub_grade", "term", "dti", "annual_inc", "revol_util"}
    assert imp["pub_rec_bankruptcies"] < imp.max()


def test_logistic_shap_importance_is_normalised_over_original_features(fitted_lr, splits):
    X_train, _, X_test, y_test = splits
    imp, method = global_importance(fitted_lr, X_test.head(500), y_test.head(500),
                                    X_train.head(200), method="shap")
    assert method == "shap"  # linear SHAP is built in, no optional dependency
    assert np.isclose(imp.sum(), 1.0)
    assert set(imp.index) <= set(X_test.columns)
    assert "home_ownership" in imp.index  # one-hot dummies were folded back


def test_permutation_method_is_labelled(fitted_lr, splits):
    X_train, _, X_test, y_test = splits
    imp, method = global_importance(fitted_lr, X_test.head(500), y_test.head(500),
                                    X_train.head(200), method="permutation")
    assert method == "permutation" and np.isclose(imp.sum(), 1.0)


def test_logistic_coefficients_sorted_by_magnitude(fitted_lr):
    table = logistic_coefficients(fitted_lr)
    assert table["Coefficient"].abs().is_monotonic_decreasing
    assert np.allclose(table["Odds_Ratio"], np.exp(table["Coefficient"]))


def test_tree_shap_matches_importance_shape(splits):
    pytest.importorskip("shap")
    X_train, y_train, X_test, y_test = splits
    sub = X_train.sample(1_500, random_state=0)
    rf = make_pipeline("Random Forest", sub).fit(sub, y_train.loc[sub.index])
    imp, method = global_importance(rf, X_test.head(200), y_test.head(200), X_train.head(100))
    assert method == "shap" and np.isclose(imp.sum(), 1.0)
