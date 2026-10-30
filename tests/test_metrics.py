"""Tests for src/metrics.py.

What this file checks: ECE is ~0 for calibrated probabilities and large for confidently wrong ones
(plus an exact hand-computed value), KS is 1 for perfect separation and ~0 for random scores,
the threshold table is monotone (recall falls and approval rises with the threshold), and the
bundled metric functions return the expected keys.
"""

import numpy as np

from src.metrics import (
    evaluate,
    expected_calibration_error,
    ks_statistic,
    model_metrics,
    threshold_table,
)


def test_ece_near_zero_for_calibrated_probabilities():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = rng.random(200_000) < p
    assert expected_calibration_error(y, p) < 0.01


def test_ece_large_for_overconfident_wrong_probabilities():
    rng = np.random.default_rng(0)
    y = rng.random(50_000) < 0.2
    assert expected_calibration_error(y, np.full(50_000, 0.8)) > 0.55


def test_ece_exact_value_single_bin():
    # all predictions 0.3, observed rate 0.5 -> |0.5 - 0.3| = 0.2
    y = np.array([0, 1] * 50)
    assert np.isclose(expected_calibration_error(y, np.full(100, 0.3)), 0.2)


def test_ks_perfect_and_random():
    y = np.array([0] * 100 + [1] * 100)
    assert np.isclose(ks_statistic(y, np.r_[np.zeros(100), np.ones(100)]), 1.0)
    rng = np.random.default_rng(1)
    yr = rng.integers(0, 2, 20_000)
    assert ks_statistic(yr, rng.random(20_000)) < 0.03


def test_threshold_table_is_monotone():
    rng = np.random.default_rng(2)
    p = rng.random(5_000)
    y = rng.random(5_000) < p
    t = threshold_table(y, p)
    assert t["recall"].is_monotonic_decreasing
    assert t["approval_rate"].is_monotonic_increasing
    assert len(t) == 19


def test_evaluate_and_model_metrics_keys():
    rng = np.random.default_rng(3)
    p = rng.random(2_000)
    y = (rng.random(2_000) < p).astype(int)
    e = evaluate(y, p, 0.5)
    assert {"roc_auc", "pr_auc", "ks", "brier", "ece", "precision", "recall"} <= set(e)
    row = model_metrics("m", y, p, y, p, 1.0, 0.1, 0.5)
    assert np.isclose(row["Generalization_Gap"], 0.0)
    assert row["Model"] == "m"
