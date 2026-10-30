"""Tests for src/scoring.py.

What this file checks: min-max scores run in the right direction (and handle constant input), Pareto
dominance on a hand-built table (including equal rows), the scorecard has the expected axes and
bounds and ranks models as expected, and the complexity score is exactly 100 minus Efficiency.
"""

import numpy as np
import pandas as pd

from src.scoring import build_scorecard, complexity_score, find_dominated_models, minmax_score


def test_minmax_direction_and_constant_input():
    s = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    assert minmax_score(s, True).tolist() == [0.0, 50.0, 100.0]
    assert minmax_score(s, False).tolist() == [100.0, 50.0, 0.0]
    assert (minmax_score(pd.Series([2.0, 2.0]), True) == 100.0).all()


def test_pareto_hand_built_table():
    table = pd.DataFrame(
        {"perf": [0.9, 0.8, 0.8, 0.7], "complexity": [50, 10, 60, 5]},
        index=["strong", "simple", "dominated", "tiny"],
    )
    dom = find_dominated_models(table, maximize=["perf"], minimize=["complexity"])
    assert dom["strong"] == [] and dom["simple"] == [] and dom["tiny"] == []
    assert set(dom["dominated"]) == {"strong", "simple"}


def test_equal_rows_do_not_dominate_each_other():
    table = pd.DataFrame({"perf": [0.8, 0.8], "complexity": [1, 1]}, index=["a", "b"])
    assert find_dominated_models(table, ["perf"], ["complexity"]) == {"a": [], "b": []}


def _inputs():
    summary = pd.DataFrame({
        "roc_auc": [0.70, 0.72, 0.73], "pr_auc": [0.30, 0.33, 0.34],
        "brier": [0.15, 0.14, 0.14], "ece": [0.02, 0.03, 0.01],
        "generalization_gap": [0.00, 0.20, 0.05], "seed_std_pr_auc": [0.001, 0.004, 0.003],
        "explanation_stability": [0.99, 0.7, 0.8],
    }, index=["LR", "RF", "XGB"])
    complexity = pd.DataFrame({
        "Structural_Units": [100, 200_000, 50_000],
        "Train_Time_Seconds": [0.1, 5.0, 2.0], "Predict_Time_Seconds": [0.01, 0.5, 0.1],
    }, index=["LR", "RF", "XGB"])
    return summary, complexity


def test_scorecard_ranks_and_bounds():
    card = build_scorecard(*_inputs())
    assert {"Performance", "Calibration", "Generalization", "Efficiency", "Robustness",
            "Explanation_Stability"} == set(card.columns)
    assert card.to_numpy().min() >= 0 and card.to_numpy().max() <= 100
    assert card.loc["XGB", "Performance"] > card.loc["LR", "Performance"]
    assert card.loc["LR", "Efficiency"] == 100  # fewest units and fastest
    assert card.loc["LR", "Generalization"] == 100  # smallest gap
    assert card.loc["RF", "Explanation_Stability"] == 0


def test_complexity_score_is_inverse_of_efficiency():
    card = build_scorecard(*_inputs())
    np.testing.assert_allclose(complexity_score(card), 100 - card["Efficiency"])
