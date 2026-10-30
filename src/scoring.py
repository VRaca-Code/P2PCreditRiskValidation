"""Relative scorecard and Pareto analysis.

What this file does
-------------------
Condenses the measured results into a few 0-100 scores per model (``build_scorecard``:
Performance, Calibration, Generalization, Efficiency, Robustness, Explanation_Stability) using
``minmax_score``, defines the single complexity score used in all plots (``complexity_score``),
and finds Pareto-dominated models (``find_dominated_models``): a model is dominated if another
is at least as good on every chosen axis and strictly better on one.

Every score is min-max scaled ACROSS THE MODELS BEING COMPARED (0 = worst, 100 = best
among them). It ranks models against each other; it is not an absolute quality scale,
and with two models every axis is trivially 0/100. Keep the raw metrics next to it.

There is deliberately no expert-assigned "interpretability" number. Explanation
stability is measured (Spearman across seeds); intrinsic transparency is a modelling
judgement and is left to the write-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def minmax_score(series, higher_is_better: bool = True) -> pd.Series:
    values = pd.Series(series, dtype=float)
    lo, hi = values.min(), values.max()
    if np.isclose(hi, lo):
        return pd.Series(100.0, index=values.index)
    score = (100 * (values - lo) / (hi - lo)).clip(0, 100)  # clip removes float noise such as -0.0
    return score if higher_is_better else (100 - score).clip(0, 100)


def build_scorecard(summary: pd.DataFrame, complexity: pd.DataFrame) -> pd.DataFrame:
    """summary: one row per model with roc_auc, pr_auc, brier, ece, generalization_gap,
    seed_std_pr_auc and (optionally) explanation_stability. complexity: complexity_table output."""
    s = summary.copy()
    card = pd.DataFrame(index=s.index)
    card["Performance"] = 0.5 * minmax_score(s["roc_auc"]) + 0.5 * minmax_score(s["pr_auc"])
    card["Calibration"] = 0.6 * minmax_score(s["brier"], False) + 0.4 * minmax_score(s["ece"], False)
    card["Generalization"] = minmax_score(s["generalization_gap"].clip(lower=0), False)

    c = complexity.reindex(s.index)
    log_units = np.log10(c["Structural_Units"].astype(float).clip(lower=1))
    card["Efficiency"] = (
        0.50 * minmax_score(log_units, False)
        + 0.35 * minmax_score(c["Train_Time_Seconds"], False)
        + 0.15 * minmax_score(c["Predict_Time_Seconds"], False)
    )
    card["Robustness"] = minmax_score(s["seed_std_pr_auc"], False)
    if "explanation_stability" in s:
        card["Explanation_Stability"] = minmax_score(s["explanation_stability"], True)
    return card


def complexity_score(scorecard: pd.DataFrame) -> pd.Series:
    """The single complexity definition used in every trade-off plot: 100 - Efficiency."""
    return 100 - scorecard["Efficiency"]


def find_dominated_models(table: pd.DataFrame, maximize: list[str], minimize: list[str]) -> dict:
    """{model: [models that Pareto-dominate it]}; dominated = no better anywhere, worse somewhere."""
    result = {}
    for a in table.index:
        dominators = []
        for b in table.index:
            if a == b:
                continue
            no_worse = all(table.loc[b, c] >= table.loc[a, c] for c in maximize) and \
                       all(table.loc[b, c] <= table.loc[a, c] for c in minimize)
            better = any(table.loc[b, c] > table.loc[a, c] for c in maximize) or \
                     any(table.loc[b, c] < table.loc[a, c] for c in minimize)
            if no_worse and better:
                dominators.append(b)
        result[a] = dominators
    return result
