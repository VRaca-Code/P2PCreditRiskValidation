"""One definition of model complexity, used by every table and plot.

What this file does
-------------------
``structural_units`` counts how big a fitted model is in its own natural unit, and
``complexity_table`` combines that with mean training and prediction time into one table per
model. scoring.py turns the table into the Efficiency / Complexity scores, so the same
definition feeds the scorecard, the radar chart and the trade-off plots.

Structural units: coefficients (logistic regression), tree nodes (forests, boosting)
or trainable parameters (neural nets, KAN). They are comparable only on a log scale
and only as an ordering aid, not as an exact exchange rate between model families.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def structural_units(pipeline) -> tuple[int, str]:
    model = pipeline.named_steps["model"] if hasattr(pipeline, "named_steps") else pipeline
    if hasattr(model, "coef_"):
        return int(model.coef_.size + np.size(model.intercept_)), "coefficients + intercept"
    if hasattr(model, "estimators_") and hasattr(model.estimators_[0], "tree_"):
        return int(sum(t.tree_.node_count for t in model.estimators_)), "total tree nodes"
    if hasattr(model, "get_booster"):
        return int(len(model.get_booster().trees_to_dataframe())), "total boosted-tree nodes"
    if hasattr(model, "n_parameters"):
        return int(model.n_parameters()), "trainable parameters"
    raise TypeError(f"Don't know how to count structural units for {type(model).__name__}")


def complexity_table(fitted: dict, train_time: dict, predict_time: dict) -> pd.DataFrame:
    """fitted: {model name: fitted pipeline}; times in seconds."""
    rows = []
    for name, pipe in fitted.items():
        units, measure = structural_units(pipe)
        rows.append({
            "Model": name,
            "Structural_Units": units,
            "Structural_Measure": measure,
            "Train_Time_Seconds": train_time.get(name, np.nan),
            "Predict_Time_Seconds": predict_time.get(name, np.nan),
        })
    return pd.DataFrame(rows).set_index("Model")
