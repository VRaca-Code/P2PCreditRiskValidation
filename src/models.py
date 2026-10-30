"""Model factory: builds the unfitted classifiers and full preprocessing + model pipelines.

What this file does
-------------------
* ``available_models`` reports which of the five models can run in this environment
  (XGBoost and DNN/KAN need optional packages that are only imported when used).
* ``make_estimator`` returns one unfitted classifier with the study's fixed hyper-parameters.
* ``make_pipeline`` wraps it as ``preprocessor -> estimator`` so imputation, scaling and
  encoding are always fitted on the same rows as the model (no leakage).

Imbalance policy
----------------
One imbalance treatment is applied uniformly to every model.

``class_weight=None`` (default) trains all models on the natural default rate, so
Brier/ECE compare intrinsic calibration. ``class_weight="balanced"`` re-weights all
of them; probabilities are then shifted and should be recalibrated before use.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ALL_MODELS
from .preprocessing import build_preprocessor


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def available_models(candidates=ALL_MODELS) -> list[str]:
    """Models whose optional dependencies are installed."""
    ok = {
        "Logistic Regression": True,
        "Random Forest": True,
        "XGBoost": _has("xgboost"),
        "DNN": _has("torch"),
        "KAN": _has("torch"),
    }
    return [m for m in candidates if ok.get(m, False)]


def make_estimator(name: str, seed: int = 0, class_weight: str | None = None, y_train=None):
    """Unfitted final estimator. `y_train` is only needed for XGBoost with class_weight."""
    balanced = class_weight == "balanced"
    if name == "Logistic Regression":
        return LogisticRegression(C=1.0, max_iter=2_000, class_weight=class_weight, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20, max_features="sqrt",
            class_weight="balanced_subsample" if balanced else None,
            random_state=seed, n_jobs=-1,
        )
    if name == "XGBoost":
        if not _has("xgboost"):
            raise ImportError("XGBoost needs the optional dependency: pip install -e '.[boost]'")
        from xgboost import XGBClassifier
        weight = 1.0
        if balanced:
            if y_train is None:
                raise ValueError("y_train is required for XGBoost with class_weight='balanced'")
            weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
            objective="binary:logistic", eval_metric="logloss", scale_pos_weight=weight,
            random_state=seed, n_jobs=-1,
        )
    if name in ("DNN", "KAN"):
        if not _has("torch"):
            raise ImportError("DNN/KAN need the optional dependency: pip install -e '.[neural]'")
        from .neural import TorchTabularClassifier
        return TorchTabularClassifier(
            arch=name.lower(), class_weight=class_weight, random_state=seed
        )
    raise ValueError(f"Unknown model {name!r}; choose from {ALL_MODELS}")


def make_pipeline(name: str, X_ref: pd.DataFrame, seed: int = 0,
                  class_weight: str | None = None, y_train=None) -> Pipeline:
    """preprocessor -> estimator. The preprocessor is fitted on whatever data `.fit` receives."""
    return Pipeline([
        ("prep", build_preprocessor(X_ref)),
        ("model", make_estimator(name, seed, class_weight, y_train)),
    ])
