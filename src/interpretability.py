"""Global feature importance on ORIGINAL features, comparable across all model families.

What this file does
-------------------
Answers "which loan characteristics does each model rely on?" in a way that can be compared
between models and between seeds:

* ``logistic_coefficients`` - odds ratios of the logistic model (intrinsic interpretability)
* ``permutation_importance_original`` - model-agnostic importance, permuting whole original columns
* ``shap_values_transformed`` / ``global_importance`` - SHAP values (exact linear SHAP for logistic
  regression, Tree SHAP for forests and boosting, Gradient SHAP for DNN/KAN), folded back to
  original features by ``aggregate_to_source`` and normalised to shares by ``importance_from_values``
* ``plot_shap_summary`` - SHAP bar / beeswarm plot with a title that really lands on the figure
The vectors it returns feed the H2 explanation-stability tests in stats.py.

Pipeline: per-row attributions on the transformed columns -> sum signed values of
all columns that come from one original feature (one-hot dummies cancel/add as they
should) -> mean absolute value -> normalise to shares that sum to 1.

Scale note: Tree SHAP for XGBoost is in log-odds, for scikit-learn forests in
probability. Absolute magnitudes are therefore not comparable across models, but the
normalised shares and every rank-based statistic (Jaccard, Spearman) are.
Class-weighted models explain the re-weighted score, not the true PD.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .preprocessing import source_feature_names


# ----------------------------------------------------------------------------- aggregation
def aggregate_to_source(values: np.ndarray, source_names: list[str]) -> pd.DataFrame:
    """Sum SIGNED per-row attributions over columns that share a source feature."""
    frame = pd.DataFrame(np.asarray(values), columns=list(source_names))
    return frame.T.groupby(level=0, sort=False).sum().T


def importance_from_values(values: np.ndarray, source_names: list[str]) -> pd.Series:
    """Mean |aggregated attribution| per original feature, normalised to sum to 1."""
    agg = aggregate_to_source(values, source_names).abs().mean(axis=0)
    total = agg.sum()
    return agg / total if total > 0 else agg


# ----------------------------------------------------------------------------- logistic regression
def logistic_coefficients(pipeline) -> pd.DataFrame:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    coef = pipeline.named_steps["model"].coef_[0]
    out = pd.DataFrame({"Feature": names, "Coefficient": coef})
    out["Odds_Ratio"] = np.exp(out["Coefficient"])
    return out.reindex(out["Coefficient"].abs().sort_values(ascending=False).index)


def linear_shap_values(coef: np.ndarray, X_t: np.ndarray, background_t: np.ndarray) -> np.ndarray:
    """Exact interventional SHAP for a linear model in log-odds: coef * (x - mean(background)).

    Equals ``shap.LinearExplainer`` with an independent masker; implemented directly
    so logistic regression needs no optional dependency.
    """
    return np.asarray(coef) * (np.asarray(X_t) - np.asarray(background_t).mean(axis=0))


# ----------------------------------------------------------------------------- permutation
def permutation_importance_original(pipeline, X, y, scoring: str = "average_precision",
                                    n_repeats: int = 3, seed: int = 0) -> pd.Series:
    """Permute whole ORIGINAL columns through the full pipeline; normalised, floored at 0."""
    result = permutation_importance(pipeline, X, y, scoring=scoring, n_repeats=n_repeats,
                                    random_state=seed, n_jobs=1)
    imp = pd.Series(np.clip(result.importances_mean, 0, None), index=X.columns)
    total = imp.sum()
    return imp / total if total > 0 else imp


# ----------------------------------------------------------------------------- SHAP
def _two_dim(values) -> np.ndarray:
    """Normalise the shapes different SHAP versions return to (n_rows, n_features)."""
    if isinstance(values, list):  # old API: one array per class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1] if values.shape[-1] > 1 else values[:, :, 0]
    return values


def shap_values_transformed(pipeline, X_explain: pd.DataFrame, X_background: pd.DataFrame) -> np.ndarray:
    """Per-row SHAP values on the transformed columns, dispatched on the model type."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = prep.transform(X_explain)
    bg = prep.transform(X_background)

    if hasattr(model, "coef_"):
        return linear_shap_values(model.coef_[0], Xt, bg)

    import shap  # optional dependency
    if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
        return _two_dim(shap.TreeExplainer(model).shap_values(Xt))
    if hasattr(model, "module_"):
        import torch
        explainer = shap.GradientExplainer(model.module_, torch.from_numpy(bg.astype(np.float32)))
        return _two_dim(explainer.shap_values(torch.from_numpy(Xt.astype(np.float32))))
    raise TypeError(f"No SHAP explainer for {type(model).__name__}")


def global_importance(pipeline, X_explain: pd.DataFrame, y_explain, X_background: pd.DataFrame,
                      method: str = "shap", seed: int = 0) -> tuple[pd.Series, str]:
    """Normalised importance per original feature, plus the method actually used.

    method="shap" silently degrades to permutation importance if SHAP is missing or
    fails for this model; the returned label records what really ran.
    """
    if method == "shap":
        try:
            values = shap_values_transformed(pipeline, X_explain, X_background)
            names = source_feature_names(pipeline.named_steps["prep"])
            return importance_from_values(values, names), "shap"
        except ImportError:
            warnings.warn("shap is not installed; falling back to permutation importance")
        except Exception as exc:  # keep long experiment grids alive
            warnings.warn(f"SHAP failed ({type(exc).__name__}: {exc}); using permutation importance")
    return permutation_importance_original(pipeline, X_explain, y_explain, seed=seed), "permutation"


# ----------------------------------------------------------------------------- plotting
def plot_shap_summary(pipeline, X_explain, X_background, title: str, kind: str = "bar",
                      max_display: int = 20, path=None, show: bool = True):
    """SHAP bar / beeswarm with a title that actually lands on the SHAP figure.

    ``show=False`` is the fix: without it SHAP draws and shows its own figure, so a
    following ``plt.title`` decorates a new blank axes.
    """
    import matplotlib.pyplot as plt
    import shap

    values = shap_values_transformed(pipeline, X_explain, X_background)
    names = pipeline.named_steps["prep"].get_feature_names_out()
    explanation = shap.Explanation(values=values, data=pipeline.named_steps["prep"].transform(X_explain),
                                   feature_names=list(names))
    plotter = shap.plots.bar if kind == "bar" else shap.plots.beeswarm
    plotter(explanation, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
