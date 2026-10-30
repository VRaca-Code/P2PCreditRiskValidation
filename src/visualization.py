"""Matplotlib figures for the study.

What this file does
-------------------
Draws the charts that communicate the results: the H1 equivalence forest plot
(``plot_equivalence_forest``), the H2 explanation-shift plot (``plot_explanation_shift``),
reliability curves (``plot_calibration``), precision-recall curves (``plot_pr_curves``), a
two-axis trade-off scatter (``plot_tradeoff``) and the relative-scorecard radar
(``plot_radar``). Added in v0.3.0: ROC curves, confusion matrices, threshold trade-off,
DNN/KAN training curves, logistic-regression coefficients, global importance bars, per-seed
boxplots, SHAP rank-agreement heatmap, sensitivity / stress / backtest / PSI line and bar charts,
a 2x2 trade-off dashboard and the with/without-lender-score comparison.
``save_all`` writes the full set of PNGs from ``experiments.run_all`` output; each figure is
wrapped so one failure (or a missing optional dependency) never stops the others.

Colour follows the model, never its rank, so a model keeps its colour in every plot.

Palette: categorical slots 1-5 of a validated order (adjacent-pair CVD and normal-vision
checks pass). Three hues are below 3:1 contrast on white, so every chart that uses them
has direct labels and the same numbers are saved as CSV tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

MODEL_COLORS = {
    "Logistic Regression": "#2a78d6",
    "Random Forest": "#eb6834",
    "XGBoost": "#1baf7a",
    "DNN": "#eda100",
    "KAN": "#e87ba4",
}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE = MODEL_COLORS["Logistic Regression"]


def _style(ax, grid_axis: str = "y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _finish(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig


def _color(name: str) -> str:
    return MODEL_COLORS.get(name, "#6250d6")


def plot_equivalence_forest(h1: pd.DataFrame, path=None):
    """H1: PR-AUC difference with its 90% interval against the equivalence band."""
    labels = (h1["Model_A"] + "  −  " + h1["Model_B"]).tolist()
    y = np.arange(len(h1))[::-1]
    margin = float(h1["Margin"].iloc[0])
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(h1) + 1.6))
    ax.axvspan(-margin, margin, color=BLUE, alpha=0.12, linewidth=0)
    ax.axvline(0, color=INK_2, linewidth=1)
    for yi, (_, r) in zip(y, h1.iterrows()):
        ax.plot([r.ci90_low, r.ci90_high], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
        ax.plot(r.PR_AUC_diff, yi, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(r.Verdict, (r.ci90_high, yi), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    ax.set_xlabel(f"PR-AUC difference (90% bootstrap interval); shaded band = equivalence margin ±{margin:g}")
    ax.set_title("H1: are models performance-equivalent?", loc="left", fontsize=12)
    _style(ax, "x")
    return _finish(fig, path)


def plot_explanation_shift(h2: pd.DataFrame, path=None):
    """H2: explanation distance beyond seed noise (theta > 0 means a real shift)."""
    kinds = [("membership", "Top-k membership (1 − Jaccard)"), ("ranking", "Ranking (1 − Spearman)")]
    pairs = (h2["Model_A"] + "  −  " + h2["Model_B"]).unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(pairs) + 1.8), sharey=True)
    y = np.arange(len(pairs))[::-1]
    for ax, (kind, title) in zip(axes, kinds):
        sub = h2[h2.kind == kind].reset_index(drop=True)
        ax.axvline(0, color=INK_2, linewidth=1)
        for yi, (_, r) in zip(y, sub.iterrows()):
            ax.plot([r.ci90_low, r.ci90_high], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round")
            ax.plot(r.theta, yi, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            if r.Supported:
                ax.annotate("shift beyond seed noise", (r.ci90_high, yi), xytext=(8, 0),
                            textcoords="offset points", va="center", fontsize=8.5, color=INK_2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("θ = cross-model distance − within-model distance")
        _style(ax, "x")
    axes[0].set_yticks(y, pairs, fontsize=9, color=INK)
    fig.suptitle("H2: do explanations differ more than seeds alone would explain?", x=0.01, ha="left", fontsize=12)
    return _finish(fig, path)


def plot_calibration(probs: dict, y_true, path=None, n_bins: int = 10):
    """Reliability curves; each model is labelled at the end of its line."""
    from .calibration import calibration_table
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    top = 0.0
    for name, p in probs.items():
        p = np.asarray(p)
        p = p[-1] if p.ndim == 2 else p
        t = calibration_table(y_true, p, n_bins)
        ax.plot(t.mean_predicted_pd, t.observed_default_rate, "-o", color=_color(name),
                linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        ax.annotate(name, (t.mean_predicted_pd.iloc[-1], t.observed_default_rate.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center")
        top = max(top, t.mean_predicted_pd.max(), t.observed_default_rate.max())
    lim = min(1.0, top * 1.15)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted PD"); ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (dashed = perfect)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "both")
    return _finish(fig, path)


def plot_pr_curves(probs: dict, y_true, path=None):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    base = float(np.mean(y_true))
    ax.axhline(base, color=INK_2, linewidth=1, linestyle="--")
    for name, p in probs.items():
        p = np.asarray(p)
        p = p[-1] if p.ndim == 2 else p
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, color=_color(name), linewidth=2, label=name)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision–recall (dashed = default rate {base:.1%})", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "both")
    return _finish(fig, path)


def plot_tradeoff(table: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, title: str, path=None,
                  min_y_span: float = 0.0):
    """Scatter of model scores with a direct label on every point.

    `min_y_span` stops matplotlib auto-zooming a tiny metric gap into a dramatic-looking one.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, r in table.iterrows():
        ax.scatter(r[x], r[y], s=110, color=_color(name), edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(name, (r[x], r[y]), xytext=(9, 6), textcoords="offset points", fontsize=9, color=INK)
    lo, hi = table[y].min(), table[y].max()
    if hi - lo < min_y_span:
        mid = (lo + hi) / 2
        ax.set_ylim(mid - min_y_span / 2, mid + min_y_span / 2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12)
    _style(ax, "both")
    return _finish(fig, path)


def plot_radar(scorecard: pd.DataFrame, path=None):
    """Relative 0-100 scores across models. Pair it with the raw tables; it is a summary only."""
    cats = scorecard.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(7.5, 7), subplot_kw={"polar": True})
    for name, row in scorecard.iterrows():
        vals = row[cats].tolist()
        ax.plot(angles + angles[:1], vals + vals[:1], color=_color(name), linewidth=2, label=name)
        ax.fill(angles + angles[:1], vals + vals[:1], color=_color(name), alpha=0.06)
    ax.set_xticks(angles, [c.replace("_", " ") for c in cats], fontsize=9, color=INK)
    ax.set_ylim(0, 100); ax.set_yticks([25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=8, colors=INK_2)
    ax.tick_params(axis="x", pad=16)  # keep axis names clear of the outer ring
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=9)
    ax.set_title("Relative scorecard (0 = worst, 100 = best among compared models)", fontsize=11, pad=24, color=INK)
    return _finish(fig, path)



# ----------------------------------------------------------------------------- v0.3.0 additions
def _mean_prob(p):
    p = np.asarray(p)
    return p.mean(axis=0) if p.ndim == 2 else p


def plot_roc_curves(probs: dict, y_true, path=None):
    """ROC curves (seed-averaged probabilities); dashed diagonal = random ranking."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--")
    for name, p in probs.items():
        p = _mean_prob(p)
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, color=_color(name), linewidth=2, label=f"{name} (AUC {roc_auc_score(y_true, p):.3f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "both")
    return _finish(fig, path)


def plot_confusion_matrices(probs: dict, y_true, threshold: float, path=None):
    """One confusion matrix per model at a shared threshold (counts, with row-normalised shares)."""
    y_true = np.asarray(y_true)
    n = len(probs)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), squeeze=False)
    for ax, (name, p) in zip(axes[0], probs.items()):
        pred = (_mean_prob(p) >= threshold).astype(int)
        cm = np.array([[np.sum((y_true == a) & (pred == b)) for b in (0, 1)] for a in (0, 1)])
        share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="white" if share[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["Paid", "Default"], fontsize=9); ax.set_yticks([0, 1], ["Paid", "Default"], fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_title(name, fontsize=10, color=INK)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0][0].set_ylabel("Actual")
    fig.suptitle(f"Confusion matrices at threshold {threshold:.2f} (cell % = share of actual class)",
                 x=0.01, ha="left", fontsize=11)
    return _finish(fig, path)


def plot_threshold_tradeoff(probs: dict, y_true, path=None):
    """Precision and recall across thresholds, one panel each, one line per model."""
    y_true = np.asarray(y_true)
    grid = np.linspace(0.02, 0.8, 60)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for name, p in probs.items():
        p = _mean_prob(p)
        prec, rec = [], []
        for t in grid:
            pred = p >= t
            tp = np.sum(pred & (y_true == 1))
            prec.append(tp / pred.sum() if pred.sum() else np.nan)
            rec.append(tp / max(np.sum(y_true == 1), 1))
        axes[0].plot(grid, prec, color=_color(name), linewidth=2, label=name)
        axes[1].plot(grid, rec, color=_color(name), linewidth=2, label=name)
    for ax, lab in zip(axes, ("Precision", "Recall")):
        ax.set_xlabel("Decision threshold on predicted PD"); ax.set_ylabel(lab)
        ax.set_title(f"{lab} vs threshold", loc="left", fontsize=11)
        _style(ax, "both")
    axes[0].legend(frameon=False, fontsize=9)
    return _finish(fig, path)


def plot_training_curves(fitted: dict, path=None):
    """Train vs validation loss per epoch for the neural models (DNN, KAN)."""
    nets = {n: p.named_steps["model"].history_ for n, p in fitted.items()
            if hasattr(p.named_steps.get("model"), "history_")}
    if not nets:
        raise ValueError("No neural model with a training history")
    fig, axes = plt.subplots(1, len(nets), figsize=(5.5 * len(nets), 4), squeeze=False)
    for ax, (name, h) in zip(axes[0], nets.items()):
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=_color(name), linewidth=2, label="train")
        ax.plot(ep, h["val_loss"], color=_color(name), linewidth=2, linestyle="--", label="validation")
        best = int(np.argmin(h["val_loss"])) + 1
        ax.axvline(best, color=INK_2, linewidth=1, linestyle=":")
        ax.set_title(f"{name}: best epoch {best}", loc="left", fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Log loss")
        ax.legend(frameon=False, fontsize=9)
        _style(ax, "both")
    return _finish(fig, path)


def plot_logistic_coefficients(coefs: pd.DataFrame, path=None, top: int = 15):
    """Largest logistic-regression coefficients (log-odds per standardised unit)."""
    c = coefs.reindex(coefs["Coefficient"].abs().sort_values(ascending=False).index).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(c) + 1.6))
    ax.barh(c["Feature"], c["Coefficient"], color=BLUE, height=0.6)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_xlabel("Coefficient (right = higher default risk)")
    ax.set_title("Logistic regression: strongest coefficients", loc="left", fontsize=12)
    _style(ax, "x")
    return _finish(fig, path)


def plot_importance_bars(importances: pd.DataFrame, path=None, top: int = 10):
    """Mean explanation share per source feature, one panel per model, with seed spread."""
    models = list(dict.fromkeys(importances["Model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        sub = importances[importances["Model"] == m].groupby("Feature")["Importance"].agg(["mean", "std"])
        sub = sub.sort_values("mean", ascending=False).head(top)[::-1]
        ax.barh(sub.index, sub["mean"], xerr=sub["std"].fillna(0), color=_color(m), height=0.6,
                error_kw={"ecolor": INK_2, "elinewidth": 1})
        ax.set_title(m, loc="left", fontsize=11); ax.set_xlabel("Share of total importance (±1 sd over seeds)")
        _style(ax, "x")
    return _finish(fig, path)


def plot_seed_boxplots(seed_metrics: pd.DataFrame, path=None, metrics=("pr_auc", "roc_auc", "brier")):
    """Per-seed spread of test metrics: the noise baseline every comparison is judged against."""
    metrics = [m for m in metrics if m in seed_metrics.columns]
    models = list(dict.fromkeys(seed_metrics["Model"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.2), squeeze=False)
    for ax, met in zip(axes[0], metrics):
        data = [seed_metrics.loc[seed_metrics.Model == m, met] for m in models]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK})
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(_color(m)); patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(models) + 1), [m.replace(" ", "\n") for m in models], fontsize=8)
        ax.set_title(met.replace("_", "-").upper(), loc="left", fontsize=11)
        _style(ax, "y")
    return _finish(fig, path)


def plot_rank_agreement(importances: pd.DataFrame, path=None):
    """Mean Spearman correlation of feature importances between every pair of (model, seed) runs, by model pair."""
    from scipy.stats import spearmanr
    wide = importances.pivot_table(index=["Model", "Seed"], columns="Feature", values="Importance").fillna(0.0)
    models = list(dict.fromkeys(wide.index.get_level_values(0)))
    mat = np.full((len(models), len(models)), np.nan)
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            vals = []
            for sa in wide.loc[a].index:
                for sb in wide.loc[b].index:
                    if a == b and sa >= sb:
                        continue
                    r = spearmanr(wide.loc[a].loc[sa], wide.loc[b].loc[sb])[0]
                    if not np.isnan(r):
                        vals.append(r)
            if vals:
                mat[i, j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(1.2 * len(models) + 3, 1.0 * len(models) + 2.4))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(models)):
        for j in range(len(models)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(models)), models, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(models)), models, fontsize=9)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Explanation rank agreement (diagonal = same model, different seeds)", loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman")
    return _finish(fig, path)


def plot_sensitivity(sens: pd.DataFrame, path=None):
    """Change in mean predicted PD when one feature is scaled, one panel per feature."""
    feats = list(dict.fromkeys(sens["Feature"]))
    fig, axes = plt.subplots(1, len(feats), figsize=(4.4 * len(feats), 4), squeeze=False, sharey=True)
    for ax, f in zip(axes[0], feats):
        for m, g in sens[sens.Feature == f].groupby("Model", sort=False):
            g = g.sort_values("Multiplier")
            ax.plot(g["Multiplier"], g["Absolute_Change"] * 100, "-o", color=_color(m), linewidth=2, markersize=5, label=m)
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_title(f, loc="left", fontsize=11); ax.set_xlabel("Feature multiplier")
        _style(ax, "both")
    axes[0][0].set_ylabel("Change in mean PD (percentage points)")
    axes[0][0].legend(frameon=False, fontsize=8)
    return _finish(fig, path)


def plot_stress(stress: pd.DataFrame, path=None):
    """Mean predicted PD per stress scenario, grouped bars by model."""
    scen = list(dict.fromkeys(stress["Scenario"]))
    models = list(dict.fromkeys(stress["Model"]))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, m in enumerate(models):
        g = stress[stress.Model == m].set_index("Scenario").reindex(scen)
        ax.bar(np.arange(len(scen)) + k * w - 0.4 + w / 2, g["Average_PD"] * 100, width=w * 0.9, color=_color(m), label=m)
    ax.set_xticks(range(len(scen)), scen)
    ax.set_ylabel("Average predicted PD (%)")
    ax.set_title("Stress scenarios (illustrative multiplicative shocks)", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "y")
    return _finish(fig, path)


def plot_psi(psi: pd.DataFrame, path=None, top: int = 12):
    """Population Stability Index per feature (train vs test) with the usual 0.1 / 0.25 guides."""
    p = psi.sort_values("PSI", ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(p) + 1.6))
    ax.barh(p["Feature"], p["PSI"], color=BLUE, height=0.6)
    for x in (0.1, 0.25):
        ax.axvline(x, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("PSI (dashed: 0.1 = moderate, 0.25 = large shift)")
    ax.set_title("Feature drift between training and test period", loc="left", fontsize=12)
    _style(ax, "x")
    return _finish(fig, path)


def plot_backtest(bt: pd.DataFrame, path=None):
    """Expanding-window backtest: PR-AUC by test year, and predicted vs observed default rate."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for m, g in bt.groupby("Model", sort=False):
        g = g.sort_values("Test_Year")
        axes[0].plot(g["Test_Year"], g["PR_AUC"], "-o", color=_color(m), linewidth=2, markersize=5, label=m)
        axes[1].plot(g["Test_Year"], g["Average_Predicted_PD"] * 100, "-o", color=_color(m), linewidth=2, markersize=5)
    obs = bt.groupby("Test_Year")["Observed_Default_Rate"].first() * 100
    axes[1].plot(obs.index, obs.values, color=INK, linewidth=2, linestyle="--", label="Observed")
    axes[0].set_title("PR-AUC by test year", loc="left", fontsize=11); axes[0].set_ylabel("PR-AUC")
    axes[1].set_title("Predicted vs observed default rate", loc="left", fontsize=11); axes[1].set_ylabel("Default rate (%)")
    for ax in axes:
        ax.set_xlabel("Test year"); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _style(ax, "both")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    return _finish(fig, path)


def plot_tradeoff_dashboard(table: pd.DataFrame, path=None):
    """2x2 grid of trade-off scatters: complexity, training time, stability and calibration vs performance."""
    panels = [("Complexity_Score", "pr_auc", "Complexity score", "PR-AUC"),
              ("Train_Time_Seconds", "pr_auc", "Training time (s)", "PR-AUC"),
              ("explanation_stability", "pr_auc", "Explanation stability", "PR-AUC"),
              ("ece", "brier", "ECE (lower is better)", "Brier (lower is better)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (x, y, xl, yl) in zip(axes.ravel(), panels):
        if x not in table or y not in table:
            ax.set_visible(False)
            continue
        for name, r in table.iterrows():
            ax.scatter(r[x], r[y], s=100, color=_color(name), edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(name, (r[x], r[y]), xytext=(8, 5), textcoords="offset points", fontsize=8.5, color=INK)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        lo, hi = table[y].min(), table[y].max()
        span = 0.05 if y == "pr_auc" else 0.02  # keep a 0.01 gap from looking like a cliff
        if hi - lo < span:
            ax.set_ylim((lo + hi) / 2 - span / 2, (lo + hi) / 2 + span / 2)
        _style(ax, "both")
    fig.suptitle("Trade-offs between models", x=0.01, ha="left", fontsize=12)
    return _finish(fig, path)


def plot_feature_set_comparison(comparison: pd.DataFrame, path=None, metric: str = "pr_auc"):
    """Mean test metric with vs without the lender-assigned score (int_rate / sub_grade)."""
    t = comparison[metric].unstack("Feature_set")
    cols = [c for c in ("no_lender_score", "with_lender_score") if c in t.columns]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    w = 0.38
    shades = {"no_lender_score": 0.55, "with_lender_score": 1.0}
    for k, c in enumerate(cols):
        for i, m in enumerate(t.index):
            ax.bar(i + (k - 0.5) * w, t.loc[m, c], width=w * 0.95, color=_color(m), alpha=shades[c],
                   label=c.replace("_", " ") if i == 0 else None)
    ax.set_xticks(range(len(t.index)), [m.replace(" ", "\n") for m in t.index], fontsize=9)
    ax.set_ylabel(f"Mean test {metric.replace('_', '-').upper()}")
    ax.set_title("Effect of the lender's own score (light = without, solid = with)", loc="left", fontsize=12)
    _style(ax, "y")
    return _finish(fig, path)


def save_all(result: dict, figure_dir: Path) -> list[Path]:
    """Write the full figure set from `experiments.run_all` output; returns the files written.

    Every figure is built independently: a failure becomes a warning, not a crash.
    """
    import warnings
    from .scoring import complexity_score

    figure_dir = Path(figure_dir)
    grid, tests = result["grid"], result["tests"]
    y, probs = grid["y_test"], grid["probs"]
    thr = grid.get("threshold", float(np.mean(y)))
    table = result["summary"].join(result["scorecard"]).join(grid["complexity"], rsuffix="_c").copy()
    table["Complexity_Score"] = complexity_score(result["scorecard"])
    rob = result.get("robustness") or {}

    jobs = [
        (plot_equivalence_forest, (tests["h1"],), {}, "h1_equivalence.png"),
        (plot_explanation_shift, (tests["h2"],), {}, "h2_explanation_shift.png"),
        (plot_calibration, (probs, y), {}, "calibration.png"),
        (plot_pr_curves, (probs, y), {}, "precision_recall.png"),
        (plot_radar, (result["scorecard"],), {}, "scorecard_radar.png"),
        (plot_tradeoff, (table, "Complexity_Score", "pr_auc", "Complexity score (100 − Efficiency)",
                         "Mean test PR-AUC", "Performance vs complexity"), {"min_y_span": 0.05},
         "performance_vs_complexity.png"),
        (plot_roc_curves, (probs, y), {}, "roc_curves.png"),
        (plot_confusion_matrices, (probs, y, thr), {}, "confusion_matrices.png"),
        (plot_threshold_tradeoff, (probs, y), {}, "threshold_tradeoff.png"),
        (plot_importance_bars, (grid["importances"],), {}, "importance_bars.png"),
        (plot_seed_boxplots, (grid["seed_metrics"],), {}, "seed_boxplots.png"),
        (plot_rank_agreement, (grid["importances"],), {}, "rank_agreement.png"),
        (plot_tradeoff_dashboard, (table,), {}, "tradeoff_dashboard.png"),
    ]
    if "Logistic Regression" in grid["fitted"]:
        from .interpretability import logistic_coefficients
        jobs.append((plot_logistic_coefficients,
                     (logistic_coefficients(grid["fitted"]["Logistic Regression"]),), {}, "logistic_coefficients.png"))
    if any(hasattr(p.named_steps.get("model"), "history_") for p in grid["fitted"].values()):
        jobs.append((plot_training_curves, (grid["fitted"],), {}, "training_curves.png"))
    if len(rob.get("psi", [])):
        jobs.append((plot_psi, (rob["psi"],), {}, "psi.png"))
    if len(rob.get("sensitivity", [])):
        jobs.append((plot_sensitivity, (rob["sensitivity"],), {}, "sensitivity.png"))
    if len(rob.get("stress", [])):
        jobs.append((plot_stress, (rob["stress"],), {}, "stress.png"))
    if len(rob.get("backtest", [])):
        jobs.append((plot_backtest, (rob["backtest"],), {}, "backtest.png"))

    written = []
    for fn, args, kwargs, name in jobs:
        try:
            plt.close(fn(*args, path=figure_dir / name, **kwargs))
            written.append(figure_dir / name)
        except Exception as exc:  # one broken figure must not lose the others
            plt.close("all")
            warnings.warn(f"Figure {name} skipped ({type(exc).__name__}: {exc})")
    return written
