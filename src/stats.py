"""Statistical tests for H1 (performance equivalence) and H2 (explanation stability).

What this file does
-------------------
Provides the inference machinery that turns raw metrics into verdicts:

* H1 - ``crossed_bootstrap_diff`` (resamples seeds and test rows), ``bootstrap_p_value``,
  ``tost_equivalence`` (equivalent / different / inconclusive against a margin) and
  ``holm_adjust`` (multiple-comparison correction)
* H2 - ``jaccard_topk`` (top-k membership overlap), ``spearman_rank`` (ranking agreement) and
  ``explanation_shift_test`` (are explanations more different between models than between
  seeds of the same model?)
* H3 - ``joint_verdict`` (performance-equivalent AND explained differently)

Design follows the "performance vs explanation" validation logic:
  * seed-to-seed variation is the noise baseline;
  * "no difference" claims need an equivalence test (TOST), not a failed difference test;
  * multiple pairwise comparisons are Holm-adjusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

_METRICS = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score}


# ----------------------------------------------------------------------------- H1
def crossed_bootstrap_diff(y, probs_a, probs_b, metric: str = "pr_auc",
                           n_boot: int = 500, seed: int = 0) -> dict:
    """Bootstrap the mean metric difference A - B, resampling seeds AND observations.

    probs_a, probs_b: arrays of shape (n_seeds, n_obs) of predicted PDs on the same
    test rows. Resampling both axes lets the interval reflect model-training noise
    (seeds) as well as test-set sampling noise (rows).
    """
    f = _METRICS[metric]
    y = np.asarray(y)
    A, B = np.atleast_2d(probs_a), np.atleast_2d(probs_b)
    if A.shape != B.shape or A.shape[1] != len(y):
        raise ValueError("probs_a / probs_b must both have shape (n_seeds, len(y))")
    n_seeds, n_obs = A.shape

    observed = float(np.mean([f(y, A[s]) - f(y, B[s]) for s in range(n_seeds)]))
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        rows = rng.integers(0, n_obs, n_obs)
        yb = y[rows]
        if yb.min() == yb.max():
            continue
        per_seed = np.array([f(yb, A[s, rows]) - f(yb, B[s, rows]) for s in range(n_seeds)])
        boot[b] = per_seed[rng.integers(0, n_seeds, n_seeds)].mean()
    boot = boot[~np.isnan(boot)]
    return {"observed": observed, "boot": boot}


def bootstrap_p_value(boot: np.ndarray, null: float = 0.0) -> float:
    """Two-sided percentile-bootstrap p-value for H0: difference == null."""
    lo = (np.sum(boot <= null) + 1) / (len(boot) + 1)
    hi = (np.sum(boot >= null) + 1) / (len(boot) + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def tost_equivalence(boot: np.ndarray, margin: float, alpha: float = 0.05) -> dict:
    """TOST via the (1 - 2*alpha) bootstrap interval: equivalent iff it lies inside (-margin, +margin).

    Verdict is one of "equivalent", "different" (the whole interval is beyond the
    margin on one side) or "inconclusive".
    """
    lo, hi = np.percentile(boot, [100 * alpha, 100 * (1 - alpha)])
    if lo > -margin and hi < margin:
        verdict = "equivalent"
    elif lo >= margin or hi <= -margin:
        verdict = "different"
    else:
        verdict = "inconclusive"
    return {"ci_low": float(lo), "ci_high": float(hi), "verdict": verdict,
            "equivalent": verdict == "equivalent"}


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# ----------------------------------------------------------------------------- H2
def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = a.index.union(b.index)
    return a.reindex(idx, fill_value=0.0), b.reindex(idx, fill_value=0.0)


def top_k_features(importance: pd.Series, k: int) -> list:
    """k most important features; ties broken by name so results are deterministic."""
    order = sorted(zip(-importance.to_numpy(), importance.index.astype(str), importance.index))
    return [name for _, _, name in order[:k]]


def jaccard_topk(a: pd.Series, b: pd.Series, k: int = 10) -> float:
    """Jaccard overlap of the two top-k feature sets (1 = identical membership)."""
    a, b = _align(a, b)
    sa, sb = set(top_k_features(a, k)), set(top_k_features(b, k))
    return len(sa & sb) / len(sa | sb)


def spearman_rank(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation of two importance vectors (nan if either is constant)."""
    a, b = _align(a, b)
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(spearmanr(a.to_numpy(), b.to_numpy())[0])


def _distance(a: pd.Series, b: pd.Series, kind: str, k: int) -> float:
    if kind == "membership":
        return 1.0 - jaccard_topk(a, b, k)
    if kind == "ranking":
        rho = spearman_rank(a, b)
        return float("nan") if np.isnan(rho) else 1.0 - rho
    raise ValueError("kind must be 'membership' or 'ranking'")


def _pair_matrix(A: pd.DataFrame, B: pd.DataFrame, kind: str, k: int) -> np.ndarray:
    return np.array([[_distance(A.iloc[i], B.iloc[j], kind, k) for j in range(len(B))]
                     for i in range(len(A))])


def _mean_offdiag(D: np.ndarray, rows: np.ndarray | None = None) -> float:
    """Mean distance over pairs of *different* original seeds (excludes self-pairs from resampling)."""
    idx = np.arange(len(D)) if rows is None else rows
    sub = D[np.ix_(idx, idx)]
    mask = idx[:, None] != idx[None, :]
    return float(np.nanmean(sub[mask])) if mask.any() else float("nan")


def explanation_shift_test(imp_a: pd.DataFrame, imp_b: pd.DataFrame, kind: str = "membership",
                           k: int = 10, n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> dict:
    """Do explanations differ BETWEEN models by more than they vary across seeds WITHIN a model?

    imp_a, imp_b: DataFrames (rows = seeds, columns = original features) of global
    importances. theta = mean cross-model distance - mean within-model distance.
    One-sided bootstrap over seeds: supported when theta > 0 beyond resampling noise.
    """
    D_aa = _pair_matrix(imp_a, imp_a, kind, k)
    D_bb = _pair_matrix(imp_b, imp_b, kind, k)
    D_ab = _pair_matrix(imp_a, imp_b, kind, k)

    within = 0.5 * (_mean_offdiag(D_aa) + _mean_offdiag(D_bb))
    cross = float(np.nanmean(D_ab))
    theta = cross - within

    rng = np.random.default_rng(seed)
    nA, nB = len(imp_a), len(imp_b)
    thetas = []
    for _ in range(n_boot):
        ia, ib = rng.integers(0, nA, nA), rng.integers(0, nB, nB)
        w = 0.5 * (_mean_offdiag(D_aa, ia) + _mean_offdiag(D_bb, ib))
        thetas.append(float(np.nanmean(D_ab[np.ix_(ia, ib)])) - w)
    thetas = np.array(thetas)
    thetas = thetas[~np.isnan(thetas)]
    return {
        "kind": kind, "d_cross": cross, "d_within": float(within), "theta": float(theta),
        "ci90_low": float(np.percentile(thetas, 5)), "ci90_high": float(np.percentile(thetas, 95)),
        "p_value": float((np.sum(thetas <= 0) + 1) / (len(thetas) + 1)),  # one-sided
    }


# ----------------------------------------------------------------------------- H3
def joint_verdict(h1_equivalent: bool, h2_membership: bool, h2_ranking: bool) -> bool:
    """H3: performance-equivalent AND explained differently (membership or ranking)."""
    return bool(h1_equivalent and (h2_membership or h2_ranking))
