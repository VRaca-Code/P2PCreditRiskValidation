"""Tests for src/stats.py.

What this file checks: Holm correction against a hand computation, Jaccard / Spearman edge cases,
the bootstrap + TOST verdicts (identical models -> equivalent, a clearly better model -> different,
a borderline case -> inconclusive), input validation, and the explanation-shift test (same
explanations -> no shift, reversed drivers -> shift beyond seed noise) and the H3 joint rule.
"""

import numpy as np
import pandas as pd
import pytest

from src.stats import (
    bootstrap_p_value,
    crossed_bootstrap_diff,
    explanation_shift_test,
    holm_adjust,
    jaccard_topk,
    joint_verdict,
    spearman_rank,
    tost_equivalence,
)


def _imp(values):
    return pd.Series(values, index=[f"f{i}" for i in range(len(values))], dtype=float)


def test_holm_matches_hand_computation():
    adj = holm_adjust([0.01, 0.04, 0.03])
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max -> 0.06
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_order():
    adj = holm_adjust([0.5, 0.9])
    assert (adj <= 1).all() and adj[1] >= adj[0]


def test_jaccard_identical_disjoint_partial():
    a = _imp([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    assert jaccard_topk(a, a, k=3) == 1.0
    b = _imp([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert jaccard_topk(a, b, k=3) == 0.0
    c = _imp([10, 9, 1, 8, 7, 6, 5, 4, 3, 2])  # shares f0,f1 with a's top 3
    assert np.isclose(jaccard_topk(a, c, k=3), 2 / 4)


def test_jaccard_ties_are_deterministic():
    a = _imp([1, 1, 1, 1])
    assert jaccard_topk(a, a, k=2) == 1.0


def test_spearman_self_reverse_constant():
    a = _imp([5, 4, 3, 2, 1])
    assert np.isclose(spearman_rank(a, a), 1.0)
    assert np.isclose(spearman_rank(a, a[::-1].set_axis(a.index)), -1.0)
    assert np.isnan(spearman_rank(a, _imp([1, 1, 1, 1, 1])))


def _scores(rng, n, signal, noise):
    y = rng.integers(0, 2, n)
    return y, y * signal + rng.normal(0, noise, n)


def test_identical_models_are_equivalent():
    rng = np.random.default_rng(0)
    y, s = _scores(rng, 3_000, 1.0, 1.0)
    P = np.vstack([s, s + rng.normal(0, 0.01, len(s))])
    res = crossed_bootstrap_diff(y, P, P, "pr_auc", n_boot=200, seed=1)
    assert abs(res["observed"]) < 1e-12
    verdict = tost_equivalence(res["boot"], margin=0.01)
    assert verdict["verdict"] == "equivalent"
    assert bootstrap_p_value(res["boot"]) > 0.5


def test_clearly_better_model_is_different_not_equivalent():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4_000)
    good = np.vstack([y * 2.0 + rng.normal(0, 1, len(y)) for _ in range(3)])
    bad = np.vstack([y * 0.2 + rng.normal(0, 1, len(y)) for _ in range(3)])
    res = crossed_bootstrap_diff(y, good, bad, "pr_auc", n_boot=200, seed=2)
    assert res["observed"] > 0.1
    assert tost_equivalence(res["boot"], margin=0.01)["verdict"] == "different"
    assert bootstrap_p_value(res["boot"]) < 0.05


def test_tost_inconclusive_when_interval_straddles_margin():
    boot = np.random.default_rng(0).normal(0.008, 0.01, 2_000)
    assert tost_equivalence(boot, margin=0.01)["verdict"] == "inconclusive"


def test_bootstrap_shape_validation():
    with pytest.raises(ValueError):
        crossed_bootstrap_diff(np.array([0, 1, 0, 1]), np.zeros((2, 4)), np.zeros((2, 3)))


def _seed_frame(base, n_seeds, noise, rng):
    rows = [np.asarray(base) + rng.normal(0, noise, len(base)) for _ in range(n_seeds)]
    return pd.DataFrame(np.abs(rows), columns=[f"f{i}" for i in range(len(base))])


def test_same_explanations_show_no_shift():
    rng = np.random.default_rng(0)
    base = np.linspace(1.0, 0.1, 20)
    A, B = _seed_frame(base, 8, 0.02, rng), _seed_frame(base, 8, 0.02, rng)
    for kind in ("membership", "ranking"):
        res = explanation_shift_test(A, B, kind=kind, k=5, n_boot=300, seed=0)
        assert abs(res["theta"]) < 0.1
        assert res["p_value"] > 0.05


def test_different_explanations_show_shift_beyond_seed_noise():
    rng = np.random.default_rng(1)
    base = np.linspace(1.0, 0.1, 20)
    A = _seed_frame(base, 8, 0.02, rng)
    B = _seed_frame(base[::-1], 8, 0.02, rng)  # reversed drivers
    for kind in ("membership", "ranking"):
        res = explanation_shift_test(A, B, kind=kind, k=5, n_boot=300, seed=0)
        assert res["theta"] > 0.5
        assert res["ci90_low"] > 0
        assert res["p_value"] < 0.05


def test_joint_verdict_logic():
    assert joint_verdict(True, True, False)
    assert joint_verdict(True, False, True)
    assert not joint_verdict(True, False, False)
    assert not joint_verdict(False, True, True)
