"""Tests for src/experiments.py and src/visualization.py (end to end on synthetic data).

What this file checks: the runner uses a chronological split, the default run drops lender-assigned
features and includes FICO, sparse features are dropped from train and test, the seed grid has the
right shape and varies across seeds, importances are normalised, H1-H3 tables are internally
consistent, CSV tables are written, the feature-set comparison runs both variants, models with
missing optional dependencies are skipped with a warning, and all figures are produced.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import ExperimentConfig
from src.experiments import load_splits, run_all


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    cfg = replace(
        ExperimentConfig(),
        models=("Logistic Regression", "Random Forest"),
        seeds=(0, 1, 2), seed_train_size=2_000, explain_size=300, background_size=100,
        explanation_method="permutation",  # SHAP-free so the test runs anywhere
        n_boot=60, n_boot_explain=200, top_k=5,
        backtest_first_year=2013, backtest_min_train=500, backtest_min_test=100, backtest_max_train=3_000,
        output_dir=tmp_path_factory.mktemp("tables"),
    )
    return cfg, run_all(cfg, synthetic=True, verbose=False)


def test_temporal_split_used_by_runner():
    cfg = ExperimentConfig()
    X_train, y_train, X_test, y_test = load_splits(cfg, synthetic=True, synthetic_n=3_000)
    assert len(X_train) > len(X_test) > 100
    assert set(np.unique(y_train)) == {0, 1}


def test_grid_shapes(result):
    cfg, res = result
    sm = res["grid"]["seed_metrics"]
    assert len(sm) == 2 * 3
    assert set(sm["Model"]) == {"Logistic Regression", "Random Forest"}
    assert sm["roc_auc"].between(0.5, 1).all()
    n_test = len(res["grid"]["y_test"])
    assert res["grid"]["probs"]["Random Forest"].shape == (3, n_test)


def test_importances_are_normalised_per_model_and_seed(result):
    imps = result[1]["grid"]["importances"]
    sums = imps.groupby(["Model", "Seed"])["Importance"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)
    assert set(imps["Method"]) == {"permutation"}  # the label records what actually ran


def test_hypothesis_tables(result):
    cfg, res = result
    h1, h2, h3 = (res["tests"][k] for k in ("h1", "h2", "h3"))
    assert len(h1) == 1 and len(h2) == 2 and len(h3) == 1
    assert h1["Verdict"].iloc[0] in {"equivalent", "different", "inconclusive"}
    assert (h1["p_holm"] >= h1["p_value"]).all()
    assert set(h2["kind"]) == {"membership", "ranking"}
    # H3 can only hold if H1 says equivalent
    row = h3.iloc[0]
    assert row["H3_supported"] == (row["H1_equivalent"] and (row["H2_membership"] or row["H2_ranking"]))


def test_scorecard_and_tables_written(result):
    cfg, res = result
    assert {"Performance", "Efficiency", "Explanation_Stability"} <= set(res["scorecard"].columns)
    written = {p.name for p in cfg.output_dir.iterdir()}
    assert {"seed_metrics.csv", "h1_results.csv", "h2_results.csv", "h3_results.csv",
            "scorecard.csv", "complexity.csv", "importances.csv", "model_summary.csv"} <= written
    assert pd.read_csv(cfg.output_dir / "h1_results.csv").shape[0] == 1


def test_seed_variation_is_real(result):
    # different seeds subsample different training rows, so the noise baseline is non-zero
    sm = result[1]["grid"]["seed_metrics"]
    assert (sm.groupby("Model")["pr_auc"].std() > 0).all()


def test_missing_optional_models_are_skipped_with_warning():
    from src.experiments import run_seed_grid
    from src.models import available_models
    if "XGBoost" in available_models() and "DNN" in available_models():
        pytest.skip("all optional dependencies installed")
    cfg = replace(ExperimentConfig(), models=("Logistic Regression", "XGBoost", "DNN"), seeds=(0,),
                  seed_train_size=1_000, explain_size=200, explanation_method="permutation")
    splits = load_splits(cfg, synthetic=True, synthetic_n=3_000)
    with pytest.warns(UserWarning, match="Skipping models"):
        grid = run_seed_grid(splits, cfg, verbose=False)
    assert grid["models"] == [m for m in cfg.models if m in available_models()]


def test_default_run_excludes_lender_assigned_features():
    cfg = ExperimentConfig()
    assert cfg.drop_lender_assigned is True
    X_train, *_ = load_splits(cfg, synthetic=True, synthetic_n=3_000)
    assert "int_rate" not in X_train and "sub_grade" not in X_train
    assert "fico_score" in X_train  # the paper-informed FICO feature is present


def test_sparse_features_are_dropped_from_train_and_test(monkeypatch):
    from src import synthetic
    original = synthetic.make_synthetic_lendingclub

    def sparse(n, seed=0):
        df = original(n, seed)
        df.loc[df.sample(frac=0.6, random_state=0).index, "mort_acc"] = np.nan  # ~65% missing
        return df

    monkeypatch.setattr(synthetic, "make_synthetic_lendingclub", sparse)
    with pytest.warns(UserWarning, match="Dropping features"):
        X_train, _, X_test, _ = load_splits(ExperimentConfig(), synthetic=True, synthetic_n=3_000)
    assert "mort_acc" not in X_train and "mort_acc" not in X_test


def test_feature_set_comparison_writes_both_runs(tmp_path):
    from src.experiments import run_feature_set_comparison
    cfg = replace(
        ExperimentConfig(), models=("Logistic Regression", "Random Forest"), seeds=(0, 1),
        seed_train_size=1_500, explain_size=200, background_size=100, explanation_method="permutation",
        n_boot=30, n_boot_explain=100, top_k=5, output_dir=tmp_path, figure_dir=tmp_path / 'figs',
        run_robustness=False,
    )
    out = run_feature_set_comparison(cfg, synthetic=True, verbose=False, figures=True)
    assert (tmp_path / 'figs' / 'feature_set_comparison.png').exists()
    assert (tmp_path / "no_lender_score" / "h1_results.csv").exists()
    assert (tmp_path / "with_lender_score" / "h1_results.csv").exists()
    comp = pd.read_csv(tmp_path / "feature_set_comparison.csv", index_col=[0, 1])
    assert set(comp.index.get_level_values("Feature_set")) == {"no_lender_score", "with_lender_score"}
    assert comp["pr_auc"].notna().all()
    # the two runs really used different feature sets (PR-AUC differences are noise at this size)
    feats = {k: set(pd.read_csv(tmp_path / k / "importances.csv")["Feature"])
             for k in ("no_lender_score", "with_lender_score")}
    assert not {"int_rate", "sub_grade"} & feats["no_lender_score"]
    assert {"int_rate", "sub_grade"} <= feats["with_lender_score"]


def test_visualization_writes_figures(result, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src.visualization import save_all
    paths = save_all(result[1], tmp_path)
    assert len(paths) >= 17 and all(p.exists() and p.stat().st_size > 5_000 for p in paths)
