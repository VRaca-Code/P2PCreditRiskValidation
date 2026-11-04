"""Experiment runner: multi-seed model grid, hypothesis tests H1-H3, tables, and the command line.

What this file does
-------------------
This is the main entry point that ties every other module together:

1. ``load_splits``     - read the data (real CSV or synthetic), clean it, split chronologically,
                         and drop features observed for too few training loans.
2. ``run_seed_grid``   - fit every model for every seed on a different training subsample,
                         score the SAME test rows, and compute a global importance vector
                         (SHAP or permutation) on the SAME explanation rows.
3. ``run_hypothesis_tests`` - H1 (PR-AUC equivalence, bootstrap + TOST), H2 (explanations
                         differ more than seeds alone explain), H3 (equivalent performance but
                         different explanations), all with Holm correction.
4. ``summarise`` / ``save_tables`` / ``run_all`` - per-model summary, scorecard and CSV outputs.
5. ``run_robustness`` - drift (PSI), sensitivity, stress and a per-model chronological backtest.
6. ``run_feature_set_comparison`` - runs the whole experiment twice, without and with the
                         lender-assigned ``int_rate`` / ``sub_grade``, and compares the results.
   The command line also writes the figure set (visualization.save_all); use --no-figures to skip.

Command line:

    python -m src.experiments --synthetic --seeds 3                 # quick pipeline check
    python -m src.experiments --csv data/raw/accepted_2007_to_2018Q4.csv
    python -m src.experiments --both-feature-sets                   # with AND without lender score

Hypotheses (see README):
  H1  pairs of models are performance-equivalent (PR-AUC, TOST) or not
  H2  explanations shift between models by more than seed-to-seed variation
  H3  some pair is performance-equivalent AND explained differently
"""

from __future__ import annotations

import argparse
import itertools
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .complexity import complexity_table
from .config import ALL_MODELS, DATE_COL, TARGET, ExperimentConfig
from .interpretability import global_importance
from .metrics import evaluate
from .models import available_models, make_pipeline
from .preprocessing import find_sparse_features, prepare_frame, split_xy, temporal_split
from .robustness import DEFAULT_STRESS, chronological_backtest, feature_psi, sensitivity_test, stress_test
from .scoring import build_scorecard
from .stats import (
    bootstrap_p_value,
    crossed_bootstrap_diff,
    explanation_shift_test,
    holm_adjust,
    joint_verdict,
    spearman_rank,
    tost_equivalence,
)


# ----------------------------------------------------------------------------- data
def load_frame(cfg: ExperimentConfig, synthetic: bool = False, synthetic_n: int = 8_000) -> pd.DataFrame:
    """Read the data (real CSV or synthetic) and return the cleaned, dated frame (target + issue date kept)."""
    if synthetic:
        from .synthetic import make_synthetic_lendingclub
        raw = make_synthetic_lendingclub(synthetic_n, seed=0)
    else:
        from .config import USECOLS
        if not cfg.csv_path.exists():
            raise FileNotFoundError(
                f"{cfg.csv_path} not found. See data/README.md, or run with --synthetic."
            )
        raw = pd.read_csv(cfg.csv_path, usecols=USECOLS, low_memory=False)
    return prepare_frame(raw, cfg.sample_size, cfg.max_issue_year, cfg.drop_lender_assigned)


def split_frame(frame: pd.DataFrame, cfg: ExperimentConfig):
    """Chronological split of a dated frame -> (X_train, y_train, X_test, y_test), sparse features dropped."""
    train, test = temporal_split(frame, DATE_COL, cfg.test_fraction)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    # Coverage is measured on TRAINING rows only, then applied to both sides.
    sparse = find_sparse_features(X_train, cfg.min_feature_coverage)
    if sparse:
        warnings.warn(f"Dropping features observed for < {cfg.min_feature_coverage:.0%} of training loans: {sparse}")
        X_train, X_test = X_train.drop(columns=sparse), X_test.drop(columns=sparse)
    return X_train, y_train, X_test, y_test


def load_splits(cfg: ExperimentConfig, synthetic: bool = False, synthetic_n: int = 8_000):
    """Return (X_train, y_train, X_test, y_test) from a chronological split."""
    return split_frame(load_frame(cfg, synthetic, synthetic_n), cfg)


# ----------------------------------------------------------------------------- grid
def _seed_subsample(X, y, seed: int, cfg: ExperimentConfig):
    n = int(len(X) * cfg.seed_train_frac)
    if cfg.seed_train_size:
        n = min(n, cfg.seed_train_size)
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return X.iloc[idx], y.iloc[idx]


def run_seed_grid(splits, cfg: ExperimentConfig, verbose: bool = True) -> dict:
    """Fit every model for every seed on the same test rows and explain it on one fixed sample.

    Seed-to-seed variation comes from (a) a different random training subsample and
    (b) the model's own random_state. Test rows and explanation rows never change.
    """
    X_train, y_train, X_test, y_test = splits
    models = [m for m in cfg.models if m in available_models()]
    skipped = [m for m in cfg.models if m not in models]
    if skipped:
        warnings.warn(f"Skipping models with missing optional dependencies: {skipped}")

    explain_rows = X_test.sample(min(cfg.explain_size, len(X_test)), random_state=0)
    explain_y = y_test.loc[explain_rows.index]
    background = X_train.sample(min(cfg.background_size, len(X_train)), random_state=0)
    threshold = float(y_train.mean())  # decision threshold at the base rate (models are not re-weighted)

    metric_rows, importance_rows = [], []
    probs: dict[str, list] = {m: [] for m in models}
    fitted, last_times = {}, {}

    for name in models:
        for seed in cfg.seeds:
            X_s, y_s = _seed_subsample(X_train, y_train, seed, cfg)
            pipe = make_pipeline(name, X_s, seed, cfg.class_weight, y_s)

            t0 = time.perf_counter()
            pipe.fit(X_s, y_s)
            fit_time = time.perf_counter() - t0
            t0 = time.perf_counter()
            p_test = pipe.predict_proba(X_test)[:, 1]
            pred_time = time.perf_counter() - t0
            p_train = pipe.predict_proba(X_s)[:, 1]

            test_m = evaluate(y_test, p_test, threshold)
            train_auc = roc_auc_score(y_s, p_train)
            metric_rows.append({
                "Model": name, "Seed": seed, **{k: test_m[k] for k in ("roc_auc", "pr_auc", "ks", "brier", "ece")},
                "precision": test_m["precision"], "recall": test_m["recall"],
                "train_roc_auc": train_auc, "generalization_gap": train_auc - test_m["roc_auc"],
                "train_time": fit_time, "predict_time": pred_time,
            })
            probs[name].append(p_test)

            imp, used = global_importance(pipe, explain_rows, explain_y, background,
                                          method=cfg.explanation_method, seed=seed)
            for feat, val in imp.items():
                importance_rows.append({"Model": name, "Seed": seed, "Feature": feat,
                                        "Importance": float(val), "Method": used})
            fitted[name] = pipe
            last_times[name] = (fit_time, pred_time)
            if verbose:
                print(f"  {name:<20} seed {seed:<3} PR-AUC {test_m['pr_auc']:.4f}  ROC-AUC {test_m['roc_auc']:.4f}"
                      f"  fit {fit_time:5.1f}s  expl={used}")

    seed_metrics = pd.DataFrame(metric_rows)
    complexity = complexity_table(
        fitted,
        {m: seed_metrics.query("Model == @m")["train_time"].mean() for m in models},
        {m: seed_metrics.query("Model == @m")["predict_time"].mean() for m in models},
    )
    return {
        "seed_metrics": seed_metrics,
        "importances": pd.DataFrame(importance_rows),
        "probs": {m: np.vstack(v) for m, v in probs.items()},
        "y_test": y_test.to_numpy(),
        "complexity": complexity,
        "fitted": fitted,  # last seed's fitted pipeline per model
        "models": models,
        "threshold": threshold,  # base-rate decision threshold used for threshold metrics
        "X_explain": explain_rows, "y_explain": explain_y, "X_background": background,
    }


# ----------------------------------------------------------------------------- robustness
def run_robustness(grid: dict, splits, frame: pd.DataFrame, cfg: ExperimentConfig, verbose: bool = True) -> dict:
    """Drift (PSI), sensitivity, stress and chronological backtest for every model.

    Sensitivity and stress use each model's last fitted pipeline on a fixed test sample; the
    backtest refits each model for every later year using only earlier years.
    """
    X_train, _, X_test, _ = splits
    sample = X_test.sample(min(cfg.robustness_rows, len(X_test)), random_state=0)
    scenarios = {k: {f: m for f, m in v.items() if f in sample.columns} for k, v in DEFAULT_STRESS.items()}

    stress_rows, sens_rows = [], []
    for name, pipe in grid["fitted"].items():
        predict = lambda d, p=pipe: p.predict_proba(d)[:, 1]
        s = stress_test(predict, sample, scenarios)
        s.insert(0, "Model", name)
        stress_rows.append(s)
        for feature in cfg.sensitivity_features:
            if feature in sample.columns:
                t = sensitivity_test(predict, sample, feature)
                t.insert(0, "Model", name)
                sens_rows.append(t)

    keep = list(X_train.columns) + [TARGET, DATE_COL]
    backtests = []
    for name in grid["models"]:
        if verbose:
            print(f"  backtest {name}")
        bt = chronological_backtest(
            lambda X, y, n=name: make_pipeline(n, X, 0, cfg.class_weight, y), frame[keep],
            first_test_year=cfg.backtest_first_year, min_train=cfg.backtest_min_train,
            min_test=cfg.backtest_min_test, max_train=cfg.backtest_max_train)
        if not bt.empty:
            bt.insert(0, "Model", name)
            backtests.append(bt)

    return {
        "psi": feature_psi(X_train, X_test),
        "stress": pd.concat(stress_rows, ignore_index=True),
        "sensitivity": pd.concat(sens_rows, ignore_index=True) if sens_rows else pd.DataFrame(),
        "backtest": pd.concat(backtests, ignore_index=True) if backtests else pd.DataFrame(),
    }


# ----------------------------------------------------------------------------- hypotheses
def _importance_matrix(importances: pd.DataFrame, model: str) -> pd.DataFrame:
    sub = importances[importances["Model"] == model]
    return sub.pivot(index="Seed", columns="Feature", values="Importance").fillna(0.0)


def run_hypothesis_tests(grid: dict, cfg: ExperimentConfig, verbose: bool = True) -> dict:
    """H1, H2 and H3 over all model pairs, with Holm correction within each family."""
    models = grid["models"]

    method_by_run = (
      grid["importances"]
      .groupby(["Model", "Seed"])["Method"]
      .agg(lambda values: set(values))
    )
    
    # Ensure H2 compares explanations generated by one consistent method.
    
    expected_runs = pd.MultiIndex.from_product(
        [models, cfg.seeds],
        names=["Model", "Seed"]
    )
  
    missing_runs = expected_runs.difference(method_by_run.index)

    if len(missing_runs) > 0:
        raise ValueError(
            "Missing explanation results for model/seed runs: "
            f"{missing_runs.tolist()}"
        )

    inconsistent_runs = method_by_run[
        method_by_run.apply(lambda methods: len(methods) != 1)
    ]


    if not inconsistent_runs.empty:
        raise ValueError(
            "Multiple explanation methods detected within model/seed runs: "
            f"{inconsistent_runs.to_dict()}"
        )

    methods_used = set(grid["importances"]["Method"].unique())

    if len(methods_used) != 1:
        raise ValueError(
            "H2 requires a consistent explanation method across all models "
            f"and seeds. Found: {sorted(methods_used)}. "
            "Check SHAP failures or rerun with explanation_method='permutation'."
        )

    actual_method = next(iter(methods_used))

    if actual_method != cfg.explanation_method:
        raise ValueError(
            f"Requested '{cfg.explanation_method}' explanations, "
            f"but the experiment produced '{actual_method}'."
        )

    pairs = list(itertools.combinations(models, 2))
    if not pairs:
        raise ValueError("Need at least two models to compare")

    # H1: performance equivalence on PR-AUC
    h1 = []
    for i, (a, b) in enumerate(pairs):
        if verbose:
            print(f"  H1 bootstrap {a} vs {b}")
        res = crossed_bootstrap_diff(grid["y_test"], grid["probs"][a], grid["probs"][b],
                                     "pr_auc", cfg.n_boot, seed=cfg.seeds[0] + i)
        tost = tost_equivalence(res["boot"], cfg.pr_auc_margin, cfg.alpha / len(pairs))
        h1.append({"Model_A": a, "Model_B": b, "PR_AUC_diff": res["observed"],
                   "ci90_low": tost["ci_low"], "ci90_high": tost["ci_high"],
                   "p_value": bootstrap_p_value(res["boot"]), "Verdict": tost["verdict"],
                   "Equivalent": tost["equivalent"], "Margin": cfg.pr_auc_margin})
    h1 = pd.DataFrame(h1)
    h1["p_holm"] = holm_adjust(h1["p_value"])

    # H2: explanation shift beyond seed variation
    h2 = []
    imps = {m: _importance_matrix(grid["importances"], m) for m in models}
    for i, (a, b) in enumerate(pairs):
        for kind in ("membership", "ranking"):
            res = explanation_shift_test(imps[a], imps[b], kind, cfg.top_k, cfg.n_boot_explain,
                                         cfg.alpha, seed=cfg.seeds[0] + i)
            h2.append({"Model_A": a, "Model_B": b, **res})
    h2 = pd.DataFrame(h2)
    h2["p_holm"] = h2.groupby("kind")["p_value"].transform(lambda s: holm_adjust(s.to_numpy()))
    h2["Supported"] = (h2["p_holm"] < cfg.alpha) & (h2["theta"] > 0)

    # H3: equivalent performance, different explanations
    h3 = []
    for _, row in h1.iterrows():
        sel = h2[(h2.Model_A == row.Model_A) & (h2.Model_B == row.Model_B)].set_index("kind")["Supported"]
        mem, rank = bool(sel["membership"]), bool(sel["ranking"])
        h3.append({"Model_A": row.Model_A, "Model_B": row.Model_B, "H1_equivalent": bool(row.Equivalent),
                   "H2_membership": mem, "H2_ranking": rank,
                   "H3_supported": joint_verdict(bool(row.Equivalent), mem, rank)})
    return {"h1": h1, "h2": h2, "h3": pd.DataFrame(h3)}


def summarise(grid: dict) -> pd.DataFrame:
    """One row per model: seed means plus within-model explanation stability."""
    sm = grid["seed_metrics"]
    summary = sm.groupby("Model").agg(
        roc_auc=("roc_auc", "mean"), pr_auc=("pr_auc", "mean"), brier=("brier", "mean"),
        ece=("ece", "mean"), generalization_gap=("generalization_gap", "mean"),
        seed_std_pr_auc=("pr_auc", "std"),
    ).reindex(grid["models"])
    stability = {}
    for m in grid["models"]:
        mat = _importance_matrix(grid["importances"], m)
        rhos = [spearman_rank(mat.iloc[i], mat.iloc[j])
                for i, j in itertools.combinations(range(len(mat)), 2)]
        stability[m] = float(np.nanmean(rhos)) if rhos else np.nan
    summary["explanation_stability"] = pd.Series(stability)
    return summary


def save_tables(out_dir: Path, grid: dict, tests: dict, summary: pd.DataFrame, scorecard: pd.DataFrame,
                robustness: dict | None = None) -> None:
    """Write every result table as CSV (robustness tables as robustness_<name>.csv)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (robustness or {}).items():
        table.to_csv(out_dir / f"robustness_{name}.csv", index=False)
    grid["seed_metrics"].to_csv(out_dir / "seed_metrics.csv", index=False)
    grid["importances"].to_csv(out_dir / "importances.csv", index=False)
    grid["complexity"].to_csv(out_dir / "complexity.csv")
    summary.to_csv(out_dir / "model_summary.csv")
    scorecard.to_csv(out_dir / "scorecard.csv")
    for key in ("h1", "h2", "h3"):
        tests[key].to_csv(out_dir / f"{key}_results.csv", index=False)


def run_all(cfg: ExperimentConfig, synthetic: bool = False, verbose: bool = True) -> dict:
    frame = load_frame(cfg, synthetic)
    splits = split_frame(frame, cfg)
    if verbose:
        X_train, y_train, X_test, y_test = splits
        print(f"train {len(X_train):,} rows (default {y_train.mean():.2%}) | "
              f"test {len(X_test):,} rows (default {y_test.mean():.2%})")
    grid = run_seed_grid(splits, cfg, verbose)
    tests = run_hypothesis_tests(grid, cfg, verbose)
    summary = summarise(grid)
    scorecard = build_scorecard(summary, grid["complexity"])
    robustness = run_robustness(grid, splits, frame, cfg, verbose) if cfg.run_robustness else {}
    save_tables(cfg.output_dir, grid, tests, summary, scorecard, robustness)
    return {"grid": grid, "tests": tests, "summary": summary, "scorecard": scorecard,
            "robustness": robustness}


def run_feature_set_comparison(cfg: ExperimentConfig, synthetic: bool = False, verbose: bool = True,
                               figures: bool = False) -> dict:
    """Run the experiment WITHOUT and WITH the lender-assigned int_rate / sub_grade.

    Each run is saved in its own sub-folder (``no_lender_score`` / ``with_lender_score``) and a
    side-by-side ``feature_set_comparison.csv`` is written to ``cfg.output_dir``. The gap between
    the two shows how much of each model's performance is just the lender's own score. With
    ``figures=True`` the full figure set is also written per run, plus a comparison chart.
    """
    results = {}
    for label, drop in (("no_lender_score", True), ("with_lender_score", False)):
        if verbose:
            print(f"\n##### feature set: {label} #####")
        sub_cfg = replace(cfg, drop_lender_assigned=drop, output_dir=cfg.output_dir / label)
        results[label] = run_all(sub_cfg, synthetic=synthetic, verbose=verbose)
        if figures:
            from .visualization import save_all
            save_all(results[label], cfg.figure_dir / label)
    comparison = pd.concat({k: v["summary"] for k, v in results.items()}, names=["Feature_set", "Model"])
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(cfg.output_dir / "feature_set_comparison.csv")
    results["comparison"] = comparison
    if figures:
        import matplotlib.pyplot as plt
        from .visualization import plot_feature_set_comparison
        plt.close(plot_feature_set_comparison(comparison, cfg.figure_dir / "feature_set_comparison.png"))
    return results


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated data (pipeline check only)")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=None, help="stratified sample size (default 200000)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--explain-method", choices=["shap", "permutation"], default="shap")
    ap.add_argument("--class-weight", choices=["balanced"], default=None)
    ap.add_argument("--with-lender-score", action="store_true",
                    help="KEEP int_rate and sub_grade (default drops them: lender-independent model)")
    ap.add_argument("--both-feature-sets", action="store_true",
                    help="run twice (without and with the lender score) and compare")
    ap.add_argument("--out", type=Path, default=None, help="folder for result tables")
    ap.add_argument("--figure-dir", type=Path, default=None, help="folder for figures")
    ap.add_argument("--no-figures", action="store_true", help="skip writing PNG figures")
    ap.add_argument("--no-robustness", action="store_true", help="skip drift / stress / backtest (faster)")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")  # figures go to files; never open windows from the command line
    cfg = ExperimentConfig()
    updates = {"seeds": tuple(range(args.seeds)), "explanation_method": args.explain_method,
               "class_weight": args.class_weight, "drop_lender_assigned": not args.with_lender_score,
               "run_robustness": not args.no_robustness}
    if args.figure_dir: updates["figure_dir"] = args.figure_dir
    if args.csv: updates["csv_path"] = args.csv
    if args.sample: updates["sample_size"] = args.sample
    if args.models: updates["models"] = tuple(args.models)
    if args.n_boot: updates["n_boot"] = args.n_boot
    if args.out: updates["output_dir"] = args.out
    cfg = replace(cfg, **updates)

    if args.both_feature_sets:
        both = run_feature_set_comparison(cfg, synthetic=args.synthetic, figures=not args.no_figures)
        pd.set_option("display.width", 160)
        print("\nFeature-set comparison (mean over seeds)\n", both["comparison"].round(4))
        print(f"\nTables written to {cfg.output_dir}" + ("" if args.no_figures else f"; figures to {cfg.figure_dir}"))
        return

    result = run_all(cfg, synthetic=args.synthetic)
    if not args.no_figures:
        from .visualization import save_all
        paths = save_all(result, cfg.figure_dir)
        print(f"\nWrote {len(paths)} figures to {cfg.figure_dir}")
    pd.set_option("display.width", 160)
    print("\nModel summary\n", result["summary"].round(4))
    print("\nH1 (PR-AUC equivalence)\n", result["tests"]["h1"].round(4).to_string(index=False))
    print("\nH3 (equivalent performance, different explanations)\n", result["tests"]["h3"].to_string(index=False))
    print(f"\nTables written to {cfg.output_dir}")


if __name__ == "__main__":
    main()
