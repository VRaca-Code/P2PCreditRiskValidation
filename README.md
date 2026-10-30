# P2PCreditRiskValidation

**Do machine-learning credit models that perform the same also explain themselves the same?**
A validation-style comparison of Logistic Regression, Random Forest, XGBoost and (optionally) a
neural network and a Kolmogorov-Arnold Network on LendingClub peer-to-peer loans. It measures
performance, calibration, complexity and **explanation stability**, and tests whether differences
are real or just seed noise.

> **Status (v0.2.0):** the code and tests are complete and run on synthetic data. Results on the
> real LendingClub file are not included yet: run the pipeline and add your numbers to the
> [Results](#results) section. Nothing in this README is a finding.

## What this file is

The project overview: the research question and hypotheses, the design choices, how to run
everything, a map of the repository, and the limitations. Every other file starts with its own
short description of what it does.

## Research question and hypotheses

Similar predictive performance does not guarantee similar explanations, so performance and
explanation stability are tested **separately**, then jointly.

| | Hypothesis | Test |
|---|---|---|
| **H1** | Model pairs differ in test PR-AUC by more than a practical margin (default ±0.01). | Crossed seed × observation bootstrap; **TOST equivalence**; Holm correction. Verdict per pair: *equivalent*, *different* or *inconclusive*. |
| **H2** | Top-10 SHAP **membership** (Jaccard) and **ranking** (Spearman) shift between models by more than they vary across seeds *within* a model. | One-sided bootstrap over seeds of θ = cross-model distance − within-model distance; Holm correction. |
| **H3** | Some pair is performance-equivalent (H1) **and** explained differently (H2). | Joint classification per pair. |
| **H4** *(exploratory)* | The KAN sits between logistic regression and the DNN on the performance / transparency trade-off. | Scorecard + learned-function plots (not yet automated). |

## Design choices

- **Lender-independent headline model.** `int_rate` and `sub_grade` are LendingClub's own underwriting outputs, so by default they are **dropped** (a lender building its own model would not have them a priori). `--both-feature-sets` runs the experiment with and without them and writes a side-by-side comparison.
- **Chronological split.** Train on earlier origination months, test on the latest 20 %. Loans issued after `max_issue_year` (default 2016) are dropped because their outcomes are right-censored.
- **Seeds as the noise baseline.** Each of 10 seeds fits on a different random training subsample (≤ 50 000 rows) with a different model `random_state`. Test and explanation rows are fixed.
- **One imbalance treatment for every model.** Default is *none* (natural default rate), so Brier / ECE compare intrinsic calibration. `--class-weight balanced` re-weights all models; recalibrate afterwards (`src/calibration.py`).
- **No leakage.** Imputation, log transforms, scaling, one-hot encoding and the sparse-feature filter use training rows only (tested).
- **Features.** FICO score (midpoint of the published band) is included; `annual_inc` and `revol_bal` are log-transformed inside the pipeline; features observed for < 70 % of training loans are dropped.
- **Explanations on original features.** SHAP values of one-hot dummies are summed per row back to their source column before taking absolute values, so rankings are comparable across models and with permutation importance.
- **No expert-assigned interpretability score.** The scorecard uses a measured *explanation stability* instead. Scores are relative (min-max across the compared models), so keep the raw tables beside them.

## Related work

*Aste et al. (UCL), "P2P Loan acceptance and default prediction with Artificial Intelligence", arXiv 1907.01800 (preprint).*
Same Kaggle source. It also keeps only resolved loans, cuts recent vintages because of censoring, excludes the lender's grade and interest rate on the argument that they are not available a priori, and uses FICO and log-income. It reports test AUC of about 0.68 to 0.69 for logistic regression and its small DNNs without lender features, i.e. roughly equal AUC, with the DNNs showing higher recall on defaults at a 0.5 threshold. This repo adds what that paper does not report: threshold-free and PR-AUC metrics, calibration, multiple seeds, formal equivalence tests, and explanation-stability tests.

## Quick start

```bash
pip install -e ".[dev]"                 # core: LR + Random Forest, stats, plots
pip install -e ".[boost,explain]"       # + XGBoost and SHAP (tree explainers)
pip install -e ".[neural]"              # + PyTorch for the DNN and KAN

pytest -q                               # tests (optional-dependency tests skip if missing)

# Pipeline check on generated data (no download; results are meaningless)
python -m src.experiments --synthetic --seeds 3 --n-boot 100
python -m src.data_quality --synthetic

# Real analysis: place the Kaggle CSV at data/raw/ (see data/README.md)
python -m src.data_quality --csv data/raw/accepted_2007_to_2018Q4.csv --out outputs/tables
python -m src.experiments --both-feature-sets --models "Logistic Regression" "Random Forest" XGBoost
python -m src.experiments --models "Logistic Regression" "Random Forest" XGBoost DNN KAN
```

Useful flags: `--sample 200000`, `--seeds 10`, `--explain-method shap|permutation`,
`--class-weight balanced`, `--with-lender-score`, `--both-feature-sets`, `--out DIR`. The notebook
`notebooks/P2PCreditRiskValidation.ipynb` walks through the same steps with plots.

Runtime note: the H1 bootstrap re-scores every seed on the full test set for each pair, so it
scales with `n_boot × seeds × pairs`. Lower `--n-boot` while developing.

## Repository layout

```
src/
  config.py           paths, columns, ExperimentConfig
  preprocessing.py    cleaning, features, temporal split, leakage-safe preprocessor
  data_quality.py     attrition table, status breakdown, missingness, default rate by year
  models.py           factory for LR / RF / XGBoost / DNN / KAN (optional deps are lazy)
  neural.py, torch_nets.py   sklearn-style wrapper, MLP and KAN layers (PyTorch)
  metrics.py          ROC/PR-AUC, KS, Brier, ECE, threshold tables
  calibration.py      reliability tables, sigmoid / isotonic recalibration
  complexity.py       one definition of structural complexity
  interpretability.py coefficients, permutation importance, SHAP on original features
  robustness.py       PSI, CV, sensitivity, stress, chronological backtest
  stats.py            bootstrap, TOST, Holm, Jaccard, Spearman, explanation-shift test
  scoring.py          relative scorecard, Pareto dominance
  experiments.py      seed grid, H1-H3, feature-set comparison, CLI
  visualization.py    figures (colour follows the model, direct labels)
  synthetic.py        LendingClub-shaped generator for tests and demos
docs/data_cleaning.md step-by-step cleansing decisions and reasons
tests/                pytest suite (one file per module)
outputs/{tables,figures}   generated, git-ignored
```

## Outputs

`outputs/tables/`: `seed_metrics.csv`, `importances.csv`, `h1_results.csv`, `h2_results.csv`,
`h3_results.csv`, `model_summary.csv`, `complexity.csv`, `scorecard.csv`, and with
`--both-feature-sets` the sub-folders `no_lender_score/`, `with_lender_score/` plus
`feature_set_comparison.csv`. `src.data_quality --out DIR` writes the `data_quality_*.csv` audit tables.
`outputs/tables/robustness_{psi,stress,sensitivity,backtest}.csv` hold the drift, stress, sensitivity and
chronological-backtest results (skip with `--no-robustness`).
`outputs/figures/` (written automatically; skip with `--no-figures`), up to 20 PNGs:
- **Hypotheses:** equivalence forest (H1), explanation shift (H2), SHAP rank-agreement heatmap, per-seed boxplots
- **Performance:** ROC, precision-recall, calibration, confusion matrices, precision/recall vs threshold
- **Explanations:** importance bars per model, logistic-regression coefficients
- **Robustness:** PSI drift, sensitivity lines, stress scenarios, chronological backtest
- **Trade-offs:** scorecard radar, performance vs complexity, 2x2 trade-off dashboard, DNN/KAN training curves
  (only when those models run), with vs without lender score (`--both-feature-sets`)

## Results

*Add after running on the full data: the data attrition table, the H1 verdict table, the H2 shift
figure, the pairs flagged by H3, and the with / without lender-score comparison, with the exact
command and commit used.*

## Limitations

- LendingClub is public, observational data, not a bank's regulated portfolio, and loans from the latest vintages are right-censored even after the year cut-off. Only accepted loans have outcomes (selection bias).
- SHAP explains model behaviour, not causal effects. Tree SHAP is in log-odds for XGBoost and in probability for scikit-learn forests, so only normalised shares and rank statistics are compared across models.
- With few models the relative scorecard is coarse (two models score 0 / 100 on every axis). Stress scenarios are illustrative, not macro-calibrated.
- Seed variation comes from training-subsample and model randomness, not from re-drawing the test set; the H1 bootstrap covers test-row noise separately.
- XGBoost, tree / gradient SHAP and the PyTorch DNN and KAN were written but not executed in the environment where this repo was built (no package index access); run `pytest` locally or in Colab to exercise them.
- H4 and KAN symbolic-formula extraction are not automated.

## License

MIT (see `LICENSE`; replace the placeholder name).
