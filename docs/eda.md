# Exploratory data analysis (what it is, where the ideas come from)

What this file does: explains each analysis in `src/eda.py`, whether the idea comes from the public
LendingClub / Bondora notebooks or is new to this project, and which weaknesses of those notebooks
we deliberately avoid. Run it with `python -m src.eda --csv <file> --out outputs/tables/eda --figure-dir outputs/figures/eda`.

The focus is LendingClub. Bondora notebooks were used for *methods* only (no Bondora data is used).

| Analysis | Output | Idea from | What we changed |
|---|---|---|---|
| Label-definition comparison | `eda_label_definitions.csv` | new | Shows default rate and sample size under our label vs the common notebook label (Current = good, Late / Grace = bad). |
| Loans and default rate by year | `eda_volume_by_year.csv` | LendingClub notebooks | Reads a falling late-year rate as possible right-censoring, as `data_cleaning.md` explains. |
| Default rate by grade, purpose, term, region, income band, employment length, home ownership, FICO band, verification | `eda_default_rate_groups.csv` | LendingClub notebooks | 95% Wilson intervals on every rate; groups under 50 loans are dropped; states grouped into 4 regions. |
| Defaulted vs paid tests | `eda_hypothesis_tests.csv` | Bondora EDA notebook (Welch t, ANOVA) | Every p-value comes with an effect size (Cohen's d, Cramér's V, epsilon-squared) and a size label. With 100k+ loans nearly every p-value is tiny, so effect size is what matters. Kruskal-Wallis replaces ANOVA because interest rates are not normal. |
| Risk-rule validation | `eda_risk_rule_validation.csv` | Bondora EDA notebook (`IsRisky` rule) | That notebook never checked its rule against defaults (it lists this as future work). We report lift, precision, recall and AUC of the flag. |
| Pricing vs realised risk | `eda_pricing_vs_realised_risk.csv` | LendingClub + Bondora (rate vs rating) | Compares interest charged with defaults actually seen, per grade. Naive: ignores recoveries and timing, so not a profit measure. |
| Missingness as signal | `eda_missingness_informativeness.csv` | new (the notebooks fill with 0 / codes) | Default rate when a feature is missing vs observed. A large gap argues against silently filling it. |
| Univariate vs explanation ranking | `univariate_vs_explanation` (in the notebook) | new | Compares a one-feature-at-a-time ranking with the SHAP ranking of each model. Features where they disagree are the ones models use through interactions. |
| Numeric correlation with default | `eda_numeric_correlation.csv` | LendingClub notebooks | Spearman instead of Pearson (robust to skew); target-only view instead of a full heatmap. |

## Weaknesses of the source notebooks that we avoid

1. **Label.** Late and grace-period loans counted as defaults; `Current` counted as good (LendingClub). Bondora's predictor labels a loan non-default if it has no `DefaultDate`, even when it is still running.
2. **Imputation before splitting, and sentinel fills.** NaN filled with 0 or arbitrary codes on the whole data; `mths_since_last_delinq = 0` means "just delinquent", not "never". Here, imputation is inside the pipeline and fitted on training data only.
3. **Truncating casts.** Bondora's predictor casts `Interest` and `DebtToIncome` to int, which drops decimals.
4. **Bug.** `x.fillna(x.mode)` in the LendingClub imputation passes the method itself instead of `x.mode()`.
5. **Evaluation.** Random splits and accuracy on about 8% defaults.
6. **Protected attributes.** Gender, age and marital status are used as model features in the Bondora predictor. This project uses none of them.

EDA keeps `grade` and `int_rate` because it is descriptive. The models still drop the lender-assigned score by default.

## Figures

Data cleaning (`python -m src.data_quality --csv <file> --out outputs/tables --figure-dir outputs/figures/data_quality`):
`dq_attrition.png` (loans left after each step), `dq_status_breakdown.png` (every raw loan status, kept vs dropped),
`dq_missingness.png` (feature coverage with the 70% cut-off), `dq_censoring.png` (default rate by year with the dropped late vintages shaded).

EDA (`python -m src.eda ... --figure-dir outputs/figures/eda`), 10 figures: label definitions, year overview, default rate by group,
effect sizes, univariate correlation, pricing vs realised risk, missingness signal, outlier shares, distributions of paid vs defaulted
loans, and the correlation heatmap between numeric features.

Outliers are measured (`eda_outlier_summary.csv`) but not removed: income and revolving balance are log-transformed and standardised,
and the tree models do not depend on scale. Check this table on the real data before keeping that choice.
