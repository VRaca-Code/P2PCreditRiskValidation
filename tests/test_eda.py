"""Tests for src/eda.py: descriptive tables, effect sizes, rule validation and EDA figures.

Uses the shared synthetic LendingClub data. Checks that the label-definition table reports what
the cleaning step drops, that group tables add up, that effect sizes behave on a planted signal,
that a useless rule gets lift near 1, that missingness and explanation-comparison tables have the
promised columns, and that every EDA figure is written.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import TARGET, ExperimentConfig
from src.eda import (
    default_rate_by,
    eda_frame,
    eda_report,
    hypothesis_tests,
    label_definition_comparison,
    missingness_informativeness,
    numeric_default_correlation,
    pricing_vs_realised_risk,
    risk_rule_validation,
    univariate_vs_explanation,
    volume_by_year,
)


@pytest.fixture(scope="module")
def frame(raw_df):
    return eda_frame(raw_df)


def test_label_definitions_show_unresolved_loans(raw_df):
    t = label_definition_comparison(raw_df)
    assert t.loc[0, "Unresolved_loans_in_sample"] == 0
    assert t.loc[1, "Loans"] == len(raw_df) > t.loc[0, "Loans"]
    assert t.loc[1, "Unresolved_loans_in_sample"] > 0


def test_eda_frame_keeps_grade_and_adds_descriptive_columns(frame):
    assert {"grade", "year", "region", "income_band", "fico_band", "term_months", "emp_years"} <= set(frame.columns)
    assert set(frame[TARGET].unique()) <= {0, 1}


def test_default_rate_by_adds_up(frame):
    g = default_rate_by(frame, "grade", min_count=1)
    assert g["Loans"].sum() == len(frame)
    assert (g["ci95_low"] <= g["Default_rate"]).all() and (g["Default_rate"] <= g["ci95_high"]).all()
    assert np.isclose(g["Share_of_loans"].sum(), 1.0)


def test_default_rate_rises_with_grade_in_planted_data(frame):
    g = default_rate_by(frame, "grade", min_count=100).set_index("Group")["Default_rate"]
    assert g.iloc[-1] > g.iloc[0]


def test_volume_by_year_counts_all_loans(frame):
    assert volume_by_year(frame)["Loans"].sum() == len(frame)


def test_hypothesis_tests_report_effect_sizes(frame):
    t = hypothesis_tests(frame).set_index("Variable")
    assert t.loc["int_rate", "Effect_size"] > 0.3  # planted: defaults pay higher rates
    assert t.loc["int_rate", "Effect_label"] in {"small", "medium", "large"}
    assert {"Cohen's d", "Cramer's V", "epsilon^2"} <= set(t["Effect_measure"])
    assert (t["p_value"].between(0, 1)).all()


def test_effect_size_is_negligible_for_pure_noise():
    rng = np.random.default_rng(0)
    n = 4_000
    f = pd.DataFrame({TARGET: rng.integers(0, 2, n), "dti": rng.normal(size=n), "term_months": rng.choice([36, 60], n)})
    t = hypothesis_tests(f, numeric=("dti",), categorical=("term_months",)).set_index("Variable")
    assert t.loc["dti", "Effect_label"] == "negligible" and t.loc["term_months", "Effect_label"] == "negligible"


def test_useless_rule_has_lift_near_one():
    rng = np.random.default_rng(1)
    n = 20_000
    f = pd.DataFrame({TARGET: rng.integers(0, 2, n), "loan_amnt": rng.uniform(1_000, 30_000, n),
                      "annual_inc": rng.uniform(20_000, 120_000, n), "emp_years": rng.integers(0, 11, n).astype(float)})
    t = risk_rule_validation(f)
    assert t["Lift"].between(0.9, 1.1).all()
    assert t["AUC_of_flag"].between(0.45, 0.55).all()


def test_rule_validation_columns(frame):
    t = risk_rule_validation(frame)
    assert {"Flagged_share", "Default_rate_flagged", "Lift", "Precision", "Recall", "AUC_of_flag"} <= set(t.columns)
    assert t["Flagged_share"].between(0, 1).all()


def test_pricing_table_has_one_row_per_grade(frame):
    t = pricing_vs_realised_risk(frame)
    assert t["Loans"].sum() == len(frame)
    assert t["Mean_interest_rate"].is_monotonic_increasing  # planted pricing rises with grade


def test_missingness_table_ignores_complete_columns_and_derived_copies(frame):
    t = missingness_informativeness(frame)
    assert "loan_amnt" not in set(t["Feature"]) and "emp_years" not in set(t["Feature"])
    assert {"Gap_pct_points", "Missing_share"} <= set(t.columns)


def test_univariate_vs_explanation_aligns_features(frame):
    corr = numeric_default_correlation(frame)  # uses EDA names (term_months, emp_years); aliases map them
    feats = corr["Feature"].head(6).tolist()
    imps = pd.DataFrame([{"Model": m, "Seed": s, "Feature": f, "Importance": 1.0 / (i + 1)}
                         for m in ("A", "B") for s in (0, 1) for i, f in enumerate(feats)])
    out = univariate_vs_explanation(corr, imps)
    assert set(out["Model"]) == {"A", "B"} and {"Rank_gap", "Model_rank", "Univariate_rank"} <= set(out.columns)


def test_eda_report_and_figures_written(raw_df, tmp_path):
    from src.visualization import save_eda_figures
    report = eda_report(raw_df, ExperimentConfig(), tmp_path / "t", verbose=False)
    assert (tmp_path / "t" / "eda_hypothesis_tests.csv").exists()
    assert not (tmp_path / "t" / "eda_frame.csv").exists()
    paths = save_eda_figures(report, tmp_path / "f")
    assert len(paths) == 10 and all(p.exists() and p.stat().st_size > 5_000 for p in paths)


def test_outlier_summary_ignores_fico_bounds_and_flags_heavy_tails(frame):
    from src.eda import outlier_summary
    o = outlier_summary(frame).set_index("Feature")
    assert "fico_range_low" not in o.index and "fico_range_high" not in o.index
    assert o.loc["revol_bal", "Outside_1.5_IQR_share"] > o.loc["loan_amnt", "Outside_1.5_IQR_share"]


def test_data_quality_figures_written(raw_df, tmp_path):
    from src.data_quality import data_quality_report
    from src.visualization import save_data_quality_figures
    report = data_quality_report(raw_df, ExperimentConfig(), verbose=False)
    paths = save_data_quality_figures(report, tmp_path, max_issue_year=2016)
    assert {p.name for p in paths} == {"dq_status_breakdown.png", "dq_attrition.png", "dq_missingness.png", "dq_censoring.png"}
    assert all(p.stat().st_size > 5_000 for p in paths)
