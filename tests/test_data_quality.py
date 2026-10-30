"""Tests for src/data_quality.py: the attrition table, status breakdown and missingness report.

Checks that row counts shrink step by step in the documented order, that unresolved loans
are reported as dropped, and that the train/test rows add up to the filtered total.
"""

import numpy as np

from src.config import ExperimentConfig
from src.data_quality import (
    attrition_table,
    data_quality_report,
    default_rate_by_year,
    missingness_report,
    status_breakdown,
)
from src.preprocessing import prepare_frame


def test_status_breakdown_marks_unresolved_statuses_as_dropped(raw_df):
    table = status_breakdown(raw_df).set_index("Raw_status")
    assert table.loc["Current", "Kept"] == False  # noqa: E712
    assert table.loc["Fully Paid", "Kept"] == True  # noqa: E712
    assert table.loc["Charged Off", "Encoded_as"] == 1
    assert np.isclose(table["Share"].sum(), 1.0)


def test_attrition_counts_are_monotone_and_split_adds_up(raw_df):
    cfg = ExperimentConfig(sample_size=None, max_issue_year=2015)
    table = attrition_table(raw_df, cfg).set_index("Step")
    steps = ["Raw rows", "Resolved loans only", "After stratified sample", "Valid issue date", "Issued <= 2015"]
    rows = table.loc[steps, "Rows"].to_numpy()
    assert (np.diff(rows) <= 0).all()
    assert rows[0] == len(raw_df) and rows[1] < rows[0]  # "Current"/"Late" loans removed
    split_total = table.loc["Training set (earlier months)", "Rows"] + table.loc["Test set (latest months)", "Rows"]
    assert split_total == table.loc["Issued <= 2015", "Rows"]


def test_attrition_vintage_filter_removes_late_years(raw_df):
    early = attrition_table(raw_df, ExperimentConfig(sample_size=None, max_issue_year=2013)).set_index("Step")
    late = attrition_table(raw_df, ExperimentConfig(sample_size=None, max_issue_year=2016)).set_index("Step")
    assert early.loc["Issued <= 2013", "Rows"] < late.loc["Issued <= 2016", "Rows"]


def test_missingness_and_default_rate_by_year(raw_df):
    frame = prepare_frame(raw_df)
    miss = missingness_report(frame.drop(columns=["loan_status", "issue_d"]))
    assert miss["missing_rate"].is_monotonic_decreasing
    assert miss.loc["mort_acc", "missing_rate"] > 0.03  # the generator blanks ~8% of mort_acc
    by_year = default_rate_by_year(frame)
    assert list(by_year.columns) == ["Issue_year", "Loans", "Default_rate"]
    assert by_year["Loans"].sum() == len(frame)


def test_full_report_returns_and_saves_every_table(raw_df, tmp_path):
    report = data_quality_report(raw_df, ExperimentConfig(sample_size=None), tmp_path, verbose=False)
    assert {"status_breakdown", "attrition", "default_rate_by_year", "missingness",
            "sparse_features"} == set(report)
    assert len(list(tmp_path.glob("data_quality_*.csv"))) == 5
