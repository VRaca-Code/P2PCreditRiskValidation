"""Central configuration: constants and the ExperimentConfig dataclass.

What this file does
-------------------
* Defines the project-wide constants: target and date column names, the random seed,
  the list of LendingClub columns to load (``USECOLS``), which columns are assigned by
  the lender (``LENDER_ASSIGNED``) and which skewed columns get a log transform
  (``LOG_FEATURES``).
* Defines ``ExperimentConfig``, one object holding every tunable setting of an experiment
  (data cut-offs, models, seeds, explanation method, statistical margins, output folders).
* Nothing here is Colab-specific: all paths are relative to the repository root, and the
  data location can be overridden with the ``P2P_CSV_PATH`` environment variable.

Used by: preprocessing.py, experiments.py, data_quality.py, models.py and the notebook.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGET = "loan_status"  # after cleaning: 1 = default, 0 = fully paid
DATE_COL = "issue_d"
RANDOM_STATE = 42

# Columns known at loan origination (no post-origination payment fields).
# fico_range_low/high are combined into one `fico_score` feature in preprocessing.
USECOLS = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_title", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "issue_d", "loan_status", "purpose", "title",
    "addr_state", "dti", "earliest_cr_line", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "mort_acc", "pub_rec_bankruptcies",
    "fico_range_low", "fico_range_high",
]

# Fields produced by LendingClub's own underwriting model. They are legitimate
# origination-time inputs, but a lender building its OWN credit model would not have them
# a priori, and they make every model partly re-learn the lender's score.
LENDER_ASSIGNED = ["int_rate", "sub_grade"]

# Strongly right-skewed money columns: log1p-transformed inside the sklearn pipeline.
# The raw columns stay in the data so stress tests can still shock them multiplicatively.
LOG_FEATURES = ("annual_inc", "revol_bal")

ALL_MODELS = ("Logistic Regression", "Random Forest", "XGBoost", "DNN", "KAN")
DEFAULT_MODELS = ("Logistic Regression", "Random Forest", "XGBoost")


def _default_csv() -> Path:
    return Path(os.environ.get(
        "P2P_CSV_PATH", ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
    ))


@dataclass
class ExperimentConfig:
    """All settings of one experiment run. Override fields with ``dataclasses.replace``."""

    # data
    csv_path: Path = None  # type: ignore[assignment]
    sample_size: int | None = 200_000
    max_issue_year: int | None = 2016  # later vintages are right-censored (see docs/data_cleaning.md)
    test_fraction: float = 0.20  # last 20% of origination dates form the test set
    # True = headline run WITHOUT int_rate / sub_grade (lender-independent model);
    # False = include them. `run_feature_set_comparison` runs both.
    drop_lender_assigned: bool = True
    min_feature_coverage: float = 0.70  # drop features observed for < 70% of TRAINING loans

    # models and seeds
    models: tuple = DEFAULT_MODELS
    seeds: tuple = tuple(range(10))
    seed_train_frac: float = 0.80   # each seed fits on a random 80% of the training rows ...
    seed_train_size: int | None = 50_000  # ... capped at this many rows
    class_weight: str | None = None  # None everywhere, or "balanced" everywhere (never mixed)

    # explanations
    explanation_method: str = "shap"  # "shap" or "permutation"
    explain_size: int = 2_000
    background_size: int = 200
    top_k: int = 10

    # statistics
    pr_auc_margin: float = 0.01  # equivalence margin for H1 (TOST)
    alpha: float = 0.05
    n_boot: int = 500
    n_boot_explain: int = 2_000

    # robustness checks (drift, sensitivity, stress, chronological backtest)
    run_robustness: bool = True
    robustness_rows: int = 2_000        # test rows used for sensitivity and stress
    sensitivity_features: tuple = ("annual_inc", "dti", "revol_util")
    backtest_first_year: int = 2014     # first year used as a backtest test year
    backtest_min_train: int = 3_000
    backtest_min_test: int = 500
    backtest_max_train: int | None = 50_000

    output_dir: Path = ROOT / "outputs" / "tables"
    figure_dir: Path = ROOT / "outputs" / "figures"

    def __post_init__(self) -> None:
        if self.csv_path is None:
            self.csv_path = _default_csv()
        self.csv_path = Path(self.csv_path)
        self.output_dir = Path(self.output_dir)
        self.figure_dir = Path(self.figure_dir)
