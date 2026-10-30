"""Data cleaning, feature engineering, chronological splitting and the leakage-safe preprocessor.

What this file does
-------------------
Turns the raw LendingClub table into model-ready data in two clearly separated stages:

1. **Row-level steps** (no statistics are learned, so they are safe to run before splitting):
   ``clean_target`` keeps resolved loans and encodes default = 1; ``engineer_features``
   parses text fields (term, employment length, dates), merges the two FICO columns into one
   score and optionally drops lender-assigned fields; ``stratified_sample`` and
   ``filter_vintages`` control the sample size and remove right-censored late vintages.
2. **Learned steps** (fitted on training rows only, inside sklearn pipelines):
   ``build_preprocessor`` imputes, log-transforms skewed money columns, scales numerics and
   one-hot encodes categoricals. ``find_sparse_features`` flags columns observed for too few
   training loans so they can be dropped from both train and test.

It also provides ``temporal_split`` (train on earlier months, test on the latest) and
``source_feature_names`` (maps every transformed column back to its original feature, which
is what lets SHAP and permutation importance be compared feature by feature).

Used by: experiments.py, data_quality.py, models.py, interpretability.py, tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import DATE_COL, LENDER_ASSIGNED, LOG_FEATURES, RANDOM_STATE, TARGET

# Only observed final outcomes are kept; open loans have no label yet.
STATUS_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}


# ----------------------------------------------------------------------------- row-level steps
def clean_target(df: pd.DataFrame) -> pd.DataFrame:
    """Keep resolved loans and encode the target as 1 = default, 0 = fully paid."""
    mapped = df[TARGET].map(STATUS_MAP)
    out = df.loc[mapped.notna()].copy()
    out[TARGET] = mapped.loc[out.index].astype(int)
    return out


def engineer_features(df: pd.DataFrame, drop_lender_assigned: bool = False) -> pd.DataFrame:
    """Row-level transformations only (no statistics are learned here)."""
    out = df.copy()
    drop = ["emp_title", "title", "grade"]  # high-cardinality or redundant with sub_grade
    if drop_lender_assigned:
        drop += LENDER_ASSIGNED
    out = out.drop(columns=drop, errors="ignore")

    out["term"] = out["term"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)

    out["emp_length"] = (
        out["emp_length"].astype(str)
        .str.replace("< 1 year", "0", regex=False)
        .str.replace("10+ years", "10", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )

    earliest = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], format="%b-%Y", errors="coerce")
    out["earliest_cr_year"] = earliest.dt.year
    out["earliest_cr_month"] = earliest.dt.month
    out = out.drop(columns=["earliest_cr_line"], errors="ignore")

    # LendingClub publishes FICO as a 5-point band; its midpoint is the usable score.
    if {"fico_range_low", "fico_range_high"} <= set(out.columns):
        out["fico_score"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
        out = out.drop(columns=["fico_range_low", "fico_range_high"])

    out["home_ownership"] = out["home_ownership"].replace(["ANY", "NONE"], "OTHER")
    return out


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Sample about n rows while preserving the default rate."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    parts = []
    for _, part in df.groupby(TARGET):
        take = max(1, round(n * len(part) / len(df)))
        parts.append(part.sample(min(take, len(part)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def filter_vintages(df: pd.DataFrame, max_issue_year: int | None) -> pd.DataFrame:
    """Drop loans issued after `max_issue_year` (their outcomes are right-censored)."""
    if max_issue_year is None:
        return df
    return df[df[DATE_COL].dt.year <= max_issue_year].copy()


def prepare_frame(
    raw: pd.DataFrame,
    sample_size: int | None = None,
    max_issue_year: int | None = None,
    drop_lender_assigned: bool = False,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """raw LendingClub rows -> cleaned, engineered frame ready for temporal_split."""
    df = clean_target(raw)
    df = stratified_sample(df, sample_size, seed)
    df = engineer_features(df, drop_lender_assigned=drop_lender_assigned)
    df = df.dropna(subset=[DATE_COL])
    return filter_vintages(df, max_issue_year)


# ----------------------------------------------------------------------------- splitting
def temporal_split(
    df: pd.DataFrame, date_col: str = DATE_COL, test_fraction: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: train on earlier origination months, test on the latest ones.

    The cutoff is a whole month, so no origination month is shared by both sides.
    """
    d = df.dropna(subset=[date_col])
    ordered = d[date_col].sort_values(kind="stable")
    cut_pos = min(int(len(ordered) * (1 - test_fraction)), len(ordered) - 1)
    cutoff = ordered.iloc[cut_pos]
    return d[d[date_col] < cutoff].copy(), d[d[date_col] >= cutoff].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features exclude the target and the issue date (kept only for splitting)."""
    return df.drop(columns=[TARGET, DATE_COL]), df[TARGET]


def find_sparse_features(X: pd.DataFrame, min_coverage: float = 0.70) -> list[str]:
    """Columns observed (non-missing) for fewer than `min_coverage` of the rows of X.

    Call it on TRAINING data only and drop the result from both train and test, so the
    decision never looks at test rows.
    """
    coverage = X.notna().mean()
    return coverage[coverage < min_coverage].index.tolist()


# ----------------------------------------------------------------------------- learned steps
def _log1p_nonneg(x):
    """log(1 + x) with negatives clipped to 0 (module-level so pipelines can be pickled/cloned)."""
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Unfitted preprocessor.

    * ``log`` block: median-impute, log1p, standardise (skewed money columns)
    * ``num`` block: median-impute, standardise (other numerics)
    * ``cat`` block: mode-impute, one-hot (rare levels < 25 rows pooled into "infrequent")
    """
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    log_cols = [c for c in LOG_FEATURES if c in numeric]
    plain_cols = [c for c in numeric if c not in log_cols]

    def numeric_block(with_log: bool) -> Pipeline:
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if with_log:
            steps.append(("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")))
        steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    return ColumnTransformer(
        [
            ("log", numeric_block(True), log_cols),
            ("num", numeric_block(False), plain_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ("onehot", OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=25, sparse_output=False
                )),
            ]), categorical),
        ],
        remainder="drop",
    )


def source_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Map every transformed column back to the original column it came from.

    One-hot dummies such as ``home_ownership_RENT`` are credited to ``home_ownership``.
    This is what lets SHAP and permutation importance be compared feature by feature.
    """
    blocks = {name: list(cols) for name, _, cols in preprocessor.transformers_ if name != "remainder"}
    numeric = blocks.get("log", []) + blocks.get("num", [])
    categorical = blocks.get("cat", [])

    mapped = []
    for name in preprocessor.get_feature_names_out():
        prefix, body = name.split("__", 1)
        if prefix in ("log", "num"):
            mapped.append(body)
        else:
            matches = [c for c in categorical if body == c or body.startswith(c + "_")]
            if not matches:
                raise ValueError(f"Cannot map transformed column {name!r} to a source feature")
            mapped.append(max(matches, key=len))
    unknown = set(mapped) - set(numeric) - set(categorical)
    if unknown:
        raise ValueError(f"Unmapped source features: {sorted(unknown)}")
    return mapped
