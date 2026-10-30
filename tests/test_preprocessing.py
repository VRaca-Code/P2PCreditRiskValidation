"""Tests for src/preprocessing.py.

What this file checks: target encoding drops unresolved loans, text fields (term, employment
length, dates) are parsed, FICO bands are merged, the lender-score ablation drops the right
columns, sampling preserves the default rate, the temporal split is chronological and disjoint,
the preprocessor learns statistics from training rows only (no leakage), log transforms reduce skew,
sparse features are flagged, and every transformed column maps back to an original feature.
"""

import numpy as np
import pandas as pd

from src.config import DATE_COL, TARGET
from src.preprocessing import (
    build_preprocessor,
    clean_target,
    engineer_features,
    find_sparse_features,
    source_feature_names,
    split_xy,
    stratified_sample,
    temporal_split,
)


def test_clean_target_drops_unresolved_and_maps_policy_labels():
    df = pd.DataFrame({TARGET: [
        "Fully Paid", "Charged Off", "Current", "Late (31-120 days)", "Default",
        "Does not meet the credit policy. Status:Fully Paid",
        "Does not meet the credit policy. Status:Charged Off",
    ]})
    out = clean_target(df)
    assert out[TARGET].tolist() == [0, 1, 1, 0, 1]
    assert out[TARGET].dtype.kind == "i"


def test_engineer_features_parses_strings(raw_df):
    out = engineer_features(clean_target(raw_df))
    assert set(out["term"].dropna().unique()) <= {36.0, 60.0}
    assert out["emp_length"].min() == 0.0  # "< 1 year" -> 0
    assert out["emp_length"].max() == 10.0  # "10+ years" -> 10
    assert pd.api.types.is_datetime64_any_dtype(out[DATE_COL])
    assert {"earliest_cr_year", "earliest_cr_month"} <= set(out.columns)
    assert not {"earliest_cr_line", "emp_title", "title", "grade"} & set(out.columns)
    assert "ANY" not in set(out["home_ownership"])


def test_lender_assigned_ablation_drops_columns(raw_df):
    out = engineer_features(clean_target(raw_df), drop_lender_assigned=True)
    assert "int_rate" not in out and "sub_grade" not in out


def test_stratified_sample_preserves_default_rate(frame):
    sample = stratified_sample(frame, 1_000, seed=0)
    assert abs(len(sample) - 1_000) <= 2
    assert abs(sample[TARGET].mean() - frame[TARGET].mean()) < 0.01


def test_temporal_split_is_chronological_and_disjoint(frame):
    train, test = temporal_split(frame, test_fraction=0.2)
    assert train[DATE_COL].max() < test[DATE_COL].min()
    assert len(train) + len(test) == len(frame)
    assert 0.1 < len(test) / len(frame) < 0.35


def test_preprocessor_statistics_come_from_training_data_only(splits):
    X_train, _, X_test, _ = splits
    prep = build_preprocessor(X_train).fit(X_train)
    imputer = prep.named_transformers_["num"].named_steps["imputer"]
    num_cols = dict((n, c) for n, _, c in prep.transformers_)["num"]
    expected = X_train[num_cols].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, expected)
    # sanity: it would differ from a median that peeked at the test rows
    peeked = pd.concat([X_train, X_test])[num_cols].median().to_numpy()
    assert not np.allclose(imputer.statistics_, peeked)


def test_fico_band_is_merged_into_one_score(raw_df):
    out = engineer_features(clean_target(raw_df))
    assert "fico_score" in out
    assert not {"fico_range_low", "fico_range_high"} & set(out.columns)
    assert out["fico_score"].between(660, 850).all()


def test_log_block_reduces_skew_of_money_columns(splits):
    from scipy.stats import skew
    X_train, _, _, _ = splits
    prep = build_preprocessor(X_train).fit(X_train)
    names = list(prep.get_feature_names_out())
    Xt = prep.transform(X_train)
    assert "log__annual_inc" in names and "log__revol_bal" in names
    raw_skew = skew(X_train["annual_inc"].dropna())
    log_skew = skew(Xt[:, names.index("log__annual_inc")])
    assert abs(log_skew) < abs(raw_skew) / 2
    assert "annual_inc" in X_train  # raw column kept so stress tests can still shock it


def test_find_sparse_features_flags_low_coverage_columns():
    X = pd.DataFrame({"full": np.arange(100.0), "mostly": np.r_[np.arange(80.0), [np.nan] * 20],
                      "sparse": np.r_[np.arange(30.0), [np.nan] * 70]})
    assert find_sparse_features(X, 0.70) == ["sparse"]
    assert find_sparse_features(X, 0.99) == ["mostly", "sparse"]


def test_source_feature_names_cover_every_transformed_column(splits):
    X_train, _, _, _ = splits
    prep = build_preprocessor(X_train).fit(X_train)
    names = source_feature_names(prep)
    assert len(names) == len(prep.get_feature_names_out())
    assert set(names) <= set(X_train.columns)
    assert "home_ownership" in names and "loan_amnt" in names


def test_split_xy_excludes_target_and_date(frame):
    X, y = split_xy(frame)
    assert TARGET not in X and DATE_COL not in X
    assert len(X) == len(y)
