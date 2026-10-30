"""Shared pytest fixtures.

What this file does: builds one small synthetic LendingClub dataset per test session and
exposes it at three stages - ``raw_df`` (raw columns), ``frame`` (cleaned and engineered) and
``splits`` (chronological X_train, y_train, X_test, y_test) - so every test file reuses the same
data without needing the 1.6 GB Kaggle CSV.
"""

import pytest

from src.preprocessing import prepare_frame, split_xy, temporal_split
from src.synthetic import make_synthetic_lendingclub


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic_lendingclub(n=6_000, seed=1)


@pytest.fixture(scope="session")
def frame(raw_df):
    return prepare_frame(raw_df)


@pytest.fixture(scope="session")
def splits(frame):
    train, test = temporal_split(frame, test_fraction=0.2)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    return X_train, y_train, X_test, y_test
