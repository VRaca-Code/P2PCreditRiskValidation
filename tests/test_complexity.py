"""Tests for src/complexity.py and the model factory in src/models.py.

What this file checks: logistic regression counts coefficients + intercept, random forests count
total tree nodes, the forest is structurally larger than the logistic model, the complexity table has
the expected columns, sklearn models are always available, and (if PyTorch is installed) the DNN
and KAN report a positive parameter count.
"""

import pytest

from src.complexity import complexity_table, structural_units
from src.models import available_models, make_pipeline


@pytest.fixture(scope="module")
def fitted(splits):
    X_train, y_train, _, _ = splits
    sub = X_train.sample(1_500, random_state=0)
    return {
        name: make_pipeline(name, sub, seed=0).fit(sub, y_train.loc[sub.index])
        for name in ("Logistic Regression", "Random Forest")
    }


def test_logistic_units_are_coefficients_plus_intercept(fitted):
    pipe = fitted["Logistic Regression"]
    units, measure = structural_units(pipe)
    n_features = len(pipe.named_steps["prep"].get_feature_names_out())
    assert units == n_features + 1
    assert "coefficients" in measure


def test_random_forest_units_are_total_nodes(fitted):
    pipe = fitted["Random Forest"]
    forest = pipe.named_steps["model"]
    assert structural_units(pipe)[0] == sum(t.tree_.node_count for t in forest.estimators_)


def test_forest_is_structurally_larger_than_logistic(fitted):
    assert structural_units(fitted["Random Forest"])[0] > structural_units(fitted["Logistic Regression"])[0]


def test_complexity_table_columns(fitted):
    table = complexity_table(fitted, {"Logistic Regression": 0.1}, {"Random Forest": 0.2})
    assert list(table.index) == ["Logistic Regression", "Random Forest"]
    assert {"Structural_Units", "Train_Time_Seconds", "Predict_Time_Seconds"} <= set(table.columns)


def test_available_models_always_includes_sklearn_models():
    assert {"Logistic Regression", "Random Forest"} <= set(available_models())


def test_torch_models_count_parameters(splits):
    pytest.importorskip("torch")
    X_train, y_train, _, _ = splits
    sub = X_train.sample(1_200, random_state=0)
    for name in ("DNN", "KAN"):
        pipe = make_pipeline(name, sub, seed=0)
        pipe.named_steps["model"].set_params(max_epochs=2)
        pipe.fit(sub, y_train.loc[sub.index])
        assert structural_units(pipe)[0] > 0
