"""P2P credit-risk model validation: performance, complexity and explanation stability.

What this file does
-------------------
Marks ``src`` as an importable Python package (``from src.metrics import ...``) and holds the
package version. The package compares Logistic Regression, Random Forest, XGBoost and
optionally a DNN and a KAN on LendingClub loans, and tests whether models with similar
performance also give similar explanations.

Module map
----------
config           settings and constants            data_quality    cleaning audit tables
preprocessing    cleaning, split, preprocessor      models          model factory
neural           sklearn wrapper for torch nets     torch_nets      MLP and KAN layers
metrics          AUC, KS, Brier, ECE, thresholds    calibration     reliability + recalibration
complexity       structural size of each model      interpretability SHAP / permutation importance
robustness       PSI, CV, stress, backtest          stats           bootstrap, TOST, Holm, Jaccard
scoring          scorecard + Pareto                 experiments     seed grid, H1-H3, CLI
visualization    figures                            synthetic       fake data for tests and demos
"""

__version__ = "0.4.0"
