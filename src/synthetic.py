"""Generator of a small, fake LendingClub-shaped dataset for tests, CI and demos.

What this file does
-------------------
``make_synthetic_lendingclub(n, seed)`` returns a DataFrame with the same raw columns and
string formats as the real Kaggle file (``" 36 months"``, ``"< 1 year"``, ``"Mar-2014"``,
FICO low/high bands), including things the cleaning code must handle: unresolved loans
("Current", "Late"), missing values, an ``ANY`` home-ownership level, and a default rate that
drifts slightly over time. Defaults are generated from a latent borrower-quality variable so
models have a real signal to learn.

It carries NO information about real credit risk: use it only to prove the pipeline runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_STATES = ["CA", "TX", "NY", "FL", "IL", "NJ", "PA", "OH", "GA", "VA"]
_PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase"]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def make_synthetic_lendingclub(n: int = 5_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    year = rng.integers(2012, 2017, n)
    month = rng.integers(0, 12, n)
    issue_d = [f"{_MONTHS[m]}-{y}" for m, y in zip(month, year)]

    loan_amnt = rng.choice(np.arange(1_000, 35_001, 500), n).astype(float)
    term_months = rng.choice([36, 60], n, p=[0.72, 0.28])
    quality = rng.normal(0, 1, n)  # latent borrower quality
    int_rate = np.clip(13 - 2.2 * quality + 1.5 * (term_months == 60) + rng.normal(0, 1, n), 5.3, 30)
    annual_inc = np.exp(rng.normal(11.0 + 0.15 * quality, 0.5, n))
    dti = np.clip(rng.normal(18 - 2.0 * quality, 7, n), 0, 45)
    revol_util = np.clip(rng.normal(50 - 6 * quality, 22, n), 0, 120)
    installment = loan_amnt * (int_rate / 1200) / (1 - (1 + int_rate / 1200) ** (-term_months))

    letter = np.digitize(int_rate, [8, 11, 14, 17, 20, 24])
    sub_grade = [f"{'ABCDEFG'[g]}{rng.integers(1, 6)}" for g in letter]

    logit = (
        -2.0
        + 0.10 * (int_rate - 13)
        + 0.020 * (dti - 18)
        + 0.010 * (revol_util - 50)
        - 0.35 * (np.log(annual_inc) - 11.0)
        + 0.35 * (term_months == 60)
        + 0.06 * (year - 2014)
        - 0.25 * quality
    )
    p_default = 1 / (1 + np.exp(-logit))
    defaulted = rng.random(n) < p_default

    status = np.where(defaulted, "Charged Off", "Fully Paid").astype(object)
    status[rng.random(n) < 0.06] = "Current"  # unresolved: must be dropped by clean_target
    status[rng.random(n) < 0.01] = "Late (31-120 days)"

    emp_length = rng.choice(
        ["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years", None], n,
        p=[0.08, 0.07, 0.10, 0.10, 0.10, 0.50, 0.05],
    )
    earliest = [f"{_MONTHS[rng.integers(0, 12)]}-{rng.integers(1975, 2011)}" for _ in range(n)]

    df = pd.DataFrame({
        "loan_amnt": loan_amnt,
        "term": [f" {t} months" for t in term_months],
        "int_rate": int_rate,
        "installment": installment,
        "grade": ["ABCDEFG"[g] for g in letter],
        "sub_grade": sub_grade,
        "emp_title": rng.choice(["Teacher", "Manager", "Driver", "Nurse", None], n),
        "emp_length": emp_length,
        "home_ownership": rng.choice(["RENT", "MORTGAGE", "OWN", "ANY"], n, p=[0.45, 0.42, 0.12, 0.01]),
        "annual_inc": annual_inc,
        "verification_status": rng.choice(["Verified", "Source Verified", "Not Verified"], n),
        "issue_d": issue_d,
        "loan_status": status,
        "purpose": rng.choice(_PURPOSES, n, p=[0.55, 0.2, 0.1, 0.1, 0.05]),
        "title": rng.choice(["Debt consolidation", "Credit card refinancing", None], n),
        "addr_state": rng.choice(_STATES, n),
        "dti": dti,
        "earliest_cr_line": earliest,
        "open_acc": rng.integers(2, 30, n).astype(float),
        "pub_rec": rng.choice([0, 0, 0, 0, 1, 2], n).astype(float),
        "revol_bal": np.exp(rng.normal(9.3, 1.0, n)),
        "revol_util": revol_util,
        "total_acc": rng.integers(4, 60, n).astype(float),
        "initial_list_status": rng.choice(["w", "f"], n),
        "application_type": rng.choice(["Individual", "Joint App"], n, p=[0.97, 0.03]),
        "mort_acc": rng.integers(0, 6, n).astype(float),
        "pub_rec_bankruptcies": rng.choice([0, 0, 0, 1], n).astype(float),
    })
    fico_low = np.clip(np.round((700 + 35 * quality + rng.normal(0, 15, n)) / 5) * 5, 660, 845)
    df["fico_range_low"] = fico_low
    df["fico_range_high"] = fico_low + 4
    df.loc[rng.random(n) < 0.08, "mort_acc"] = np.nan
    df.loc[rng.random(n) < 0.01, "revol_util"] = np.nan
    return df
