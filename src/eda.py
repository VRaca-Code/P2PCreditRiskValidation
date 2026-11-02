"""Exploratory data analysis of the cleaned LendingClub loans (before any modelling).

What this file does
-------------------
Describes WHO borrows and WHERE defaults concentrate, so the model comparison can be read
against real portfolio facts. Everything here is descriptive; nothing is fitted and nothing is
used as a model input. Tables (each returned as a DataFrame and optionally saved as CSV):

* ``label_definition_comparison`` - how many loans and what default rate under (a) this repo's
                                    definition (resolved loans only) and (b) the common Kaggle
                                    notebook definition (Current = good, Late / Grace = bad).
                                    Shows how much the label choice alone moves the default rate.
* ``volume_by_year``              - loans, amount and default rate per issue year.
* ``default_rate_by``             - loans, defaults, default rate and 95% Wilson interval per
                                    group of any column; used for grade, purpose, term, region,
                                    income band, employment length, home ownership, FICO band,
                                    verification status.
* ``interest_by_outcome``         - mean interest rate for paid vs defaulted loans, by grade.
* ``numeric_default_correlation`` - Spearman correlation of each numeric feature with default.
* ``hypothesis_tests``            - Bondora-notebook style tests (Welch t, Kruskal-Wallis, chi-square)
                                    but reported with EFFECT SIZES (Cohen's d, epsilon^2, Cramer's V),
                                    because with hundreds of thousands of loans every p-value is tiny.
* ``risk_rule_validation``        - the Bondora notebook's hand-made "risky borrower" rule, adapted
                                    to LendingClub and CHECKED against actual defaults (lift,
                                    precision, recall, AUC) - the validation that notebook lists as future work.
* ``pricing_vs_realised_risk``    - per lender grade: interest charged vs defaults actually seen.
* ``missingness_informativeness`` - does a feature being MISSING itself signal default?
                                    (justifies keeping/dropping sparse features instead of filling with 0).
* ``univariate_vs_explanation``   - compares a plain univariate ranking of features with the
                                    model SHAP ranking: where they disagree, models use interactions.
* ``outlier_summary``             - per numeric feature: share beyond 1.5 x IQR, share above the 99.9th
                                    percentile, max / median; documents WHY outliers are kept (log
                                    transform, standardisation and tree models are robust to them).
* ``eda_report``                  - runs everything, prints and saves the tables.

What is original here: the label-definition comparison, effect sizes next to p-values, rule
validation against outcomes, missingness-as-signal, and linking EDA to the explanation study.

EDA keeps the lender's ``grade`` and ``int_rate`` (they are descriptive here); the MODEL still
drops them by default - see docs/data_cleaning.md. Run:

    python -m src.eda --synthetic --out outputs/tables/eda
    python -m src.eda --csv data/raw/accepted_2007_to_2018Q4.csv --out outputs/tables/eda

Figures for these tables: ``visualization.save_eda_figures``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATE_COL, TARGET, USECOLS, ExperimentConfig
from .preprocessing import STATUS_MAP, clean_target, stratified_sample

# Approximate US Census regions, for readable geographic groups instead of 50 states.
REGIONS = {
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS", "TN",
              "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
_STATE_TO_REGION = {s: r for r, states in REGIONS.items() for s in states}

# Statuses the popular Kaggle notebooks call "bad": unresolved late loans are counted as defaults.
NOTEBOOK_BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off",
                "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"}


def eda_frame(raw: pd.DataFrame, sample_size: int | None = None) -> pd.DataFrame:
    """Resolved loans with readable descriptive columns (grade kept, no imputation, all years kept).

    Adds: ``year``, ``term_months``, ``emp_years``, ``fico_score``, ``fico_band``, ``region``,
    ``income_band``. Row filters are only the resolved-status filter and the optional sample.
    """
    df = stratified_sample(clean_target(raw), sample_size)
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], format="%b-%Y", errors="coerce")
    out["year"] = out[DATE_COL].dt.year
    out["term_months"] = out["term"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    out["emp_years"] = (
        out["emp_length"].astype(str).str.replace("< 1 year", "0", regex=False)
        .str.replace("10+ years", "10", regex=False).str.extract(r"(\d+)", expand=False).astype(float)
    )
    if {"fico_range_low", "fico_range_high"} <= set(out.columns):
        out["fico_score"] = (out["fico_range_low"] + out["fico_range_high"]) / 2
        out["fico_band"] = pd.cut(out["fico_score"], [0, 660, 700, 740, 780, 900],
                                  labels=["<660", "660-699", "700-739", "740-779", "780+"], right=False)
    if "addr_state" in out:
        out["region"] = out["addr_state"].map(_STATE_TO_REGION).fillna("Other")
    if "annual_inc" in out:
        out["income_band"] = pd.cut(out["annual_inc"], [-np.inf, 40_000, 70_000, 100_000, 150_000, np.inf],
                                    labels=["<40k", "40-70k", "70-100k", "100-150k", "150k+"])
    if "home_ownership" in out:
        out["home_ownership"] = out["home_ownership"].replace(["ANY", "NONE"], "OTHER")
    return out


def _wilson(k: pd.Series, n: pd.Series, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    """95% Wilson score interval for a proportion k / n (well behaved for small groups)."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def default_rate_by(frame: pd.DataFrame, col: str, min_count: int = 50, sort: str = "index") -> pd.DataFrame:
    """Loans, defaults and default rate per value of ``col``; groups under ``min_count`` loans are dropped."""
    g = frame.groupby(col, observed=True)[TARGET].agg(Loans="size", Defaults="sum")
    g = g[g["Loans"] >= min_count].copy()
    g["Default_rate"] = g["Defaults"] / g["Loans"]
    g["ci95_low"], g["ci95_high"] = _wilson(g["Defaults"], g["Loans"])
    g["Share_of_loans"] = g["Loans"] / g["Loans"].sum()
    g = g.sort_values("Default_rate", ascending=False) if sort == "rate" else g.sort_index()
    return g.reset_index().rename(columns={col: "Group"}).assign(Variable=col)[
        ["Variable", "Group", "Loans", "Defaults", "Default_rate", "ci95_low", "ci95_high", "Share_of_loans"]]


def volume_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Loans, total amount and default rate per issue year (a falling late-year rate = right-censoring)."""
    g = frame.groupby("year").agg(Loans=(TARGET, "size"), Defaults=(TARGET, "sum"),
                                  Amount=("loan_amnt", "sum"))
    g["Default_rate"] = g["Defaults"] / g["Loans"]
    return g.reset_index().rename(columns={"year": "Issue_year"})


def interest_by_outcome(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean interest rate of paid vs defaulted loans, overall and within each lender grade."""
    if "int_rate" not in frame:
        return pd.DataFrame()
    rate = pd.to_numeric(frame["int_rate"].astype(str).str.rstrip("%"), errors="coerce")
    d = frame.assign(_rate=rate, Outcome=frame[TARGET].map({0: "Fully paid", 1: "Defaulted"}))
    overall = d.groupby("Outcome")["_rate"].mean().rename("Mean_interest_rate").reset_index().assign(Grade="All")
    if "grade" not in d:
        return overall[["Grade", "Outcome", "Mean_interest_rate"]]
    by_grade = (d.groupby(["grade", "Outcome"])["_rate"].mean().rename("Mean_interest_rate")
                .reset_index().rename(columns={"grade": "Grade"}))
    return pd.concat([overall, by_grade], ignore_index=True)[["Grade", "Outcome", "Mean_interest_rate"]]


def numeric_default_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of every numeric column with the default flag, strongest first."""
    skip = {TARGET, "year"}
    cols = [c for c in frame.select_dtypes(include="number").columns if c not in skip]
    corr = frame[cols].corrwith(frame[TARGET], method="spearman")
    out = pd.DataFrame({"Feature": corr.index, "Spearman_with_default": corr.values,
                        "Observed": frame[cols].notna().mean().values})
    return out.reindex(out["Spearman_with_default"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def label_definition_comparison(raw: pd.DataFrame) -> pd.DataFrame:
    """Default rate and sample size under this repo's label versus the common notebook label.

    Notebook label: bad = charged off / default / grace period / late; EVERYTHING else (including
    loans still Current) = good. Current loans have not finished, so calling them good lowers the
    default rate and mixes unresolved loans into the target.
    """
    status = raw[TARGET]
    resolved = status.map(STATUS_MAP)
    nb_bad = status.isin(NOTEBOOK_BAD)
    rows = [
        ("This repo: resolved loans only", int(resolved.notna().sum()), float(resolved.dropna().mean()),
         "Fully Paid = 0; Charged Off / Default = 1; everything unresolved is dropped"),
        ("Notebook style: all loans, unresolved counted", int(status.notna().sum()), float(nb_bad[status.notna()].mean()),
         "Current = good; Late / Grace Period / Charged Off / Default = bad"),
        ("Notebook style, Current excluded", int((status.notna() & (status != "Current")).sum()),
         float(nb_bad[status.notna() & (status != "Current")].mean()), "late loans still labelled bad before they resolve"),
    ]
    table = pd.DataFrame(rows, columns=["Definition", "Loans", "Default_rate", "Rule"])
    table["Unresolved_loans_in_sample"] = [0, int((status.isin(["Current", "In Grace Period", "Late (16-30 days)",
                                                                "Late (31-120 days)"])).sum()),
                                           int(status.isin(["In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]).sum())]
    return table


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def hypothesis_tests(frame: pd.DataFrame, numeric=("int_rate", "loan_amnt", "dti", "annual_inc", "revol_util", "fico_score"),
                     categorical=("term_months", "purpose", "grade", "home_ownership", "verification_status", "region")) -> pd.DataFrame:
    """Defaulted vs paid loans: Welch t-test per numeric feature, chi-square per categorical feature.

    Reports an effect size beside every p-value (Cohen's d; Cramer's V) and a plain-language size
    label, so "significant" is never mistaken for "important". Also tests whether the lender grade
    separates interest rates (Kruskal-Wallis with epsilon-squared).
    """
    from scipy import stats

    rows = []
    d = frame.copy()
    if "int_rate" in d:
        d["int_rate"] = pd.to_numeric(d["int_rate"].astype(str).str.rstrip("%"), errors="coerce")
    for c in numeric:
        if c not in d:
            continue
        paid, bad = d.loc[d[TARGET] == 0, c].dropna(), d.loc[d[TARGET] == 1, c].dropna()
        if len(paid) < 2 or len(bad) < 2:
            continue
        t = stats.ttest_ind(bad, paid, equal_var=False)
        eff = _cohens_d(bad, paid)
        label = "negligible" if abs(eff) < 0.2 else "small" if abs(eff) < 0.5 else "medium" if abs(eff) < 0.8 else "large"
        rows.append({"Test": "Welch t (defaulted - paid)", "Variable": c, "Statistic": float(t.statistic),
                     "p_value": float(t.pvalue), "Effect_size": eff, "Effect_measure": "Cohen's d",
                     "Effect_label": label, "Mean_defaulted": float(bad.mean()), "Mean_paid": float(paid.mean())})
    for c in categorical:
        if c not in d:
            continue
        table = pd.crosstab(d[c], d[TARGET])
        table = table[table.sum(axis=1) >= 20]
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p, _, _ = stats.chi2_contingency(table)
        n = table.to_numpy().sum()
        v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
        label = "negligible" if v < 0.1 else "small" if v < 0.3 else "medium" if v < 0.5 else "large"
        rows.append({"Test": "Chi-square (default vs group)", "Variable": c, "Statistic": float(chi2),
                     "p_value": float(p), "Effect_size": v, "Effect_measure": "Cramer's V", "Effect_label": label,
                     "Mean_defaulted": np.nan, "Mean_paid": np.nan})
    if {"grade", "int_rate"} <= set(d.columns):
        groups = [g["int_rate"].dropna().to_numpy() for _, g in d.groupby("grade") if g["int_rate"].notna().sum() > 1]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            n = sum(len(g) for g in groups)
            eps = float(h * (n + 1) / (n**2 - 1))  # epsilon-squared: share of rank variance explained
            rows.append({"Test": "Kruskal-Wallis (int_rate across grades)", "Variable": "grade", "Statistic": float(h),
                         "p_value": float(p), "Effect_size": eps, "Effect_measure": "epsilon^2",
                         "Effect_label": "large" if eps > 0.14 else "medium" if eps > 0.06 else "small",
                         "Mean_defaulted": np.nan, "Mean_paid": np.nan})
    return pd.DataFrame(rows)


def risk_rule_validation(frame: pd.DataFrame, burden_threshold: float = 0.35, max_emp_years: float = 1.0) -> pd.DataFrame:
    """Check a hand-made 'risky borrower' rule against actual defaults.

    Rule (adapted from the Bondora notebook): loan amount above ``burden_threshold`` of annual
    income AND employment of at most ``max_emp_years``. A second, LendingClub-native rule uses
    the instalment burden (12 x instalment / annual income). For each rule the table gives the
    share flagged, default rate inside and outside, lift, precision, recall and AUC of the flag.
    """
    from sklearn.metrics import roc_auc_score

    d = frame.dropna(subset=["annual_inc", "emp_years"]).copy()
    d = d[d["annual_inc"] > 0]
    y = d[TARGET].to_numpy()
    rules = {
        f"Loan > {burden_threshold:.0%} of income AND employed <= {max_emp_years:g}y": (d["loan_amnt"] / d["annual_inc"] > burden_threshold) & (d["emp_years"] <= max_emp_years),
        f"Loan > {burden_threshold:.0%} of income": d["loan_amnt"] / d["annual_inc"] > burden_threshold,
        "Employed <= 1y": d["emp_years"] <= max_emp_years,
    }
    if "installment" in d:
        rules["Instalment burden > 25% of income"] = 12 * d["installment"] / d["annual_inc"] > 0.25
    rows, base = [], float(y.mean())
    for name, flag in rules.items():
        flag = flag.to_numpy()
        n_flag = int(flag.sum())
        inside = float(y[flag].mean()) if n_flag else np.nan
        outside = float(y[~flag].mean()) if (~flag).any() else np.nan
        tp = int((flag & (y == 1)).sum())
        rows.append({"Rule": name, "Flagged_share": n_flag / len(d), "Default_rate_flagged": inside,
                     "Default_rate_not_flagged": outside, "Lift": inside / base if n_flag else np.nan,
                     "Precision": tp / n_flag if n_flag else np.nan, "Recall": tp / max(int(y.sum()), 1),
                     "AUC_of_flag": float(roc_auc_score(y, flag)) if 0 < n_flag < len(d) else np.nan,
                     "Base_default_rate": base})
    return pd.DataFrame(rows)


def pricing_vs_realised_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Per lender grade: mean interest rate charged vs default rate actually realised.

    NAIVE comparison: it ignores recoveries, loss given default and payment timing, so it is not
    a profit measure. It shows whether higher grades are priced in proportion to realised risk.
    """
    if not {"grade", "int_rate"} <= set(frame.columns):
        return pd.DataFrame()
    d = frame.assign(_rate=pd.to_numeric(frame["int_rate"].astype(str).str.rstrip("%"), errors="coerce"))
    g = d.groupby("grade").agg(Loans=(TARGET, "size"), Mean_interest_rate=("_rate", "mean"), Default_rate=(TARGET, "mean"))
    g["Default_rate_pct"] = g["Default_rate"] * 100
    g["Rate_minus_default_pct_points"] = g["Mean_interest_rate"] - g["Default_rate_pct"]
    g["Rate_per_default_point"] = g["Mean_interest_rate"] / g["Default_rate_pct"].replace(0, np.nan)
    return g.reset_index()


def missingness_informativeness(frame: pd.DataFrame, min_missing: int = 30) -> pd.DataFrame:
    """For each partly missing feature: default rate when missing vs when observed.

    A big gap means the missingness itself carries signal (filling with 0 or the median would
    erase it). A small gap means the feature can be imputed without losing information.
    """
    rows = []
    for c in frame.columns:
        if c in {TARGET, DATE_COL, "year", "emp_years", "term_months", "fico_band", "region", "income_band"}:
            continue  # derived columns would just repeat their source column
        miss = frame[c].isna()
        n_miss = int(miss.sum())
        if n_miss < min_missing or n_miss > len(frame) - min_missing:
            continue
        y = frame[TARGET]
        lo, hi = _wilson(pd.Series([y[miss].sum()]), pd.Series([n_miss]))
        rows.append({"Feature": c, "Missing_share": n_miss / len(frame), "Default_rate_missing": float(y[miss].mean()),
                     "Default_rate_observed": float(y[~miss].mean()),
                     "Gap_pct_points": 100 * float(y[miss].mean() - y[~miss].mean()),
                     "Missing_ci95_low": float(lo.iloc[0]), "Missing_ci95_high": float(hi.iloc[0])})
    out = pd.DataFrame(rows)
    return out.reindex(out["Gap_pct_points"].abs().sort_values(ascending=False).index).reset_index(drop=True) if len(out) else out


# EDA column names -> the model's source-feature names, so the two rankings can be compared.
_ALIASES = {"term_months": "term", "emp_years": "emp_length", "fico_score": "fico_score"}


def univariate_vs_explanation(correlation: pd.DataFrame, importances: pd.DataFrame) -> pd.DataFrame:
    """Rank each feature by |univariate Spearman| and by mean model importance; return both ranks per model.

    ``correlation`` is ``numeric_default_correlation`` output; ``importances`` is the seed grid's
    importance table (Model, Seed, Feature, Importance). Only numeric features present in both are
    compared. A large rank gap = the model uses the feature differently than a one-at-a-time view.
    """
    uni = correlation.assign(Feature=correlation["Feature"].replace(_ALIASES))
    uni["Univariate_rank"] = uni["Spearman_with_default"].abs().rank(ascending=False)
    rows = []
    for model, sub in importances.groupby("Model"):
        mean = sub.groupby("Feature")["Importance"].mean()
        joined = uni.merge(mean.rename("Model_importance").reset_index(), on="Feature")
        if len(joined) < 3:
            continue
        joined["Model_rank"] = joined["Model_importance"].rank(ascending=False)
        joined["Model"] = model
        joined["Rank_gap"] = joined["Univariate_rank"] - joined["Model_rank"]
        rows.append(joined[["Model", "Feature", "Spearman_with_default", "Univariate_rank",
                            "Model_importance", "Model_rank", "Rank_gap"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def outlier_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Outlier share per numeric feature (Tukey fences and extreme tail); nothing is removed."""
    rows = []
    for c in frame.select_dtypes(include="number").columns:
        if c in {TARGET, "year", "fico_range_low", "fico_range_high"}:  # FICO bounds repeat fico_score
            continue
        x = frame[c].dropna()
        if x.nunique() < 5:
            continue
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        fence = ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() if iqr > 0 else 0.0
        med = x.median()
        rows.append({"Feature": c, "Outside_1.5_IQR_share": float(fence),
                     "Above_p99.9_share": float((x > x.quantile(0.999)).mean()),
                     "Max": float(x.max()), "Median": float(med),
                     "Max_over_median": float(x.max() / med) if med else np.nan,
                     "Skewness": float(x.skew())})
    out = pd.DataFrame(rows)
    return out.sort_values("Outside_1.5_IQR_share", ascending=False).reset_index(drop=True) if len(out) else out


GROUP_VARIABLES = ["grade", "purpose", "term_months", "region", "income_band", "emp_years",
                   "home_ownership", "fico_band", "verification_status"]


def eda_report(raw: pd.DataFrame, cfg: ExperimentConfig | None = None, out_dir: Path | None = None,
               verbose: bool = True) -> dict:
    """Run every EDA table on raw LendingClub rows; print and optionally save CSVs."""
    cfg = cfg or ExperimentConfig()
    frame = eda_frame(raw, cfg.sample_size)
    groups = [default_rate_by(frame, c) for c in GROUP_VARIABLES if c in frame]
    report = {
        "label_definitions": label_definition_comparison(raw),
        "volume_by_year": volume_by_year(frame),
        "default_rate_groups": pd.concat(groups, ignore_index=True) if groups else pd.DataFrame(),
        "interest_by_outcome": interest_by_outcome(frame),
        "numeric_correlation": numeric_default_correlation(frame),
        "hypothesis_tests": hypothesis_tests(frame),
        "risk_rule_validation": risk_rule_validation(frame),
        "pricing_vs_realised_risk": pricing_vs_realised_risk(frame),
        "missingness_informativeness": missingness_informativeness(frame),
        "outlier_summary": outlier_summary(frame),
        "frame": frame,  # kept in memory for plotting; not written to disk
    }
    if verbose:
        pd.set_option("display.width", 170)
        for name, table in report.items():
            if name == "frame":
                continue
            print(f"\n=== {name} ===")
            print(table.round(4).to_string(index=False, max_rows=40))
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in report.items():
            if name != "frame":
                table.to_csv(out_dir / f"eda_{name}.csv", index=False)
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated data")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=None, help="stratified sample size (default: config value)")
    ap.add_argument("--out", type=Path, default=None, help="folder for CSV tables")
    ap.add_argument("--figure-dir", type=Path, default=None, help="folder for figures (omit to skip)")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    cfg = ExperimentConfig(csv_path=args.csv) if args.csv else ExperimentConfig()
    if args.sample:
        cfg.sample_size = args.sample
    if args.synthetic:
        from .synthetic import make_synthetic_lendingclub
        raw = make_synthetic_lendingclub(8_000, seed=0)
    else:
        raw = pd.read_csv(cfg.csv_path, usecols=USECOLS, low_memory=False)
    report = eda_report(raw, cfg, args.out)
    if args.figure_dir:
        from .visualization import save_eda_figures
        paths = save_eda_figures(report, args.figure_dir)
        print(f"\nWrote {len(paths)} EDA figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
