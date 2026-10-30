"""Data-quality report: how many loans survive each cleaning step, and what is missing.

What this file does
-------------------
Makes the cleaning process auditable instead of implicit. It re-runs the same steps as
``preprocessing.prepare_frame`` while counting rows and default rates after each one:

* ``status_breakdown``    - every raw ``loan_status`` value and whether it is kept or dropped
* ``attrition_table``     - rows and default rate after: resolved filter, sampling, date parsing,
                            vintage cut-off, then the train / test split
* ``missingness_report``  - per-feature missing rate and number of distinct values
* ``default_rate_by_year``- loans and default rate per issue year (shows right-censoring of late years)
* ``data_quality_report`` - runs all of the above, prints them and optionally saves CSVs

Run it before any modelling:

    python -m src.data_quality --synthetic
    python -m src.data_quality --csv data/raw/accepted_2007_to_2018Q4.csv

The reasons behind each step are written up in docs/data_cleaning.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import DATE_COL, TARGET, USECOLS, ExperimentConfig
from .preprocessing import (
    STATUS_MAP,
    clean_target,
    engineer_features,
    filter_vintages,
    find_sparse_features,
    split_xy,
    stratified_sample,
    temporal_split,
)


def status_breakdown(raw: pd.DataFrame) -> pd.DataFrame:
    """Count of each raw loan_status and whether the cleaning step keeps it."""
    counts = raw[TARGET].value_counts(dropna=False).rename("Rows").reset_index()
    counts.columns = ["Raw_status", "Rows"]
    counts["Kept"] = counts["Raw_status"].isin(STATUS_MAP)
    counts["Encoded_as"] = counts["Raw_status"].map(STATUS_MAP)  # 0 paid, 1 default, NaN dropped
    counts["Share"] = counts["Rows"] / counts["Rows"].sum()
    return counts


def attrition_table(raw: pd.DataFrame, cfg: ExperimentConfig | None = None) -> pd.DataFrame:
    """Rows and default rate after each cleaning step, in the order the pipeline applies them."""
    cfg = cfg or ExperimentConfig()
    rows = [("Raw rows", len(raw), float("nan"), "as loaded from the CSV")]

    df = clean_target(raw)
    rows.append(("Resolved loans only", len(df), df[TARGET].mean(),
                 "Fully Paid / Charged Off / Default kept; Current, Late, In Grace Period dropped"))

    df = stratified_sample(df, cfg.sample_size)
    rows.append(("After stratified sample", len(df), df[TARGET].mean(),
                 f"sample_size={cfg.sample_size}; default rate preserved"))

    df = engineer_features(df, drop_lender_assigned=cfg.drop_lender_assigned)
    n_before = len(df)
    df = df.dropna(subset=[DATE_COL])
    rows.append(("Valid issue date", len(df), df[TARGET].mean(),
                 f"{n_before - len(df):,} rows with unparseable issue_d removed"))

    df = filter_vintages(df, cfg.max_issue_year)
    rows.append((f"Issued <= {cfg.max_issue_year}", len(df), df[TARGET].mean(),
                 "later vintages are right-censored (most have not had time to default)"))

    train, test = temporal_split(df, DATE_COL, cfg.test_fraction)
    rows.append(("Training set (earlier months)", len(train), train[TARGET].mean(), "fits models and preprocessors"))
    rows.append(("Test set (latest months)", len(test), test[TARGET].mean(), "never used for fitting"))

    table = pd.DataFrame(rows, columns=["Step", "Rows", "Default_rate", "Note"])
    table["Rows_removed"] = (table["Rows"].shift(1) - table["Rows"]).where(~table["Step"].str.contains("set"))
    return table[["Step", "Rows", "Rows_removed", "Default_rate", "Note"]]


def missingness_report(X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature dtype, missing rate and distinct-value count, most-missing first."""
    return pd.DataFrame({
        "dtype": X.dtypes.astype(str),
        "missing_count": X.isna().sum(),
        "missing_rate": X.isna().mean(),
        "n_unique": X.nunique(dropna=True),
    }).sort_values("missing_rate", ascending=False)


def default_rate_by_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Loans and default rate per issue year. A sharp fall in the latest years signals censoring."""
    year = frame[DATE_COL].dt.year.rename("Issue_year")
    return (frame.groupby(year)[TARGET].agg(Loans="size", Default_rate="mean").reset_index())


def data_quality_report(raw: pd.DataFrame, cfg: ExperimentConfig | None = None,
                        out_dir: Path | None = None, verbose: bool = True) -> dict:
    """Run every check above on raw LendingClub rows; print and optionally save the tables."""
    cfg = cfg or ExperimentConfig()
    cleaned = engineer_features(stratified_sample(clean_target(raw), cfg.sample_size),
                                drop_lender_assigned=cfg.drop_lender_assigned)
    cleaned = cleaned.dropna(subset=[DATE_COL])
    train, _ = temporal_split(filter_vintages(cleaned, cfg.max_issue_year), DATE_COL, cfg.test_fraction)
    X_train, _ = split_xy(train)

    report = {
        "status_breakdown": status_breakdown(raw),
        "attrition": attrition_table(raw, cfg),
        "default_rate_by_year": default_rate_by_year(cleaned),  # all years, before the vintage cut
        "missingness": missingness_report(X_train),
        "sparse_features": pd.DataFrame({"Feature": find_sparse_features(X_train, cfg.min_feature_coverage)}),
    }
    if verbose:
        pd.set_option("display.width", 170)
        pd.set_option("display.max_colwidth", 80)
        for name, table in report.items():
            print(f"\n=== {name} ===")
            print(table.round(4).to_string(index=False) if name != "missingness" else table.round(4).to_string())
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in report.items():
            table.to_csv(out_dir / f"data_quality_{name}.csv", index=name == "missingness")
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="use generated data")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="folder for CSV copies of each table")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig(csv_path=args.csv) if args.csv else ExperimentConfig()
    if args.synthetic:
        from .synthetic import make_synthetic_lendingclub
        raw = make_synthetic_lendingclub(8_000, seed=0)
    else:
        raw = pd.read_csv(cfg.csv_path, usecols=USECOLS, low_memory=False)
    data_quality_report(raw, cfg, args.out)


if __name__ == "__main__":
    main()
