# Data

The real analysis uses the public **LendingClub accepted-loans** file
`accepted_2007_to_2018Q4.csv` (about 1.6 GB). It is **not** committed to this
repository.

1. Download it from Kaggle ("Lending Club Loan Data", file `accepted_2007_to_2018Q4.csv`)
   and accept the dataset's terms.
2. Place it at `data/raw/accepted_2007_to_2018Q4.csv`, or point to it with the
   `P2P_CSV_PATH` environment variable or the `--csv` flag.

No download is needed to try the code: `python -m src.experiments --synthetic`
generates a small LendingClub-shaped dataset (see `src/synthetic.py`). Synthetic
results only prove the pipeline runs; they say nothing about real credit risk.

Only columns known at origination are used (see `USECOLS` in `src/config.py`).
Loans that are still open (`Current`, `Late`, ...) are dropped because their
final outcome is unknown.
