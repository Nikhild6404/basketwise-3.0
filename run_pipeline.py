"""Run the full BasketWise analytics pipeline end to end.

    python run_pipeline.py                 # use the dataset shipped in data/raw
    python run_pipeline.py --regenerate    # rebuild the synthetic data first

Steps: (0) generate data -> (1) clean & transform -> (2) descriptive ->
(3) classification -> (4) clustering -> (5) association rules ->
(6) regression -> (7) business case. Figures go to outputs/figures,
tables to outputs/tables, headline numbers to outputs/results_summary.json.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

import association      # noqa: E402
import business_case    # noqa: E402
import classification   # noqa: E402
import clean_transform  # noqa: E402
import clustering       # noqa: E402
import config as C      # noqa: E402
import descriptive      # noqa: E402
import generate_data    # noqa: E402
import regression       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true", help="rebuild the synthetic raw dataset first")
    args = ap.parse_args()

    t0 = time.time()
    if args.regenerate or not C.RAW_FILE.exists():
        generate_data.main()
    if C.RESULTS_FILE.exists():
        C.RESULTS_FILE.unlink()          # fresh summary each run
    for step in (clean_transform, descriptive, classification, clustering,
                 association, regression, business_case):
        step.main()
    print(f"\nDone in {time.time() - t0:.0f}s. See outputs/figures, outputs/tables "
          f"and outputs/results_summary.json")


if __name__ == "__main__":
    main()
