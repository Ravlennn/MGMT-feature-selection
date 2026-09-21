import argparse
from pathlib import Path

from mgmt_features.pipelines import run_upenn_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Apply all three feature-selection approaches to clean UPenn-GBM radiomics.")
    parser.add_argument("--quick", action="store_true", help="Small smoke configuration for checking the pipeline.")
    args = parser.parse_args()

    run_upenn_pipeline(
        ROOT / "data/processed/upenn/upenn_clean.csv",
        ROOT / "results/upenn",
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
