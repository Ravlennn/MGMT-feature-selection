import argparse
from pathlib import Path

from mgmt_features.pipelines import run_reference_pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Run the three methods on the 53-patient reference table.")
    parser.add_argument("--quick", action="store_true", help="Small smoke configuration for checking the pipeline.")
    args = parser.parse_args()

    run_reference_pipeline(
        ROOT / "data/raw/do_2022/Training dataset.csv",
        ROOT / "results/reference",
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
