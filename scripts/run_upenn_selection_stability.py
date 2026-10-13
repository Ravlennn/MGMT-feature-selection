"""Measure how stable the three UPenn feature selectors are under resampling."""

import argparse
import hashlib
from pathlib import Path

from mgmt_features.data import load_upenn_raw
from mgmt_features.selection_stability import (
    METHODS,
    StabilityConfig,
    run_selection_stability,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patient-bootstrap Jaccard stability for UPenn feature selection."
    )
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/processed/upenn/upenn_structural_raw.csv",
    )
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--selector-seed", type=int, default=11)
    parser.add_argument("--selector-profile", choices=("practical", "legacy"), default="practical")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output = args.output or ROOT / "results/upenn/selection_stability" / f"bootstrap_{args.draws}"
    config = StabilityConfig(
        draws=args.draws,
        bootstrap_seed=args.bootstrap_seed,
        selector_seed=args.selector_seed,
        selector_profile=args.selector_profile,
    )
    df, X, y = load_upenn_raw(input_path)
    summary = run_selection_stability(
        X=X,
        y=y,
        subject_ids=df["SubjectID"],
        output_dir=output,
        input_sha256=_sha256(input_path),
        config=config,
        methods=tuple(args.methods),
    )
    if not summary.empty:
        print("\nWithin-method stability (all pairs of bootstrap draws):")
        print(summary.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
