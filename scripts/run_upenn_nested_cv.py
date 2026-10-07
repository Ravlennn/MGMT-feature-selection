"""Three article-derived selectors and baselines with fold-safe UPenn evaluation."""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from mgmt_features.data import load_upenn_raw
from mgmt_features.nested_cv import NestedCVConfig, run_nested_cv
from mgmt_features.nested_statistics import summarize_nested_cv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/processed/upenn/upenn_structural_raw.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Outer-CV comparison of Le, Do, Calabrese and baselines."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--random-repeats", type=int)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=2000,
        help="Patient-level stratified bootstrap draws for conditional 95%% CIs.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--methods", nargs="+",
        choices=["le_fscore", "all_features", "random_9", "random_32",
                 "random_do_k", "do_xgb_ga", "calabrese_mi_rfe"],
        help="Run only these methods; omitting it processes all and reuses checkpoints.",
    )
    selection.add_argument(
        "--skip-ga", action="store_true",
        help="Complete the reduced Le/Calabrese comparison without Do or random_do_k.",
    )
    parser.add_argument(
        "--selector-profile", choices=["practical", "legacy"], default="practical",
        help="Legacy UPenn-project selector parameters are substantially more expensive.",
    )
    parser.add_argument("--ga-seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--quick", action="store_true",
        help="Small smoke configuration; results are not for scientific reporting.",
    )
    args = parser.parse_args()
    output = args.output or ROOT / "results/upenn/nested_cv" / (
        "quick" if args.quick else "full"
    )
    config = NestedCVConfig(
        inner_splits=2 if args.quick else 5,
        random_repeats=(
            args.random_repeats if args.random_repeats is not None
            else (2 if args.quick else 50)
        ),
        c_values=(0.1, 1.0) if args.quick else NestedCVConfig().c_values,
        ga_seeds=tuple(args.ga_seeds) if args.ga_seeds else ((11,) if args.quick else (11, 22, 33, 44, 55)),
        selector_profile=args.selector_profile,
        quick=args.quick,
    )
    df, X, y = load_upenn_raw(args.input)
    methods = (
        ("le_fscore", "all_features", "calabrese_mi_rfe", "random_9", "random_32")
        if args.skip_ga else tuple(args.methods) if args.methods else None
    )
    summary = run_nested_cv(
        X=X,
        y=y,
        subject_ids=df["SubjectID"],
        output_dir=output,
        input_sha256=_sha256(args.input),
        config=config,
        methods=methods,
    )
    print(summary.to_string(index=False))
    if args.methods is None:
        method_report, comparisons = summarize_nested_cv(
            summary,
            pd.read_csv(output / "oof_predictions.csv"),
            output,
            bootstrap_samples=args.bootstrap_samples,
        )
        print("\nMethod inference (fixed repeat 0, patient bootstrap CI):")
        print(method_report.to_string(index=False))
        print("\nPaired AUROC comparisons (positive delta favors method_a):")
        print(comparisons.to_string(index=False))
    print(f"Saved: {output}")
    if args.quick:
        print("Quick mode is only a code smoke check, not a final experiment.")


if __name__ == "__main__":
    main()
