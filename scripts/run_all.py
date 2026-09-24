import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, quick: bool = False):
    command = [
        sys.executable,
        str(ROOT / "scripts" / script),
    ]

    if (
        quick
        and script
        in {"run_reference.py", "run_upenn.py"}
    ):
        command.append("--quick")

    print(
        "\n$ " + " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build data, run feature selection, "
            "evaluate UPenn subsets and generate results summary."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke run with reduced GA/RF-RFE settings."
        ),
    )

    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help=(
            "Reuse existing validated reference results."
        ),
    )

    args = parser.parse_args()

    run("build_upenn.py")

    if not args.skip_reference:
        run(
            "run_reference.py",
            quick=args.quick,
        )

    run(
        "run_upenn.py",
        quick=args.quick,
    )

    # Must run before build_report.py because results_summary.md
    # now includes the ROC-AUC comparison table.
    run("evaluate_upenn_selected.py")

    run("build_report.py")


if __name__ == "__main__":
    main()
