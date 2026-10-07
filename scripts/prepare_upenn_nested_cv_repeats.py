"""Reuse completed fold checkpoints when shortening the random baselines.

Only random_repeats changes. Each already completed method/repeat/fold result
is unchanged, so its checkpoint can be moved to a new output directory with
separate run metadata. Generated CSV reports are deliberately not copied.
"""

import argparse
import copy
import json
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "results/upenn/nested_cv/full"
METHODS = ("le_fscore", "all_features", "calabrese_mi_rfe", "random_9", "random_32")
CHECKPOINT_NAME = re.compile(r"repeat_(\d+)_fold_(\d+)\.json")


def main():
    parser = argparse.ArgumentParser(
        description="Reuse existing nested-CV checkpoints with fewer random repeats."
    )
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = (args.output or source.parent / f"repeats_{args.random_repeats}").resolve()
    if args.random_repeats < 1:
        raise ValueError("random-repeats must be positive.")
    if output == source or source in output.parents:
        raise ValueError("Choose a separate output directory outside the source.")
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")

    metadata_path = source / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Original run metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    previous = int(metadata["config"]["random_repeats"])
    if args.random_repeats > previous:
        raise ValueError(f"Requested {args.random_repeats} exceeds the old limit {previous}.")
    if metadata["config"]["quick"]:
        raise ValueError("Source must be the full experiment, not --quick.")

    updated = copy.deepcopy(metadata)
    updated["config"]["random_repeats"] = args.random_repeats
    output.parent.mkdir(parents=True, exist_ok=True)
    copied = {method: 0 for method in METHODS}
    with TemporaryDirectory(prefix="upenn_cv_repeats_", dir=output.parent) as temp:
        stage = Path(temp)
        (stage / "run_metadata.json").write_text(
            json.dumps(updated, indent=2), encoding="utf-8"
        )
        for method in METHODS:
            folder = source / "checkpoints" / method
            if not folder.is_dir():
                continue
            for checkpoint in folder.glob("repeat_*_fold_*.json"):
                match = CHECKPOINT_NAME.fullmatch(checkpoint.name)
                if match is None or int(match.group(1)) >= args.random_repeats:
                    continue
                destination = stage / "checkpoints" / method
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(checkpoint, destination / checkpoint.name)
                copied[method] += 1
        stage.rename(output)
    print(f"Prepared: {output}")
    for method, count in copied.items():
        print(f"{method}: {count} fold checkpoints reused")
    print("Next: run_upenn_nested_cv.py --skip-ga --random-repeats "
          f"{args.random_repeats} --output \"{output}\"")


if __name__ == "__main__":
    main()
