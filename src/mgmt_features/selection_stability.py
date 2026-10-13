"""Patient-bootstrap stability of UPenn radiomics feature selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from mgmt_features.preprocessing import FoldSafeRadiomicsCleaner
from mgmt_features.selectors.cv_adapters import (
    FScoreTopK,
    MIRFRFEFeatures,
    XGBGeneticRFFeatures,
)


METHODS = ("le_fscore", "calabrese_mi_rfe", "do_xgb_ga")


@dataclass(frozen=True)
class StabilityConfig:
    draws: int = 20
    bootstrap_seed: int = 2026
    selector_seed: int = 11
    selector_profile: str = "practical"


def _implementation_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for relative in (
        "selection_stability.py",
        "preprocessing.py",
        "selectors/cv_adapters.py",
        "selectors/fscore.py",
        "selectors/mi_rf_rfe.py",
        "selectors/genetic_rf.py",
        "selectors/xgb_importance.py",
        "config.py",
    ):
        digest.update(relative.encode("utf-8"))
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def _bootstrap_indices(y: pd.Series, config: StabilityConfig) -> list[list[int]]:
    """Resample patients with replacement within each MGMT class."""
    labels = y.to_numpy(dtype=int)
    rng = np.random.default_rng(config.bootstrap_seed)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) < 2 for indices in class_indices):
        raise ValueError("Both MGMT classes need at least two patients.")
    draws = []
    for _ in range(config.draws):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        draws.append(rng.permutation(sampled).tolist())
    return draws


def _selector(method: str, config: StabilityConfig):
    if method == "le_fscore":
        return FScoreTopK(k=9)
    if method == "calabrese_mi_rfe":
        return MIRFRFEFeatures(
            random_state=config.selector_seed, profile=config.selector_profile
        )
    if method == "do_xgb_ga":
        return XGBGeneticRFFeatures(
            random_state=config.selector_seed, profile=config.selector_profile
        )
    raise ValueError(f"Unknown feature selector: {method}")


def _write_json(path: Path, value: dict | list) -> None:
    """Replace a complete checkpoint atomically so interrupted writes can be resumed."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _summarize(output_dir: Path, draws: int, feature_names: set[str]) -> pd.DataFrame:
    summary_rows, selected_rows, frequency_rows, pair_rows = [], [], [], []
    for method in METHODS:
        paths = [output_dir / "checkpoints" / method / f"draw_{i:03d}.json"
                 for i in range(draws)]
        if not all(path.is_file() for path in paths):
            continue
        checkpoints = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        sets = []
        for draw, result in enumerate(checkpoints):
            selected = result["selected_features"]
            if (result["draw"] != draw or not selected or
                    len(selected) != len(set(selected)) or
                    not set(selected) <= feature_names):
                raise ValueError(f"Invalid checkpoint for {method}, draw {draw}.")
            selected_set = set(selected)
            sets.append(selected_set)
            selected_rows.extend(
                {"method": method, "draw": draw, "feature": feature}
                for feature in selected
            )

        for draw_a, draw_b in combinations(range(draws), 2):
            intersection = len(sets[draw_a] & sets[draw_b])
            union = len(sets[draw_a] | sets[draw_b])
            pair_rows.append({
                "method": method, "draw_a": draw_a, "draw_b": draw_b,
                "intersection": intersection, "union": union,
                "jaccard": intersection / union,
            })
        scores = np.array([row["jaccard"] for row in pair_rows if row["method"] == method])
        counts = np.array([len(features) for features in sets])
        frequencies = pd.Series([feature for features in sets for feature in features]).value_counts()
        frequency_rows.extend(
            {"method": method, "feature": feature, "draws_selected": int(count),
             "selection_frequency": float(count / draws)}
            for feature, count in frequencies.items()
        )
        summary_rows.append({
            "method": method, "bootstrap_draws": draws,
            "pairs": len(scores), "mean_jaccard": float(scores.mean()),
            "median_jaccard": float(np.median(scores)),
            "sd_jaccard": float(scores.std(ddof=1)) if len(scores) > 1 else None,
            "min_jaccard": float(scores.min()), "max_jaccard": float(scores.max()),
            "mean_selected_features": float(counts.mean()),
            "sd_selected_features": float(counts.std(ddof=1)),
            "distinct_features_selected": len(frequencies),
        })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.to_csv(output_dir / "stability_summary.csv", index=False)
        pd.DataFrame(pair_rows).to_csv(output_dir / "pairwise_jaccard.csv", index=False)
        pd.DataFrame(selected_rows).to_csv(output_dir / "selected_features_by_draw.csv", index=False)
        pd.DataFrame(frequency_rows).sort_values(
            ["method", "draws_selected", "feature"], ascending=[True, False, True]
        ).to_csv(output_dir / "selection_frequency.csv", index=False)
    return summary


def run_selection_stability(
    X: pd.DataFrame,
    y: pd.Series,
    subject_ids: pd.Series,
    output_dir: Path,
    input_sha256: str,
    config: StabilityConfig,
    methods: tuple[str, ...] = METHODS,
) -> pd.DataFrame:
    if config.draws < 2:
        raise ValueError("At least two draws are required to compute Jaccard.")
    if config.selector_profile not in ("practical", "legacy"):
        raise ValueError("Unknown selector profile.")
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(METHODS):
        raise ValueError("Choose distinct methods from le_fscore, calabrese_mi_rfe, do_xgb_ga.")
    if not X.columns.is_unique or len(X) != len(y) or len(X) != len(subject_ids):
        raise ValueError("X, y and unique feature names must describe the same patients.")
    if subject_ids.isna().any() or subject_ids.duplicated().any():
        raise ValueError("Patient identifiers must be present and unique.")

    output_dir = Path(output_dir)
    metadata = {
        "protocol": "upenn_stratified_patient_bootstrap_feature_selection_v1",
        "config": asdict(config),
        "input_sha256": input_sha256,
        "implementation_sha256": _implementation_hash(),
        "patients": len(X), "raw_features": X.shape[1],
        "note": "Fixed selector seed within all draws isolates patient sampling variability.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError(f"Dataset, code or settings changed; choose a new output directory: {output_dir}")
    else:
        if any(output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {output_dir}")
        _write_json(metadata_path, metadata)

    draws = _bootstrap_indices(y.reset_index(drop=True), config)
    indices_path = output_dir / "bootstrap_indices.json"
    if indices_path.is_file():
        if json.loads(indices_path.read_text(encoding="utf-8")) != draws:
            raise ValueError("Bootstrap indices do not match the saved run.")
    else:
        _write_json(indices_path, draws)

    subject_ids = subject_ids.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    for method in methods:
        checkpoint_dir = output_dir / "checkpoints" / method
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for draw, indices in enumerate(draws):
            checkpoint = checkpoint_dir / f"draw_{draw:03d}.json"
            if checkpoint.is_file():
                result = json.loads(checkpoint.read_text(encoding="utf-8"))
                if result["draw"] != draw or result["sample_indices"] != indices:
                    raise ValueError(f"Checkpoint samples changed: {checkpoint}")
                print(f"Reusing {method}, draw {draw + 1}/{config.draws}", flush=True)
                continue

            print(f"Running {method}, draw {draw + 1}/{config.draws}", flush=True)
            sample_X = X.iloc[indices].reset_index(drop=True)
            sample_y = y.iloc[indices].reset_index(drop=True)
            cleaner = FoldSafeRadiomicsCleaner()
            sample_clean = cleaner.fit_transform(sample_X)
            selector = _selector(method, config)
            selector.fit(sample_clean, sample_y)
            selected = selector.get_feature_names_out().tolist()
            if not selected or len(selected) != len(set(selected)):
                raise ValueError(f"Empty or duplicate selection for {method}, draw {draw}.")
            _write_json(checkpoint, {
                "method": method, "draw": draw,
                "sample_indices": indices,
                "unique_patients": len(set(indices)),
                "cleaned_features": sample_clean.shape[1],
                "selected_features": selected,
                "selector_seed": config.selector_seed,
                "stage1_candidates": getattr(selector, "candidate_count_", None),
            })

    return _summarize(output_dir, config.draws, set(X.columns))
