"""Fold-safe outer evaluation of radiomics feature-selection strategies."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mgmt_features.preprocessing import FoldSafeRadiomicsCleaner
from mgmt_features.selectors.cv_adapters import (
    FScoreTopK,
    MIRFRFEFeatures,
    RandomKFeatures,
    XGBGeneticRFFeatures,
)


@dataclass(frozen=True)
class NestedCVConfig:
    outer_splits: int = 5
    inner_splits: int = 5
    random_repeats: int = 50
    seed: int = 42
    c_values: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    ga_seeds: tuple[int, ...] = (11, 22, 33, 44, 55)
    selector_profile: str = "practical"
    quick: bool = False


def _metadata(config: NestedCVConfig, input_sha256: str, X: pd.DataFrame) -> dict:
    settings = asdict(config)
    settings["c_values"] = list(config.c_values)
    settings["ga_seeds"] = list(config.ga_seeds)
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in [
        root / "nested_cv.py",
        root / "preprocessing.py",
        root / "selectors/cv_adapters.py",
        root / "selectors/fscore.py",
        root / "selectors/xgb_importance.py",
        root / "selectors/genetic_rf.py",
        root / "selectors/mi_rf_rfe.py",
        root / "config.py",
    ]:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "protocol": "nested_cv_v2_three_selectors_and_baselines",
        "config": settings,
        "input_sha256": input_sha256,
        "patients": len(X),
        "raw_features": X.shape[1],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "implementation_sha256": digest.hexdigest(),
    }


def _prepare_output(output_dir: Path, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_metadata.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError(
                f"Existing results in {output_dir} use a different dataset or "
                "configuration. Choose a new output directory."
            )
    else:
        if any(output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {output_dir}")
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _pipeline(
    method: str, seed: int, config: NestedCVConfig, cache_dir: Path,
    matched_k: int | None = None,
) -> Pipeline:
    if method == "le_fscore":
        selector = FScoreTopK(k=9)
    elif method == "random_9":
        selector = RandomKFeatures(k=9, random_state=seed)
    elif method == "random_32":
        selector = RandomKFeatures(k=32, random_state=seed)
    elif method == "random_do_k":
        if matched_k is None or matched_k < 1:
            raise ValueError("random_do_k needs the selected count from Do on this fold.")
        selector = RandomKFeatures(k=matched_k, random_state=seed)
    elif method == "all_features":
        selector = "passthrough"
    elif method == "do_xgb_ga":
        selector = XGBGeneticRFFeatures(
            random_state=seed, profile=config.selector_profile, quick=config.quick
        )
    elif method == "calabrese_mi_rfe":
        selector = MIRFRFEFeatures(
            random_state=seed, profile=config.selector_profile, quick=config.quick
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return Pipeline(
        [
            ("cleaner", FoldSafeRadiomicsCleaner()),
            ("selector", selector),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=5000,
                    random_state=config.seed,
                ),
            ),
        ],
        memory=str(cache_dir),
    )


def _run_one_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    method: str,
    selector_seed: int,
    fold: int,
    config: NestedCVConfig,
    output_dir: Path,
    matched_k: int | None = None,
) -> dict:
    inner_cv = StratifiedKFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=config.seed + fold,
    )
    search = GridSearchCV(
        estimator=_pipeline(
            method, selector_seed, config, output_dir / "pipeline_cache", matched_k
        ),
        param_grid={"classifier__C": list(config.c_values)},
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        error_score="raise",
    )
    started = time.monotonic()
    search.fit(X.iloc[train_idx], y.iloc[train_idx])
    probabilities = search.predict_proba(X.iloc[valid_idx])[:, 1]
    fitted = search.best_estimator_
    cleaner = fitted.named_steps["cleaner"]
    selected = (
        cleaner.get_feature_names_out().tolist()
        if method == "all_features"
        else fitted.named_steps["selector"].get_feature_names_out().tolist()
    )
    selector = fitted.named_steps["selector"]
    return {
        "validation_indices": valid_idx.tolist(),
        "probabilities": probabilities.tolist(),
        "selected_features": selected,
        "cleaned_features": len(cleaner.feature_names_out_),
        "stage1_candidates": int(getattr(selector, "candidate_count_", len(selected))),
        "selector_seed": selector_seed,
        "selector_internal_accuracy": getattr(selector, "internal_score_", None),
        "ga_best_generation": getattr(selector, "best_generation_", None),
        "best_C": float(search.best_params_["classifier__C"]),
        "inner_auc": float(search.best_score_),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _checkpoint_path(output_dir: Path, method: str, repeat: int, fold: int) -> Path:
    return output_dir / "checkpoints" / method / f"repeat_{repeat:03d}_fold_{fold:02d}.json"


def _save_checkpoint(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    subject_ids: pd.Series,
    output_dir: Path,
    input_sha256: str,
    config: NestedCVConfig = NestedCVConfig(),
    methods: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Produce matched outer-fold OOF predictions, with resumable fold checkpoints."""
    if len(X) != len(y) or len(X) != len(subject_ids):
        raise ValueError("X, y and SubjectID must have the same number of rows.")
    if set(y.unique()) != {0, 1} or subject_ids.duplicated().any():
        raise ValueError("Expected both MGMT classes and unique SubjectID values.")
    if config.outer_splits < 2 or config.inner_splits < 2 or config.random_repeats < 1:
        raise ValueError("Splits must be >=2 and random_repeats must be >=1.")
    if any(c <= 0 for c in config.c_values) or not config.c_values:
        raise ValueError("All regularization C values must be positive.")
    if not config.ga_seeds or len(set(config.ga_seeds)) != len(config.ga_seeds):
        raise ValueError("GA seeds must be nonempty and unique.")
    if config.selector_profile not in {"practical", "legacy"}:
        raise ValueError("Unknown selector profile.")

    available = (
        "le_fscore", "all_features", "calabrese_mi_rfe", "do_xgb_ga",
        "random_9", "random_32", "random_do_k",
    )
    requested = methods if methods is not None else available
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(available):
        raise ValueError(f"Choose one or more unique methods from {available}.")
    if "random_do_k" in requested and "do_xgb_ga" in requested:
        requested = tuple(m for m in requested if m != "random_do_k") + ("random_do_k",)
    if "do_xgb_ga" in requested and not config.quick and len(config.ga_seeds) < 5:
        raise ValueError("Full GA comparison requires at least five GA seeds.")

    output_dir = Path(output_dir)
    _prepare_output(output_dir, _metadata(config, input_sha256, X))
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    subject_ids = subject_ids.reset_index(drop=True)
    outer_cv = list(
        StratifiedKFold(
            n_splits=config.outer_splits, shuffle=True, random_state=config.seed
        ).split(X, y)
    )
    run_specs = [
        (method, repeat)
        for method in requested
        for repeat in range(
            config.random_repeats if method.startswith("random_")
            else len(config.ga_seeds) if method == "do_xgb_ga" else 1
        )
    ]

    fold_rows, prediction_rows, feature_rows, summary_rows = [], [], [], []
    for method, repeat in run_specs:
        oof = np.full(len(y), np.nan)
        fold_scores, selected_counts = [], []
        for fold, (train_idx, valid_idx) in enumerate(outer_cv, start=1):
            matched_ga_repeat = (
                repeat % len(config.ga_seeds) if method == "random_do_k" else None
            )
            matched_k = None
            if matched_ga_repeat is not None:
                ga_checkpoint = _checkpoint_path(
                    output_dir, "do_xgb_ga", matched_ga_repeat, fold
                )
                if not ga_checkpoint.exists():
                    raise FileNotFoundError(
                        f"{ga_checkpoint} is required for random_do_k. Run Do first."
                    )
                matched_k = len(json.loads(ga_checkpoint.read_text(
                    encoding="utf-8"
                ))["selected_features"])
            checkpoint = _checkpoint_path(output_dir, method, repeat, fold)
            if checkpoint.exists():
                result = json.loads(checkpoint.read_text(encoding="utf-8"))
                if result["validation_indices"] != valid_idx.tolist():
                    raise ValueError(f"Validation indices changed for {checkpoint}")
                print(f"Reusing {method}, repeat {repeat}, fold {fold}", flush=True)
            else:
                selector_seed = (
                    config.ga_seeds[repeat] if method == "do_xgb_ga"
                    else config.seed + 10000 * repeat + fold
                )
                print(f"Running {method}, repeat {repeat}, fold {fold}", flush=True)
                result = _run_one_fold(
                    X, y, train_idx, valid_idx, method, selector_seed, fold, config,
                    output_dir, matched_k,
                )
                _save_checkpoint(checkpoint, result)

            probabilities = np.asarray(result["probabilities"], dtype=float)
            if len(probabilities) != len(valid_idx) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid probabilities in {checkpoint}")
            oof[valid_idx] = probabilities
            y_valid = y.iloc[valid_idx]
            fold_auc = roc_auc_score(y_valid, probabilities)
            fold_scores.append(fold_auc)
            selected = result["selected_features"]
            selected_counts.append(len(selected))
            fold_rows.append(
                {
                    "method": method, "repeat": repeat, "fold": fold,
                    "matched_ga_repeat": matched_ga_repeat,
                    "train_patients": len(train_idx), "validation_patients": len(valid_idx),
                    "cleaned_features": result["cleaned_features"],
                    "stage1_candidates": result["stage1_candidates"],
                    "selector_seed": result["selector_seed"],
                    "selector_internal_accuracy": result["selector_internal_accuracy"],
                    "ga_best_generation": result["ga_best_generation"],
                    "selected_features": len(selected), "best_C": result["best_C"],
                    "inner_auc": result["inner_auc"], "outer_auc": fold_auc,
                    "outer_auprc": average_precision_score(y_valid, probabilities),
                    "elapsed_seconds": result["elapsed_seconds"],
                }
            )
            for position, feature in enumerate(selected, start=1):
                feature_rows.append(
                    {"method": method, "repeat": repeat, "fold": fold,
                     "position": position, "feature": feature}
                )
            for pos, probability in zip(valid_idx, probabilities):
                prediction_rows.append(
                    {"method": method, "repeat": repeat, "fold": fold,
                     "SubjectID": subject_ids.iloc[pos], "target": int(y.iloc[pos]),
                     "probability": float(probability)}
                )

        if np.isnan(oof).any():
            raise RuntimeError(f"Incomplete OOF predictions: {method}, repeat {repeat}")
        labels = (oof >= 0.5).astype(int)
        summary_rows.append(
            {
                "method": method, "repeat": repeat,
                "matched_ga_repeat": (
                    repeat % len(config.ga_seeds) if method == "random_do_k" else None
                ),
                "mean_fold_auc": float(np.mean(fold_scores)),
                "sd_fold_auc": float(np.std(fold_scores, ddof=1)),
                "pooled_oof_auc": roc_auc_score(y, oof),
                "pooled_oof_auprc": average_precision_score(y, oof),
                "balanced_accuracy": balanced_accuracy_score(y, labels),
                "accuracy": accuracy_score(y, labels),
                "mean_selected_features": float(np.mean(selected_counts)),
            }
        )

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output_dir / "selected_features_by_fold.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary
