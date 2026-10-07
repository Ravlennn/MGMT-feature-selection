"""Patient-level uncertainty and paired comparisons for fixed OOF predictions.

All intervals condition on the fitted outer-fold models. Bootstrapping the
predictions does not repeat model fitting or feature selection.
"""

from __future__ import annotations

from math import erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


METHODS = (
    "le_fscore", "all_features", "calabrese_mi_rfe", "do_xgb_ga",
    "random_9", "random_32", "random_do_k",
)
METHODS_WITHOUT_GA = tuple(
    method for method in METHODS if method not in {"do_xgb_ga", "random_do_k"}
)
COMPARISONS = (
    ("le_fscore", "random_9"),
    ("calabrese_mi_rfe", "random_32"),
    ("do_xgb_ga", "random_do_k"),
    ("le_fscore", "all_features"),
    ("calabrese_mi_rfe", "all_features"),
    ("do_xgb_ga", "all_features"),
    ("le_fscore", "calabrese_mi_rfe"),
    ("le_fscore", "do_xgb_ga"),
    ("calabrese_mi_rfe", "do_xgb_ga"),
)


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    """AUC with midranks for ties; also accepts bootstrap duplicate patients."""
    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = len(y) - n_pos
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def _delong_paired(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired DeLong test for two score vectors on the same patients."""
    positive = y == 1
    x = np.stack((a[positive], b[positive]))
    z = np.stack((a[~positive], b[~positive]))
    n_pos, n_neg = x.shape[1], z.shape[1]
    if min(n_pos, n_neg) < 2:
        raise ValueError("DeLong needs at least two patients in each class.")
    concordance = (x[:, :, None] > z[:, None, :]).astype(float)
    concordance += 0.5 * (x[:, :, None] == z[:, None, :])
    by_positive = concordance.mean(axis=2)
    by_negative = concordance.mean(axis=1)
    difference = by_positive[0].mean() - by_positive[1].mean()
    contributions_positive = by_positive[0] - by_positive[1]
    contributions_negative = by_negative[0] - by_negative[1]
    variance = (np.var(contributions_positive, ddof=1) / n_pos
                + np.var(contributions_negative, ddof=1) / n_neg)
    if variance <= 0:
        return 1.0 if np.isclose(difference, 0) else 0.0
    return erfc(abs(difference) / sqrt(2 * variance))


def _holm(p_values: list[float]) -> list[float]:
    """Holm adjustment across the predefined contrasts in this report."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def summarize_nested_cv(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write conditional AUC CIs and paired contrasts using repeat 0 only.

    Run-level AUC spread is reported separately. Neither GA seeds nor random
    feature repeats are treated as independent patients.
    """
    if bootstrap_samples < 100:
        raise ValueError("At least 100 bootstrap samples are required.")
    selected = tuple(method for method in METHODS if method in set(summary["method"]))
    if selected not in {METHODS, METHODS_WITHOUT_GA}:
        raise ValueError(
            "The statistical report requires either all seven methods or the "
            "five methods selected by --skip-ga."
        )
    comparisons = tuple(
        pair for pair in COMPARISONS if pair[0] in selected and pair[1] in selected
    )

    needed = {"method", "repeat", "SubjectID", "target", "probability", "fold"}
    if needed - set(predictions.columns):
        raise ValueError(f"Missing prediction fields: {sorted(needed - set(predictions.columns))}")
    if predictions.duplicated(["method", "repeat", "SubjectID"]).any():
        raise ValueError("Each method/repeat must have one OOF score per patient.")
    if not np.isfinite(predictions["probability"]).all():
        raise ValueError("Predictions must contain finite probabilities.")

    primary = predictions[predictions["repeat"] == 0]
    reference = primary[primary["method"] == METHODS[0]][
        ["SubjectID", "target", "fold"]
    ].sort_values("SubjectID").reset_index(drop=True)
    if reference.empty or set(reference["target"]) != {0, 1}:
        raise ValueError("Primary OOF predictions require both target classes.")
    ids = reference["SubjectID"].tolist()
    y = reference["target"].to_numpy(dtype=int)
    scores = {}
    for method in selected:
        part = primary[primary["method"] == method].set_index("SubjectID")
        if len(part) != len(ids) or set(part.index) != set(ids):
            raise ValueError(f"Incomplete primary predictions: {method}")
        part = part.loc[ids]
        if not np.array_equal(part["target"].to_numpy(), y):
            raise ValueError(f"Inconsistent target labels: {method}")
        if not np.array_equal(part["fold"].to_numpy(), reference["fold"].to_numpy()):
            raise ValueError(f"Outer fold mismatch: {method}")
        scores[method] = part["probability"].to_numpy(dtype=float)

    counts = summary.groupby("method")["repeat"].nunique()
    if summary.duplicated(["method", "repeat"]).any():
        raise ValueError("Duplicate method/repeat in summary.")
    for method in selected:
        n = int(counts[method])
        repeats = set(summary.loc[summary["method"] == method, "repeat"])
        if repeats != set(range(n)):
            raise ValueError(f"Missing run(s) in summary: {method}")
        part = predictions[predictions["method"] == method]
        if len(part) != len(ids) * n or set(part["repeat"]) != repeats:
            raise ValueError(f"Incomplete repeated OOF predictions: {method}")
        for repeat, run in part.groupby("repeat"):
            if set(run["SubjectID"]) != set(ids):
                raise ValueError(f"Incomplete patient coverage: {method}, {repeat}")
            run = run.set_index("SubjectID").loc[ids]
            if (not np.array_equal(run["target"].to_numpy(), y)
                    or not np.array_equal(run["fold"].to_numpy(),
                                          reference["fold"].to_numpy())):
                raise ValueError(f"Targets or outer folds changed: {method}, {repeat}")
            reported = summary.loc[
                (summary["method"] == method) & (summary["repeat"] == repeat),
                "pooled_oof_auc",
            ].iloc[0]
            if not np.isclose(reported, _auc(y, run["probability"].to_numpy())):
                raise ValueError(f"Summary AUC differs from predictions: {method}, {repeat}")

    # Resample patients (within target strata), using identical draws for all
    # selected methods. Repeats and folds are never bootstrap sampling units.
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    rng = np.random.default_rng(bootstrap_seed)
    samples = {method: np.empty(bootstrap_samples) for method in selected}
    for draw in range(bootstrap_samples):
        indices = np.concatenate((
            rng.choice(positives, size=len(positives), replace=True),
            rng.choice(negatives, size=len(negatives), replace=True),
        ))
        for method in selected:
            samples[method][draw] = _auc(y[indices], scores[method][indices])

    method_rows = []
    for method in selected:
        runs = summary[summary["method"] == method].sort_values("repeat")
        primary_run = runs.iloc[0]
        lower, upper = np.quantile(samples[method], [0.025, 0.975])
        method_rows.append({
            "method": method,
            "patients": len(y),
            "n_repeats": len(runs),
            "primary_repeat": 0,
            "primary_auc": float(roc_auc_score(y, scores[method])),
            "primary_auc_ci_low": lower,
            "primary_auc_ci_high": upper,
            "primary_mean_fold_auc": primary_run["mean_fold_auc"],
            "primary_sd_fold_auc": primary_run["sd_fold_auc"],
            "runs_mean_oof_auc": runs["pooled_oof_auc"].mean(),
            "runs_sd_oof_auc": (runs["pooled_oof_auc"].std(ddof=1)
                                if len(runs) > 1 else np.nan),
            "runs_min_oof_auc": runs["pooled_oof_auc"].min(),
            "runs_max_oof_auc": runs["pooled_oof_auc"].max(),
        })

    contrast_rows = []
    for first, second in comparisons:
        deltas = samples[first] - samples[second]
        lower, upper = np.quantile(deltas, [0.025, 0.975])
        contrast_rows.append({
            "method_a": first, "method_b": second,
            "delta_auc_a_minus_b": _auc(y, scores[first]) - _auc(y, scores[second]),
            "delta_ci_low": lower, "delta_ci_high": upper,
            "delong_p_two_sided": _delong_paired(y, scores[first], scores[second]),
        })
    for row, adjusted in zip(
        contrast_rows, _holm([row["delong_p_two_sided"] for row in contrast_rows])
    ):
        row["delong_p_holm"] = adjusted
        row["n_holm_comparisons"] = len(comparisons)

    method_report = pd.DataFrame(method_rows)
    paired_report = pd.DataFrame(contrast_rows)
    output_dir = Path(output_dir)
    method_report.to_csv(output_dir / "method_inference.csv", index=False)
    paired_report.to_csv(output_dir / "paired_comparisons.csv", index=False)
    return method_report, paired_report
