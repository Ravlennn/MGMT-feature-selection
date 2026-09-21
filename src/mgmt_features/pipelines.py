from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mgmt_features.analysis import (
    compare_feature_sets,
    parse_reference_family,
    upenn_composition,
)
from mgmt_features.config import (
    CALABRESE,
    LE_TOP_K,
    RANDOM_STATE,
    REFERENCE_GA,
    UPENN_GA,
)
from mgmt_features.data import load_author_training_data, load_upenn_clean
from mgmt_features.preprocessing import drop_features_with_missing_values
from mgmt_features.selectors.fscore import compute_f_scores
from mgmt_features.selectors.genetic_rf import GeneticRFSelector
from mgmt_features.selectors.mi_rf_rfe import select_mi_rf_rfe
from mgmt_features.selectors.xgb_importance import compute_xgb_importance


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_feature_list(path: Path, features: list[str]) -> None:
    pd.DataFrame({"feature": features}).to_csv(path, index=False)


def run_reference_pipeline(
    data_path: Path,
    output_dir: Path,
    *,
    quick: bool = False,
) -> pd.DataFrame:
    """Run Le, Do and Calabrese-inspired selectors on the 53-patient reference table."""
    output_dir = _ensure(output_dir)
    _, X_raw, y = load_author_training_data(data_path)
    X, removed = drop_features_with_missing_values(X_raw)

    print("=" * 80)
    print("REFERENCE PIPELINE — 53-PATIENT TCGA TABLE")
    print("=" * 80)
    print(f"Patients: {len(X)}")
    print(f"Raw radiomics: {X_raw.shape[1]}")
    print(f"Dropped NaN columns: {len(removed)}")
    print(f"Features used: {X.shape[1]}")

    # Le -----------------------------------------------------------------
    le_dir = _ensure(output_dir / "le")
    le_ranking = compute_f_scores(X, y)
    le_selected = le_ranking.head(LE_TOP_K).copy()
    le_ranking.to_csv(le_dir / "ranking.csv", index=False)
    le_selected.to_csv(le_dir / "selected_features.csv", index=False)

    # Do -----------------------------------------------------------------
    do_dir = _ensure(output_dir / "do")
    xgb_ranking = compute_xgb_importance(X, y, random_state=RANDOM_STATE)
    xgb_candidates_df = xgb_ranking[xgb_ranking["gain"] > 0].copy()
    xgb_candidates = xgb_candidates_df["feature"].tolist()
    xgb_ranking.to_csv(do_dir / "xgb_ranking.csv", index=False)
    xgb_candidates_df.to_csv(do_dir / "xgb_candidates.csv", index=False)

    ga_params = REFERENCE_GA.copy()
    if quick:
        ga_params.update(population_size=12, generations=3, n_estimators=50)
    ga = GeneticRFSelector(random_state=RANDOM_STATE, **ga_params)
    ga_result = ga.fit(X[xgb_candidates], y)
    ga_result.history.to_csv(do_dir / "ga_history.csv", index=False)
    _save_feature_list(do_dir / "selected_features.csv", ga_result.selected_features)

    # Calabrese-inspired --------------------------------------------------
    cal_dir = _ensure(output_dir / "calabrese")
    cal_params = CALABRESE.copy()
    if quick:
        cal_params.update(n_splits=2, n_estimators=50, rfe_step=64)
    cal_ranking, cal_selected = select_mi_rf_rfe(
        X, y, random_state=RANDOM_STATE, **cal_params
    )
    cal_ranking.to_csv(cal_dir / "ranking.csv", index=False)
    cal_selected.to_csv(cal_dir / "selected_features.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "method": "le_fscore",
                "input_features": X.shape[1],
                "stage1_features": X.shape[1],
                "selected_features": len(le_selected),
                "internal_score": None,
                "note": "Le top-9 F-score reproduction",
            },
            {
                "method": "do_xgb_ga",
                "input_features": X.shape[1],
                "stage1_features": len(xgb_candidates),
                "selected_features": len(ga_result.selected_features),
                "internal_score": ga_result.best_score,
                "note": "XGBoost gain > 0 followed by GA-RF",
            },
            {
                "method": "calabrese_mi_rfe",
                "input_features": X.shape[1],
                "stage1_features": min(CALABRESE["mi_features"], X.shape[1]),
                "selected_features": len(cal_selected),
                "internal_score": None,
                "note": "Method sanity check on shared TCGA table; original UCSF cohort not bundled",
            },
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    feature_sets = {
        "le_fscore": set(le_selected["feature"]),
        "do_xgb_ga": set(ga_result.selected_features),
        "calabrese_mi_rfe": set(cal_selected["feature"]),
    }
    pairwise, membership, common = compare_feature_sets(feature_sets)
    comparison_dir = _ensure(output_dir / "comparison")
    pairwise.to_csv(comparison_dir / "pairwise_overlap.csv", index=False)
    membership.to_csv(comparison_dir / "feature_membership.csv", index=False)
    common.to_csv(comparison_dir / "common_all_methods.csv", index=False)

    family_rows = []
    for method, features in feature_sets.items():
        families = pd.Series([parse_reference_family(f) for f in features]).value_counts()
        for family, count in families.items():
            family_rows.append({"method": method, "family": family, "count": int(count)})
    pd.DataFrame(family_rows).to_csv(comparison_dir / "feature_families.csv", index=False)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "patients": len(X),
                "raw_features": X_raw.shape[1],
                "dropped_nan_features": len(removed),
                "clean_features": X.shape[1],
                "quick_mode": quick,
            },
            file,
            indent=2,
        )

    print("\nReference summary:")
    print(summary.to_string(index=False))
    return summary


def run_upenn_pipeline(
    data_path: Path,
    output_dir: Path,
    *,
    quick: bool = False,
) -> pd.DataFrame:
    """Apply the three article-derived feature-selection approaches to one clean UPenn table."""
    output_dir = _ensure(output_dir)
    df, X, y = load_upenn_clean(data_path)

    print("=" * 80)
    print("UPENN FEATURE-SELECTION PIPELINE")
    print("=" * 80)
    print(f"Patients: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(df["MGMT"].value_counts().to_string())

    # Le -----------------------------------------------------------------
    le_dir = _ensure(output_dir / "le")
    le_ranking = compute_f_scores(X, y)
    le_selected = le_ranking.head(LE_TOP_K).copy()
    le_ranking.to_csv(le_dir / "ranking.csv", index=False)
    le_selected.to_csv(le_dir / "selected_features.csv", index=False)

    # Do -----------------------------------------------------------------
    do_dir = _ensure(output_dir / "do")
    xgb_ranking = compute_xgb_importance(X, y, random_state=RANDOM_STATE)
    xgb_candidates_df = xgb_ranking[xgb_ranking["gain"] > 0].copy()
    xgb_candidates = xgb_candidates_df["feature"].tolist()
    xgb_ranking.to_csv(do_dir / "xgb_ranking.csv", index=False)
    xgb_candidates_df.to_csv(do_dir / "xgb_candidates.csv", index=False)

    ga_params = UPENN_GA.copy()
    if quick:
        ga_params.update(population_size=10, generations=2, n_estimators=30)
    ga = GeneticRFSelector(random_state=RANDOM_STATE, **ga_params)
    ga_result = ga.fit(X[xgb_candidates], y)
    ga_result.history.to_csv(do_dir / "ga_history.csv", index=False)
    _save_feature_list(do_dir / "selected_features.csv", ga_result.selected_features)

    # Calabrese -----------------------------------------------------------
    cal_dir = _ensure(output_dir / "calabrese")
    cal_params = CALABRESE.copy()
    if quick:
        cal_params.update(n_splits=2, n_estimators=50, rfe_step=128)
    cal_ranking, cal_selected = select_mi_rf_rfe(
        X, y, random_state=RANDOM_STATE, **cal_params
    )
    cal_ranking.to_csv(cal_dir / "ranking.csv", index=False)
    cal_selected.to_csv(cal_dir / "selected_features.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "method": "le_fscore",
                "input_features": X.shape[1],
                "stage1_features": X.shape[1],
                "selected_features": len(le_selected),
                "internal_score": None,
            },
            {
                "method": "do_xgb_ga",
                "input_features": X.shape[1],
                "stage1_features": len(xgb_candidates),
                "selected_features": len(ga_result.selected_features),
                "internal_score": ga_result.best_score,
            },
            {
                "method": "calabrese_mi_rfe",
                "input_features": X.shape[1],
                "stage1_features": min(CALABRESE["mi_features"], X.shape[1]),
                "selected_features": len(cal_selected),
                "internal_score": None,
            },
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    feature_sets = {
        "le_fscore": set(le_selected["feature"]),
        "do_xgb_ga": set(ga_result.selected_features),
        "calabrese_mi_rfe": set(cal_selected["feature"]),
    }
    pairwise, membership, common = compare_feature_sets(feature_sets)
    comparison_dir = _ensure(output_dir / "comparison")
    pairwise.to_csv(comparison_dir / "pairwise_overlap.csv", index=False)
    membership.to_csv(comparison_dir / "feature_membership.csv", index=False)
    common.to_csv(comparison_dir / "common_all_methods.csv", index=False)
    upenn_composition(feature_sets).to_csv(
        comparison_dir / "feature_composition.csv", index=False
    )

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "patients": len(df),
                "features": X.shape[1],
                "methylated": int((y == 1).sum()),
                "unmethylated": int((y == 0).sum()),
                "quick_mode": quick,
            },
            file,
            indent=2,
        )

    print("\nUPenn summary:")
    print(summary.to_string(index=False))
    return summary
