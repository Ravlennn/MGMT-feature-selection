from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "upenn" / "upenn_clean.csv"
RESULTS_DIR = ROOT / "results" / "upenn"
OUTPUT_PATH = RESULTS_DIR / "comparison" / "predictive_comparison.csv"

RANDOM_STATE = 42


def load_selected_features(path: Path) -> list[str]:
    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {path}")

    return df["feature"].astype(str).tolist()


def evaluate_subset(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
) -> dict:
    features = [f for f in features if f in X.columns]

    if not features:
        raise ValueError("No selected features found in UPenn dataset.")

    X_selected = X[features]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                    max_iter=5000,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        model,
        X_selected,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    predictions = (probabilities >= 0.5).astype(int)

    return {
        "selected_features": len(features),
        "roc_auc": roc_auc_score(y, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "accuracy": accuracy_score(y, predictions),
    }


def main():
    print("=" * 80)
    print("UPENN — DOWNSTREAM COMPARISON OF SELECTED FEATURE SETS")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)

    y = df["target"].astype(int)

    metadata_columns = {
        "SubjectID",
        "MGMT",
        "target",
    }

    X = df[
        [
            column
            for column in df.columns
            if column not in metadata_columns
        ]
    ].copy()

    feature_files = {
        "le_fscore": (
            RESULTS_DIR
            / "le"
            / "selected_features.csv"
        ),
        "do_xgb_ga": (
            RESULTS_DIR
            / "do"
            / "selected_features.csv"
        ),
        "calabrese_mi_rfe": (
            RESULTS_DIR
            / "calabrese"
            / "selected_features.csv"
        ),
    }

    rows = []

    for method, path in feature_files.items():
        print(f"\nEvaluating: {method}")

        features = load_selected_features(path)

        metrics = evaluate_subset(
            X=X,
            y=y,
            features=features,
        )

        rows.append(
            {
                "method": method,
                **metrics,
            }
        )

        print(
            f"features={metrics['selected_features']} | "
            f"AUC={metrics['roc_auc']:.4f} | "
            f"balanced_acc={metrics['balanced_accuracy']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        by="roc_auc",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(result.to_string(index=False))

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()