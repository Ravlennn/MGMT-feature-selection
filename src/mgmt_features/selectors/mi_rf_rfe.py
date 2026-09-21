from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import StratifiedKFold


def select_mi_rf_rfe(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    mi_features: int = 1024,
    final_features: int = 32,
    n_splits: int = 5,
    n_estimators: int = 1000,
    rfe_step: int = 16,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calabrese-inspired Mutual Information -> RF-RFE feature selection."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")
    if X.isna().any().any():
        raise ValueError("X contains missing values.")

    n_mi = min(mi_features, X.shape[1])
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    mi_ranking = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
    mi_ranking = mi_ranking.sort_values(
        ["mi_score", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    mi_ranking.insert(0, "mi_rank", range(1, len(mi_ranking) + 1))

    candidates = mi_ranking.head(n_mi)["feature"].tolist()
    X_candidates = X[candidates]

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_rankings = []
    for fold, (train_idx, _) in enumerate(cv.split(X_candidates, y), start=1):
        print(f"RF-RFE fold {fold}/{n_splits}")
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state + fold,
            n_jobs=-1,
        )
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=min(final_features, X_candidates.shape[1]),
            step=rfe_step,
        )
        rfe.fit(X_candidates.iloc[train_idx], y.iloc[train_idx])
        fold_rankings.append(
            pd.Series(rfe.ranking_, index=candidates, name=f"fold_{fold}")
        )

    ranks = pd.concat(fold_rankings, axis=1)
    fold_columns = [column for column in ranks.columns if column.startswith("fold_")]
    ranks["average_rank"] = ranks[fold_columns].mean(axis=1)
    ranks["rank_std"] = ranks[fold_columns].std(axis=1)
    result = ranks.reset_index().rename(columns={"index": "feature"})
    result = result.merge(
        mi_ranking[["feature", "mi_rank", "mi_score"]],
        on="feature",
        how="left",
    )
    result = result.sort_values(
        ["average_rank", "mi_rank"], ascending=[True, True]
    ).reset_index(drop=True)
    result.insert(0, "final_rank", range(1, len(result) + 1))
    return result, result.head(min(final_features, len(result))).copy()
