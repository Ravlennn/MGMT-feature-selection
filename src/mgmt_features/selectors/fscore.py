import numpy as np
import pandas as pd


def compute_f_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """F-score ranking used in the Le et al. reproduction."""
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if X.isna().any().any():
        raise ValueError("X contains missing values.")
    if set(pd.Series(y).unique()) != {0, 1}:
        raise ValueError("Expected binary target {0, 1}.")

    y = pd.Series(np.asarray(y), index=X.index)
    positive = X.loc[y == 1]
    negative = X.loc[y == 0]

    mean_all = X.mean(axis=0)
    numerator = (
        (positive.mean(axis=0) - mean_all) ** 2
        + (negative.mean(axis=0) - mean_all) ** 2
    )
    denominator = positive.var(axis=0, ddof=1) + negative.var(axis=0, ddof=1)
    scores = (numerator / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ranking = pd.DataFrame({"feature": X.columns, "f_score": scores.to_numpy()})
    ranking = ranking.sort_values(
        ["f_score", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking
