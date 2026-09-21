from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def drop_features_with_missing_values(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Author-style TCGA preprocessing: drop every column containing NaN."""
    removed = X.columns[X.isna().any()].tolist()
    return X.drop(columns=removed).copy(), removed


@dataclass
class FullDatasetCleaningResult:
    X: pd.DataFrame
    constant_features: list[str]
    duplicate_features: list[str]
    imputed_features: list[str]
    imputed_values: int


def clean_upenn_feature_matrix(X: pd.DataFrame) -> FullDatasetCleaningResult:
    """
    Clean the full UPenn feature matrix for feature-selection comparison.

    This project compares feature-processing methods on one prepared UPenn table,
    rather than estimating a clinical model with nested validation. Therefore the
    deterministic structural cleaning is intentionally performed once before the
    three selectors are applied.
    """
    X = X.copy().replace([np.inf, -np.inf], np.nan)

    constant_features = [
        column for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]
    X = X.drop(columns=constant_features)

    duplicate_mask = X.T.duplicated(keep="first")
    duplicate_features = duplicate_mask[duplicate_mask].index.tolist()
    X = X.drop(columns=duplicate_features)

    imputed_features = X.columns[X.isna().any()].tolist()
    imputed_values = int(X.isna().sum().sum())
    if imputed_features:
        medians = X.median(axis=0)
        if medians.isna().any():
            bad = medians[medians.isna()].index.tolist()
            raise ValueError(f"Cannot median-impute all-NaN features: {bad[:20]}")
        X = X.fillna(medians)

    if X.isna().any().any():
        raise ValueError("NaN values remain after UPenn cleaning.")

    return FullDatasetCleaningResult(
        X=X,
        constant_features=constant_features,
        duplicate_features=duplicate_features,
        imputed_features=imputed_features,
        imputed_values=imputed_values,
    )
