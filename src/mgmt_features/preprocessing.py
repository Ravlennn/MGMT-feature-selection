from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def drop_features_with_missing_values(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Author-style TCGA preprocessing: drop every column containing NaN."""
    removed = X.columns[X.isna().any()].tolist()
    return X.drop(columns=removed).copy(), removed


class FoldSafeRadiomicsCleaner(TransformerMixin, BaseEstimator):
    """Learn UPenn feature cleaning on a training fold only.

    A fitted cleaner retains the original input feature names, the columns kept
    after train-only constant/duplicate removal, and train-only medians. The
    validation fold contributes nothing to these decisions.
    """

    @staticmethod
    def _validate_frame(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame with named features.")
        if not X.columns.is_unique:
            raise ValueError("X contains duplicate feature names.")
        nonnumeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if nonnumeric:
            raise ValueError(f"X contains nonnumeric features: {nonnumeric[:10]}")
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, X: pd.DataFrame, y=None):
        X = self._validate_frame(X)
        if X.empty or X.shape[1] == 0:
            raise ValueError("Training data must contain patients and features.")

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.constant_features_ = [
            column for column in X.columns if X[column].nunique(dropna=True) <= 1
        ]
        X = X.drop(columns=self.constant_features_)
        duplicate_mask = X.T.duplicated(keep="first")
        self.duplicate_features_ = duplicate_mask[duplicate_mask].index.tolist()
        X = X.drop(columns=self.duplicate_features_)
        if X.shape[1] == 0:
            raise ValueError("No nonconstant, unique features remain in training data.")

        self.feature_names_out_ = X.columns.tolist()
        self.medians_ = X.median(axis=0)
        if self.medians_.isna().any():
            raise ValueError("Cannot calculate training medians for all-NaN features.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        X = self._validate_frame(X)
        missing = set(self.feature_names_out_) - set(X.columns)
        if missing:
            raise ValueError(f"Required features are missing: {sorted(missing)[:10]}")
        X = X.loc[:, self.feature_names_out_].fillna(self.medians_)
        if X.isna().any().any():
            raise ValueError("NaN values remain after training-median imputation.")
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "feature_names_out_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features must match those seen during fit.")
        return np.asarray(self.feature_names_out_, dtype=object)


@dataclass
class FullDatasetCleaningResult:
    X: pd.DataFrame
    constant_features: list[str]
    duplicate_features: list[str]
    imputed_features: list[str]
    imputed_values: int


def clean_upenn_feature_matrix(X: pd.DataFrame) -> FullDatasetCleaningResult:
    """
    Legacy full-cohort cleaning for exploratory feature-selection comparisons.

    For unbiased outer-fold evaluation, use FoldSafeRadiomicsCleaner inside the
    nested CV pipeline on the uncleaned merged cohort instead.
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
