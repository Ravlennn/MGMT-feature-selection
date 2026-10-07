"""Scikit-learn adapters for feature selection inside a CV pipeline."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from mgmt_features.config import CALABRESE, UPENN_GA
from mgmt_features.selectors.fscore import compute_f_scores
from mgmt_features.selectors.genetic_rf import GeneticRFSelector
from mgmt_features.selectors.mi_rf_rfe import select_mi_rf_rfe


class FScoreTopK(TransformerMixin, BaseEstimator):
    def __init__(self, k: int = 9):
        self.k = k

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.k < 1:
            raise ValueError("k must be positive.")
        ranking = compute_f_scores(X, y)
        self.selected_features_ = ranking.head(min(self.k, len(ranking)))["feature"].tolist()
        self.candidate_count_ = X.shape[1]
        if not self.selected_features_:
            raise ValueError("No features are available for F-score selection.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


class RandomKFeatures(TransformerMixin, BaseEstimator):
    """Choose K distinct columns without using labels, with a fixed seed."""

    def __init__(self, k: int = 9, random_state: int = 42):
        self.k = k
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        if self.k < 1 or X.shape[1] == 0:
            raise ValueError("k must be positive and X must contain features.")
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(X.shape[1], size=min(self.k, X.shape[1]), replace=False)
        self.selected_features_ = X.columns[np.sort(indices)].tolist()
        self.candidate_count_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


class XGBGeneticRFFeatures(TransformerMixin, BaseEstimator):
    """Fit XGBoost gain and GA-RF only on the data passed to fit."""

    def __init__(self, random_state: int = 42, profile: str = "practical", quick: bool = False):
        self.random_state = random_state
        self.profile = profile
        self.quick = quick

    def fit(self, X: pd.DataFrame, y: pd.Series):
        try:
            from mgmt_features.selectors.xgb_importance import compute_xgb_importance
        except ModuleNotFoundError as exc:
            if exc.name == "xgboost":
                raise ModuleNotFoundError(
                    "Do/GA-RF requires xgboost; install the project with pip install -e ."
                ) from exc
            raise
        ranking = compute_xgb_importance(X, y, random_state=self.random_state)
        candidates = ranking.loc[ranking["gain"] > 0, "feature"].tolist()
        if not candidates:
            raise ValueError("XGBoost found no positive-gain candidates on this fold.")
        if self.quick:
            params = UPENN_GA.copy()
            params.update(population_size=4, generations=1, n_splits=2, n_estimators=5)
        elif self.profile == "practical":
            params = UPENN_GA.copy()
            params.update(population_size=12, generations=5, n_splits=3, n_estimators=50)
        elif self.profile == "legacy":
            params = UPENN_GA.copy()
        else:
            raise ValueError(f"Unknown selector profile: {self.profile}")
        ga = GeneticRFSelector(random_state=self.random_state, **params)
        result = ga.fit(X[candidates], y)
        self.selected_features_ = result.selected_features
        self.candidate_count_ = len(candidates)
        self.internal_score_ = result.best_score
        self.best_generation_ = result.best_generation
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)


class MIRFRFEFeatures(TransformerMixin, BaseEstimator):
    """Fit MI and RF-RFE rankings within the current training fold."""

    def __init__(self, random_state: int = 42, profile: str = "practical", quick: bool = False):
        self.random_state = random_state
        self.profile = profile
        self.quick = quick

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.quick:
            params = dict(mi_features=32, final_features=4, n_splits=2,
                          n_estimators=10, rfe_step=4)
        elif self.profile == "practical":
            params = dict(mi_features=1024, final_features=32, n_splits=3,
                          n_estimators=100, rfe_step=64)
        elif self.profile == "legacy":
            params = CALABRESE.copy()
        else:
            raise ValueError(f"Unknown selector profile: {self.profile}")
        _, selected = select_mi_rf_rfe(X, y, random_state=self.random_state, **params)
        self.selected_features_ = selected["feature"].tolist()
        self.candidate_count_ = min(params["mi_features"], X.shape[1])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        return X.loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)
