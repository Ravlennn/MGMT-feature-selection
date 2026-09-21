import pandas as pd
from xgboost import XGBClassifier

from mgmt_features.config import RANDOM_STATE, XGB_PARAMS


def compute_xgb_importance(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """XGBoost gain ranking used as the first stage of Do et al."""
    model = XGBClassifier(
        **XGB_PARAMS,
        random_state=random_state,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(X, y)
    importance = model.get_booster().get_score(importance_type="gain")

    result = pd.DataFrame(
        {
            "feature": X.columns,
            "gain": [float(importance.get(feature, 0.0)) for feature in X.columns],
        }
    )
    result = result.sort_values(
        ["gain", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result
