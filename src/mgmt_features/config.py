"""Single place for the experiment parameters used by the pipelines."""

RANDOM_STATE = 42

# Le et al.
LE_TOP_K = 9

# Do et al. XGBoost stage.
XGB_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "learning_rate": 0.3,
    "gamma": 0,
    "max_depth": 6,
    "reg_lambda": 1,
    "n_estimators": 100,
}

# GA parameters used for the compact reproduction on the 53-patient table.
REFERENCE_GA = {
    "population_size": 100,
    "generations": 20,
    "crossover_rate": 0.8,
    "mutation_rate": 0.05,
    "initial_feature_probability": 0.5,
    "elite_size": 2,
    "n_splits": 5,
    "n_estimators": 100,
}

# Computationally practical UPenn application of the same GA-RF idea.
UPENN_GA = {
    "population_size": 50,
    "generations": 20,
    "crossover_rate": 0.8,
    "mutation_rate": 0.05,
    "initial_feature_probability": 0.5,
    "elite_size": 2,
    "n_splits": 5,
    "n_estimators": 100,
}

# Calabrese-inspired MI -> RF-RFE settings.
CALABRESE = {
    "mi_features": 1024,
    "final_features": 32,
    "n_splits": 5,
    "n_estimators": 1000,
    "rfe_step": 16,
}
