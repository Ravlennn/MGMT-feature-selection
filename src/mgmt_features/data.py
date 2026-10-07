from pathlib import Path

import pandas as pd


AUTHOR_METADATA_COLUMNS = [
    "ID",
    "IDH1_status",
    "MGMT_status",
    "Methyl_class",
    "G-CIMP",
    "Exp_class",
    "Therapy_class",
    "Age",
    "Gender",
]

MGMT_MAPPING = {
    "UNMETHYLATED": 0,
    "METHYLATED": 1,
}

UPENN_METADATA_COLUMNS = [
    "SubjectID",
    "MGMT",
    "target",
]


def load_author_training_data(path: str | Path):
    """Load the 53-patient TCGA-GBM table used for the reference experiments."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    unnamed_columns = [
        column for column in df.columns
        if str(column).startswith("Unnamed:")
    ]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    missing_metadata = [
        column for column in AUTHOR_METADATA_COLUMNS
        if column not in df.columns
    ]
    if missing_metadata:
        raise ValueError(f"Missing expected metadata columns: {missing_metadata}")

    radiomics_columns = [
        column for column in df.columns
        if column not in AUTHOR_METADATA_COLUMNS
    ]
    X = df[radiomics_columns].copy()

    y = (
        df["MGMT_status"]
        .astype(str)
        .str.upper()
        .map(MGMT_MAPPING)
    )
    if y.isna().any():
        unknown = sorted(df.loc[y.isna(), "MGMT_status"].astype(str).unique())
        raise ValueError(f"Unknown MGMT labels: {unknown}")

    return df, X, y.astype(int)


def load_upenn_clean(path: str | Path):
    """Load the canonical UPenn table produced by scripts/build_upenn.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"UPenn processed dataset not found: {path}. "
            "Run: python scripts/build_upenn.py"
        )

    df = pd.read_csv(path)
    missing = set(UPENN_METADATA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing UPenn metadata columns: {sorted(missing)}")

    features = [
        column for column in df.columns
        if column not in UPENN_METADATA_COLUMNS
    ]
    X = df[features].copy()
    y = df["target"].astype(int).copy()

    if X.isna().any().any():
        raise ValueError("Processed UPenn dataset still contains NaN values.")

    return df, X, y


def load_upenn_raw(path: str | Path):
    """Load the merged UPenn cohort before feature cleaning for nested CV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"UPenn uncleaned dataset not found: {path}. "
            "Run: python scripts/build_upenn.py"
        )

    df = pd.read_csv(path)
    missing = set(UPENN_METADATA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing UPenn metadata columns: {sorted(missing)}")
    if df["SubjectID"].isna().any() or df["SubjectID"].duplicated().any():
        raise ValueError("SubjectID must be present and unique for each patient.")
    expected = df["MGMT"].map({"Methylated": 1, "Unmethylated": 0})
    if expected.isna().any() or not expected.equals(df["target"]):
        raise ValueError("MGMT and target must contain consistent binary labels.")

    X = df.drop(columns=UPENN_METADATA_COLUMNS)
    if X.shape[1] == 0:
        raise ValueError("UPenn table contains no radiomics features.")
    return df, X, df["target"].astype(int)
