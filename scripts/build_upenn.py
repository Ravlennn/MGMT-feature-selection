from pathlib import Path
import json

import pandas as pd

from mgmt_features.preprocessing import clean_upenn_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
CLINICAL_PATH = ROOT / "data/raw/upenn/clinical/UPENN-GBM_clinical_info_v2.1.csv"
RADIOMICS_DIR = ROOT / "data/raw/upenn/radiomics"
OUTPUT_DIR = ROOT / "data/processed/upenn"
OUTPUT_PATH = OUTPUT_DIR / "upenn_clean.csv"

VALID_MGMT = {"Methylated": 1, "Unmethylated": 0}
STRUCTURAL_BLOCKS = [
    "FLAIR_ED", "FLAIR_ET", "FLAIR_NC",
    "T1GD_ED", "T1GD_ET", "T1GD_NC",
    "T1_ED", "T1_ET", "T1_NC",
    "T2_ED", "T2_ET", "T2_NC",
]


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def main():
    clinical = pd.read_csv(CLINICAL_PATH)
    id_column = "SubjectID" if "SubjectID" in clinical.columns else "ID" if "ID" in clinical.columns else None
    if id_column is None or "MGMT" not in clinical.columns:
        raise ValueError("Clinical file must contain ID/SubjectID and MGMT columns.")
    if id_column != "SubjectID":
        clinical = clinical.rename(columns={id_column: "SubjectID"})

    clinical["SubjectID"] = _normalize_id(clinical["SubjectID"])
    clinical = clinical[clinical["MGMT"].isin(VALID_MGMT)].copy()
    clinical["target"] = clinical["MGMT"].map(VALID_MGMT).astype(int)
    merged = clinical[["SubjectID", "MGMT", "target"]].copy()

    print("=" * 80)
    print("BUILD UPENN STRUCTURAL DATASET")
    print("=" * 80)
    print(f"MGMT-labelled clinical patients: {len(merged)}")

    for block in STRUCTURAL_BLOCKS:
        path = RADIOMICS_DIR / f"Radiomic_Features_CaPTk_automaticsegm_{block}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing structural radiomics block: {path}")
        block_df = pd.read_csv(path)
        if "SubjectID" not in block_df.columns:
            raise ValueError(f"SubjectID missing in {path.name}")
        block_df["SubjectID"] = _normalize_id(block_df["SubjectID"])
        if block_df["SubjectID"].duplicated().any():
            raise ValueError(f"Duplicate SubjectID values in {path.name}")

        before = len(merged)
        merged = merged.merge(block_df, on="SubjectID", how="inner", validate="one_to_one")
        after = len(merged)
        print(f"{block:<10} {before:>3} -> {after:>3}")

    metadata = ["SubjectID", "MGMT", "target"]
    raw_feature_columns = [c for c in merged.columns if c not in metadata]
    cleaned = clean_upenn_feature_matrix(merged[raw_feature_columns])

    final_df = pd.concat(
        [merged[metadata].reset_index(drop=True), cleaned.X.reset_index(drop=True)], axis=1
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    summary = {
        "clinical_mgmt_labelled": int(len(clinical)),
        "final_patients": int(len(final_df)),
        "methylated": int((final_df["target"] == 1).sum()),
        "unmethylated": int((final_df["target"] == 0).sum()),
        "raw_structural_features": int(len(raw_feature_columns)),
        "removed_constants": int(len(cleaned.constant_features)),
        "removed_exact_duplicates": int(len(cleaned.duplicate_features)),
        "imputed_features": int(len(cleaned.imputed_features)),
        "imputed_values": int(cleaned.imputed_values),
        "final_features": int(cleaned.X.shape[1]),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nFinal UPenn dataset")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
