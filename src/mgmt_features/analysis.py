from __future__ import annotations

from itertools import combinations

import pandas as pd


UPENN_FAMILIES = {
    "Intensity": "INTENSITY",
    "Histogram": "HISTOGRAM",
    "Volumetric": "VOLUMETRIC",
    "Morphologic": "MORPHOLOGIC",
    "GLCM": "GLCM",
    "GLRLM": "GLRLM",
    "GLSZM": "GLSZM",
    "NGTDM": "NGTDM",
    "LBP": "LBP",
}


def parse_upenn_feature(feature: str) -> dict[str, str]:
    parts = feature.split("_", 3)
    if len(parts) < 4:
        return {"modality": "OTHER", "region": "OTHER", "family": "OTHER"}

    modality, region, family_raw, _ = parts
    family = UPENN_FAMILIES.get(family_raw, family_raw.upper())

    # Shape is derived from the same segmentation geometry across modalities.
    if family in {"MORPHOLOGIC", "VOLUMETRIC"}:
        modality = "SHAPE_SHARED"

    return {"modality": modality, "region": region, "family": family}


def parse_reference_family(feature: str) -> str:
    if feature.startswith("HISTO_"):
        return "HISTO"
    if feature.startswith("TEXTURE_GLCM_"):
        return "GLCM"
    if feature.startswith("TEXTURE_GLRLM_"):
        return "GLRLM"
    if feature.startswith("TEXTURE_GLSZM_"):
        return "GLSZM"
    if feature.startswith("TEXTURE_NGTDM_"):
        return "NGTDM"
    if feature.startswith("TEXTURE_GLOBAL_"):
        return "GLOBAL_TEXTURE"
    if feature.startswith("INTENSITY_"):
        return "INTENSITY"
    if feature.startswith("VOLUME_"):
        return "VOLUME"
    if feature.startswith("SPATIAL_"):
        return "SPATIAL"
    if feature.startswith("TGM_"):
        return "TGM"
    return "OTHER"


def compare_feature_sets(feature_sets: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return pairwise overlap, membership table and features common to all methods."""
    pairwise_rows = []
    for method_a, method_b in combinations(feature_sets, 2):
        a = feature_sets[method_a]
        b = feature_sets[method_b]
        intersection = a & b
        union = a | b
        pairwise_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": len(intersection) / len(union) if union else 0.0,
            }
        )

    all_features = sorted(set().union(*feature_sets.values()))
    membership_rows = []
    for feature in all_features:
        row = {"feature": feature}
        for method, values in feature_sets.items():
            row[method] = int(feature in values)
        row["selected_by_methods"] = sum(row[method] for method in feature_sets)
        membership_rows.append(row)

    common = sorted(set.intersection(*feature_sets.values())) if feature_sets else []
    return (
        pd.DataFrame(pairwise_rows),
        pd.DataFrame(membership_rows).sort_values(
            ["selected_by_methods", "feature"], ascending=[False, True]
        ),
        pd.DataFrame({"feature": common}),
    )


def upenn_composition(feature_sets: dict[str, set[str]]) -> pd.DataFrame:
    rows = []
    for method, features in feature_sets.items():
        parsed = pd.DataFrame([parse_upenn_feature(feature) for feature in features])
        for dimension in ["family", "modality", "region"]:
            counts = parsed[dimension].value_counts()
            for value, count in counts.items():
                rows.append(
                    {
                        "method": method,
                        "dimension": dimension,
                        "value": value,
                        "count": int(count),
                        "fraction": float(count / len(features)) if features else 0.0,
                    }
                )
    return pd.DataFrame(rows)
