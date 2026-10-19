"""Compare individual selected feature names across reference and UPenn runs.

Matches identify candidate metric names, not verified equality of extraction
formulas, discretization, software versions or (for NET/NC) segmentations.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from compare_reference_upenn import (
    METHODS, ROOT, jaccard, load_reference, load_runs, load_upenn_header,
    parse_feature, read_metadata, rows, sha256, write_csv,
)


# Explicit, inspectable aliases; unknown metric labels are not matched.
ALIASES = {
    "GLCM": {
        "AutoCorrelation": "AutoCorrelation", "Correlation": "Correlation",
        "Energy": "Energy", "Entropy": "Entropy", "Homogeneity": "Homogeneity",
        "Contrast": "Contrast",
    },
    "GLRLM": {
        "SRLGE": "ShortRunLowGreyLevelEmphasis",
        "LRLGE": "LongRunLowGreyLevelEmphasis",
        "RLN": "RunLengthNonuniformity", "SRE": "ShortRunEmphasis",
        "GLV": "GreyLevelVariance", "RLV": "RunLengthVariance",
        "RP": "RunPercentage",
    },
    "GLSZM": {
        "SZE": "SmallZoneEmphasis", "ZSN": "ZoneSizeNonUniformity",
        "ZP": "ZonePercentage", "GLN": "GreyLevelNonUniformity",
        "HGZE": "HighGreyLevelEmphasis", "LGZE": "LowGreyLevelEmphasis",
        "SZHGE": "SmallZoneHighGreyLevelEmphasis",
        "SZLGE": "SmallZoneLowGreyLevelEmphasis",
        "LZHGE": "LargeZoneHighGreyLevelEmphasis",
        "ZSV": "ZoneSizeVariance",
    },
    "NGTDM": {"Contrast": "Contrast"},
    "INTENSITY": {"Mean": "Mean"},
}

UPENN_NAME = re.compile(
    r"^(FLAIR|T1GD|T1|T2)_(ED|ET|NC)_(GLCM|GLRLM|GLSZM|NGTDM|Intensity)_(.+)$"
)
PARAMETER = re.compile(r"^(?:Bins-\d+|Radius-\d+)_")
REF_INTENSITY = re.compile(r"^INTENSITY_(Mean)_(ED|ET|NET)_(FLAIR|T1Gd|T1|T2)$")
CTX = ["source", "method", "repeat", "fold", "draw", "scenario"]


def canonical(feature: str, cohort: str, scenario: str) -> tuple[str, str]:
    """Return candidate identity and reason when a name cannot be mapped."""
    if cohort == "reference":
        intensity = REF_INTENSITY.fullmatch(feature)
        if intensity:
            metric, region, modality = intensity.groups()
            family = "INTENSITY"
        else:
            parsed = parse_feature(feature, cohort)
            modality, region, family, metric = (
                parsed[field] for field in ("modality", "region", "family", "measure_label")
            )
            if family not in ALIASES:
                return "", "unsupported_family_or_histogram_ambiguous"
            metric = ALIASES[family].get(metric, "")
            if not metric:
                return "", "unknown_reference_metric"
    else:
        match = UPENN_NAME.fullmatch(feature)
        if not match:
            return "", "unsupported_upenn_family"
        modality, region, family, metric = match.groups()
        family = family.upper()
        while True:
            stripped = PARAMETER.sub("", metric, count=1)
            if stripped == metric:
                break
            metric = stripped
        if metric not in ALIASES[family].values():
            return "", "upenn_metric_not_in_alias_map"

    if region in ("NET", "NC"):
        if scenario == "strict_ed_et":
            return "", "NET_NC_segmentation_not_equated"
        region = "NC"  # Exploratory candidate only, never a verified segment match.
    return f"{modality.upper()}|{region}|{family}|{metric}", "candidate_name_match_only"


def unique_map(features: set[str], cohort: str, scenario: str) -> tuple[dict[str, str], dict[str, str]]:
    keys: dict[str, list[str]] = defaultdict(list)
    reasons = {}
    for feature in sorted(features):
        key, reason = canonical(feature, cohort, scenario)
        if key:
            keys[key].append(feature)
        else:
            reasons[feature] = reason
    duplicates = {key for key, names in keys.items() if len(names) > 1}
    for key in duplicates:
        for name in keys[key]:
            reasons[name] = "ambiguous_multiple_columns_same_key"
    return {key: names[0] for key, names in keys.items() if key not in duplicates}, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("both", "nested", "bootstrap"), default="both")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/processed/upenn/upenn_structural_raw.csv")
    parser.add_argument("--nested-dir", type=Path, default=ROOT / "results/upenn/nested_cv/repeats_20")
    parser.add_argument("--stability-dir", type=Path, default=ROOT / "results/upenn/selection_stability/bootstrap_20")
    parser.add_argument("--output", type=Path, default=ROOT / "results/upenn/reference_feature_overlap")
    args = parser.parse_args()

    reference, reference_hashes = load_reference()
    ranking = ROOT / "results/reference/le/ranking.csv"
    reference_raw = {r["feature"].strip() for r in rows(ranking, {"feature"})}
    if not set().union(*reference.values()) <= reference_raw:
        raise ValueError("Reference selections absent from the 704-feature ranking")
    upenn_raw = load_upenn_header(args.raw)
    raw_sha = sha256(args.raw)
    selected_runs = {}
    input_metadata = {}
    sources = ("nested", "bootstrap") if args.source == "both" else (args.source,)
    for source in sources:
        directory = args.nested_dir if source == "nested" else args.stability_dir
        meta = read_metadata(directory, raw_sha)
        if int(meta.get("raw_features", -1)) != len(upenn_raw):
            raise ValueError(f"UPenn column count differs from {directory}/run_metadata.json")
        file = directory / ("selected_features_by_fold.csv" if source == "nested"
                            else "selected_features_by_draw.csv")
        selected_runs.update(load_runs(file, source, meta))
        input_metadata[source] = {"selection_sha256": sha256(file),
                                  "implementation_sha256": meta.get("implementation_sha256")}
    unknown = set().union(*selected_runs.values()) - upenn_raw
    if unknown:
        raise ValueError(f"Selected UPenn columns missing in raw table: {sorted(unknown)[:5]}")

    per_run, matches, coverage, maps = [], [], [], []
    for scenario in ("strict_ed_et", "exploratory_NET_as_NC"):
        ref_map, ref_reasons = unique_map(reference_raw, "reference", scenario)
        up_map, up_reasons = unique_map(upenn_raw, "upenn", scenario)
        universe = set(ref_map) & set(up_map)
        for feature in sorted(set().union(*reference.values())):
            key, reason = canonical(feature, "reference", scenario)
            coverage.append({"scenario": scenario, "reference_feature": feature,
                             "candidate_key": key, "status": ref_reasons.get(feature)
                             or ("available_in_both_raw_matrices" if key in universe
                                 else "metric_absent_in_upenn_raw")})
        for key in sorted(universe):
            maps.append({"scenario": scenario, "candidate_key": key,
                         "reference_raw_feature": ref_map[key], "upenn_raw_feature": up_map[key],
                         "status": "same_metric_name_unverified_extraction"})

        for (source, method, repeat, fold, draw), up_features in sorted(selected_runs.items(), key=str):
            ref_features = reference[method]
            r = {key: ref_map[key] for key in universe if ref_map[key] in ref_features}
            u = {key: up_map[key] for key in universe if up_map[key] in up_features}
            common = set(r) & set(u)
            ctx = dict(zip(CTX, (source, method, repeat, fold, draw, scenario)))
            per_run.append({**ctx, "reference_selected_total": len(ref_features),
                            "upenn_selected_total": len(up_features),
                            "shared_raw_candidate_universe": len(universe),
                            "reference_selected_mappable": len(r),
                            "upenn_selected_mappable": len(u),
                            "shared_individual_features": len(common),
                            # No evaluable reference denominator means no score, not zero.
                            "selected_feature_jaccard": jaccard(set(r), set(u)) if r else "",
                            "reference_selected_recall": len(common) / len(r) if r else ""})
            for key in sorted(common):
                matches.append({**ctx, "candidate_key": key,
                                "reference_feature": r[key], "upenn_feature": u[key],
                                "status": "same_metric_name_unverified_extraction"})

    summary = []
    for scenario in ("strict_ed_et", "exploratory_NET_as_NC"):
        for source in sources:
            for method in METHODS:
                group = [r for r in per_run if r["scenario"] == scenario and
                         r["source"] == source and r["method"] == method]
                scores = [r["selected_feature_jaccard"] for r in group
                          if r["selected_feature_jaccard"] != ""]
                summary.append({"scenario": scenario, "source": source, "method": method,
                                "runs": len(group), "shared_raw_candidate_universe":
                                group[0]["shared_raw_candidate_universe"],
                                "reference_selected_total": group[0]["reference_selected_total"],
                                "reference_selected_mappable": group[0]["reference_selected_mappable"],
                                "mean_upenn_selected_mappable": mean(r["upenn_selected_mappable"] for r in group),
                                "mean_shared_individual_features": mean(r["shared_individual_features"] for r in group),
                                "mean_selected_feature_jaccard": mean(scores) if scores else "",
                                "sd_selected_feature_jaccard": stdev(scores) if len(scores) > 1 else ""})

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "summary.csv", list(summary[0]), summary)
    write_csv(args.output / "overlap_by_run.csv", list(per_run[0]), per_run)
    write_csv(args.output / "shared_individual_features.csv",
              CTX + ["candidate_key", "reference_feature", "upenn_feature", "status"], matches)
    write_csv(args.output / "reference_selected_coverage.csv", list(coverage[0]), coverage)
    write_csv(args.output / "candidate_raw_feature_map.csv",
              ["scenario", "candidate_key", "reference_raw_feature", "upenn_raw_feature", "status"], maps)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "protocol": "candidate_individual_feature_name_overlap_v1",
        "raw_upenn_sha256": raw_sha, "reference_ranking_sha256": sha256(ranking),
        "reference_selection_sha256": reference_hashes,
        "upenn_runs": input_metadata,
        "warning": "Names only: extraction formulas, discretization and NET/NC equivalence not verified.",
    }, indent=2), encoding="utf-8")
    for r in summary:
        print(f"{r['scenario']} {r['source']} {r['method']}: "
              f"{r['mean_shared_individual_features']:.2f} shared features/run, "
              f"J={r['mean_selected_feature_jaccard']}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
