"""Audit cross-cohort selection overlap without equating unlike radiomic features.

Run on already saved nested-CV and patient-bootstrap selections; no model fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("le_fscore", "do_xgb_ga", "calabrese_mi_rfe")
REFERENCE_FILES = {
    "le_fscore": ROOT / "results/reference/le/selected_features.csv",
    "do_xgb_ga": ROOT / "results/reference/do/selected_features.csv",
    "calabrese_mi_rfe": ROOT / "results/reference/calabrese/selected_features.csv",
}
REF_TEXTURE = re.compile(
    r"^TEXTURE_(GLCM|GLRLM|GLSZM|NGTDM)_(ED|ET|NET)_(FLAIR|T1GD|T1Gd|T1|T2)_(.+)$"
)
REF_HISTOGRAM = re.compile(r"^HISTO_(ED|ET|NET)_(FLAIR|T1GD|T1Gd|T1|T2)_(.+)$")
UPENN = re.compile(
    r"^(FLAIR|T1GD|T1|T2)_(ED|ET|NC)_(GLCM|GLRLM|GLSZM|NGTDM|Histogram)_(.+)$"
)
PARAMETER = re.compile(r"^(?:Bins-\d+|Radius-\d+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path, columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Use the existing full run, not --quick.")
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not columns <= set(reader.fieldnames):
            raise ValueError(f"{path} needs columns: {sorted(columns)}")
        result = list(reader)
    if not result:
        raise ValueError(f"No rows in {path}")
    return result


def parse_feature(feature: str, cohort: str) -> dict[str, str]:
    modality = region = family = measure = ""
    if cohort == "reference":
        texture = REF_TEXTURE.fullmatch(feature)
        hist = REF_HISTOGRAM.fullmatch(feature)
        if texture:
            family, region, modality, measure = texture.groups()
        elif hist:
            region, modality, measure = hist.groups()
            family = "HISTOGRAM"
    else:
        matched = UPENN.fullmatch(feature)
        if matched:
            modality, region, family, measure = matched.groups()
            family = family.upper()
            while True:
                stripped = PARAMETER.sub("", measure, count=1)
                if stripped == measure:
                    break
                measure = stripped
    modality = modality.upper()
    broad = f"{modality}|{family}" if family else ""
    strict = f"{modality}|{region}|{family}" if family and region in {"ED", "ET"} else ""
    if not family:
        status = "unsupported_name_or_family"
    elif region in {"NC", "NET"}:
        status = "region_not_equated"
    else:
        status = "block_comparable_only"
    return {
        "cohort": cohort, "feature": feature, "modality": modality,
        "region": region, "family": family, "measure_label": measure,
        "broad_block": broad, "strict_block": strict, "status": status,
    }


def write_csv(path: Path, fields: list[str], data: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def load_reference() -> tuple[dict[str, set[str]], dict[str, str]]:
    selected, hashes = {}, {}
    for method, path in REFERENCE_FILES.items():
        features = [row["feature"].strip() for row in rows(path, {"feature"})]
        if not all(features) or len(features) != len(set(features)):
            raise ValueError(f"Empty or duplicate reference feature: {path}")
        selected[method] = set(features)
        hashes[method] = sha256(path)
    return selected, hashes


def load_upenn_header(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        columns = next(csv.reader(source), [])
    if len(columns) != len(set(columns)) or not {"SubjectID", "MGMT", "target"} <= set(columns):
        raise ValueError(f"Invalid UPenn raw header: {path}")
    return set(columns) - {"SubjectID", "MGMT", "target"}


def load_runs(path: Path, source: str, metadata: dict) -> dict[tuple, set[str]]:
    coordinates = {"method", "feature", "repeat", "fold"} if source == "nested" else {
        "method", "feature", "draw"
    }
    grouped: dict[tuple, set[str]] = defaultdict(set)
    for row in rows(path, coordinates):
        method = row["method"]
        if method not in METHODS:
            continue  # Baselines do not select features by the three article methods.
        if source == "nested":
            key = (source, method, int(row["repeat"]), int(row["fold"]), "")
        else:
            key = (source, method, "", "", int(row["draw"]))
        feature = row["feature"].strip()
        if not feature or feature in grouped[key]:
            raise ValueError(f"Empty or repeated feature in {path}, run {key}")
        grouped[key].add(feature)

    if source == "nested":
        config = metadata["config"]
        if config.get("quick") or len(config.get("ga_seeds", [])) < 5:
            raise ValueError("Nested results must be the full run with at least five GA seeds.")
        folds = range(1, int(config["outer_splits"]) + 1)
        expected = {
            (source, method, repeat, fold, "")
            for method in METHODS
            for repeat in range(len(config["ga_seeds"]) if method == "do_xgb_ga" else 1)
            for fold in folds
        }
    else:
        draws = int(metadata["config"]["draws"])
        if draws < 20:
            raise ValueError("Stability comparison needs at least 20 bootstrap draws.")
        expected = {(source, method, "", "", draw) for method in METHODS for draw in range(draws)}
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped), key=str)
        unexpected = sorted(set(grouped) - expected, key=str)
        raise ValueError(f"Incomplete or unexpected {source} selections in {path}: "
                         f"missing={missing[:5]}, unexpected={unexpected[:5]}")
    return grouped


def read_metadata(directory: Path, raw_sha: str) -> dict:
    path = directory / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("input_sha256") != raw_sha:
        raise ValueError(f"{path} belongs to a different UPenn input file.")
    return metadata


def jaccard(first: set[str], second: set[str]) -> float | None:
    union = first | second
    return len(first & second) / len(union) if union else None


def joined(items: set[str]) -> str:
    return "; ".join(sorted(items))


def metric_label(feature: dict[str, str]) -> str:
    # Only identical printed metric labels are candidate matches. No synonyms
    # or equality of extraction parameters/software are assumed.
    return re.sub(r"[^a-z0-9]", "", feature["measure_label"].lower())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("both", "nested", "bootstrap"), default="both")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/processed/upenn/upenn_structural_raw.csv")
    parser.add_argument("--nested-dir", type=Path, default=ROOT / "results/upenn/nested_cv/repeats_20")
    parser.add_argument("--stability-dir", type=Path, default=ROOT / "results/upenn/selection_stability/bootstrap_20")
    parser.add_argument("--output", type=Path, default=ROOT / "results/upenn/reference_overlap")
    args = parser.parse_args()

    reference, reference_hashes = load_reference()
    raw_features = load_upenn_header(args.raw)
    raw_sha = sha256(args.raw)
    selected_runs: dict[tuple, set[str]] = {}
    source_metadata = {}
    sources = ("nested", "bootstrap") if args.source == "both" else (args.source,)
    for source in sources:
        directory = args.nested_dir if source == "nested" else args.stability_dir
        metadata = read_metadata(directory, raw_sha)
        if int(metadata.get("raw_features", -1)) != len(raw_features):
            raise ValueError(f"Feature count disagrees with {directory / 'run_metadata.json'}")
        selection_file = directory / (
            "selected_features_by_fold.csv" if source == "nested" else "selected_features_by_draw.csv"
        )
        selected_runs.update(load_runs(selection_file, source, metadata))
        source_metadata[source] = {
            "input_sha256": metadata["input_sha256"],
            "implementation_sha256": metadata.get("implementation_sha256"),
            "config": metadata["config"],
            "selection_file_sha256": sha256(selection_file),
        }
    unknown = set().union(*selected_runs.values()) - raw_features
    if unknown:
        raise ValueError(f"Selected UPenn features missing from raw table: {sorted(unknown)[:5]}")

    ref_parsed = {f: parse_feature(f, "reference") for f in set().union(*reference.values())}
    up_parsed = {f: parse_feature(f, "upenn") for f in raw_features}
    raw_strict = defaultdict(set)
    raw_broad = defaultdict(set)
    for feature, parsed in up_parsed.items():
        if parsed["strict_block"]:
            raw_strict[parsed["strict_block"]].add(feature)
        if parsed["broad_block"]:
            raw_broad[parsed["broad_block"]].add(feature)

    availability = []
    for method in METHODS:
        for feature in sorted(reference[method]):
            parsed = ref_parsed[feature]
            availability.append({
                "method": method, "reference_feature": feature,
                "status": parsed["status"],
                "strict_block": parsed["strict_block"],
                "upenn_raw_strict_block_features": len(raw_strict[parsed["strict_block"]])
                if parsed["strict_block"] else "",
                "broad_block": parsed["broad_block"],
                "upenn_raw_broad_block_features": len(raw_broad[parsed["broad_block"]])
                if parsed["broad_block"] else "",
            })

    comparisons, shared, possible = [], [], []
    for (source, method, repeat, fold, draw), up_features in sorted(selected_runs.items(), key=str):
        ref_features = reference[method]
        ref_strict = {ref_parsed[f]["strict_block"] for f in ref_features} - {""}
        up_strict = {up_parsed[f]["strict_block"] for f in up_features} - {""}
        ref_broad = {ref_parsed[f]["broad_block"] for f in ref_features} - {""}
        up_broad = {up_parsed[f]["broad_block"] for f in up_features} - {""}
        common_strict = ref_strict & up_strict
        common_broad = ref_broad & up_broad
        context = {"source": source, "method": method, "repeat": repeat,
                   "fold": fold, "draw": draw}
        comparisons.append({
            **context, "reference_selected_features": len(ref_features),
            "upenn_selected_features": len(up_features),
            "reference_strict_blocks": len(ref_strict),
            "upenn_strict_blocks": len(up_strict),
            "shared_strict_blocks": len(common_strict),
            "strict_block_jaccard": jaccard(ref_strict, up_strict),
            "reference_strict_blocks_in_upenn_raw": len(ref_strict & raw_strict.keys()),
            "reference_broad_blocks": len(ref_broad),
            "upenn_broad_blocks": len(up_broad),
            "shared_broad_blocks": len(common_broad),
            "broad_block_jaccard": jaccard(ref_broad, up_broad),
        })
        for level, matched in (("strict", common_strict), ("broad", common_broad)):
            field = "strict_block" if level == "strict" else "broad_block"
            for block in sorted(matched):
                ref_at_block = {f for f in ref_features if ref_parsed[f][field] == block}
                up_at_block = {f for f in up_features if up_parsed[f][field] == block}
                shared.append({**context, "level": level, "block": block,
                               "reference_features": joined(ref_at_block),
                               "upenn_features": joined(up_at_block)})
                if level == "strict":
                    for ref in sorted(ref_at_block):
                        for up in sorted(up_at_block):
                            if (ref_parsed[ref]["family"] != "HISTOGRAM"
                                    and metric_label(ref_parsed[ref])
                                    and metric_label(ref_parsed[ref]) == metric_label(up_parsed[up])):
                                possible.append({**context, "block": block,
                                                 "reference_feature": ref,
                                                 "upenn_feature": up,
                                                 "status": "same_metric_label_unverified"})

    summary = []
    for source in sources:
        for method in METHODS:
            group = [row for row in comparisons if row["source"] == source and row["method"] == method]
            strict_scores = [row["strict_block_jaccard"] for row in group
                             if row["strict_block_jaccard"] is not None]
            broad_scores = [row["broad_block_jaccard"] for row in group
                            if row["broad_block_jaccard"] is not None]
            summary.append({
                "source": source, "method": method, "runs": len(group),
                "mean_shared_strict_blocks": mean(row["shared_strict_blocks"] for row in group),
                "mean_strict_block_jaccard": mean(strict_scores) if strict_scores else "",
                "sd_strict_block_jaccard": stdev(strict_scores) if len(strict_scores) > 1 else "",
                "mean_broad_block_jaccard": mean(broad_scores) if broad_scores else "",
                "reference_strict_blocks": group[0]["reference_strict_blocks"],
                "reference_strict_blocks_in_upenn_raw": group[0]["reference_strict_blocks_in_upenn_raw"],
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "comparison_by_run.csv", list(comparisons[0]), comparisons)
    write_csv(args.output / "summary.csv", list(summary[0]), summary)
    write_csv(args.output / "shared_blocks.csv",
              ["source", "method", "repeat", "fold", "draw", "level", "block",
               "reference_features", "upenn_features"], shared)
    write_csv(args.output / "possible_metric_matches.csv",
              ["source", "method", "repeat", "fold", "draw", "block",
               "reference_feature", "upenn_feature", "status"], possible)
    write_csv(args.output / "reference_availability.csv", list(availability[0]), availability)
    audit = [ref_parsed[f] for f in sorted(ref_parsed)] + [up_parsed[f] for f in sorted(up_parsed)]
    write_csv(args.output / "feature_name_audit.csv", list(audit[0]), audit)
    (args.output / "run_metadata.json").write_text(json.dumps({
        "protocol": "cross_cohort_blocks_v1_no_feature_identity_claim",
        "raw_input_sha256": raw_sha,
        "reference_selection_sha256": reference_hashes,
        "upenn_sources": source_metadata,
        "strict_regions": ["ED", "ET"],
        "excluded_region_equivalence": "NET != NC unless independently verified",
        "no_verified_one_to_one_feature_mapping": True,
    }, indent=2), encoding="utf-8")
    print("\nCross-cohort block comparison (descriptive only):")
    for row in summary:
        print(f"{row['source']:9s} {row['method']:18s} runs={row['runs']:2d} "
              f"mean_strict_J={row['mean_strict_block_jaccard']:.3f} "
              f"mean_broad_J={row['mean_broad_block_jaccard']:.3f}")
    print(f"Saved: {args.output}")
    print("Block overlap is not feature-level overlap; metric-name matches need manual verification.")


if __name__ == "__main__":
    main()
