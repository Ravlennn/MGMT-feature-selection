from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/reference"
UPENN = ROOT / "results/upenn"
OUT = ROOT / "docs/results_summary.md"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = [str(column) for column in df.columns]
    rows = [headers] + [[str(value) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    def fmt(row):
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    return "\n".join([fmt(headers), separator] + [fmt(row) for row in rows[1:]])


def main():
    required = [
        REFERENCE / "summary.csv",
        UPENN / "summary.csv",
        UPENN / "comparison/pairwise_overlap.csv",
        UPENN / "comparison/feature_composition.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pipeline outputs:\n" + "\n".join(missing) +
            "\nRun scripts/run_reference.py and scripts/run_upenn.py first."
        )

    reference = pd.read_csv(REFERENCE / "summary.csv")
    upenn = pd.read_csv(UPENN / "summary.csv")
    pairwise = pd.read_csv(UPENN / "comparison/pairwise_overlap.csv")
    composition = pd.read_csv(UPENN / "comparison/feature_composition.csv")

    upenn_meta = {}
    meta_path = UPENN / "metadata.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as file:
            upenn_meta = json.load(file)

    lines = [
        "# MGMT radiomics feature-selection — generated results summary",
        "",
        "## Study goal",
        "",
        "Compare three article-derived radiomics feature-processing approaches: Le F-score, Do XGBoost→GA-RF, and Calabrese-inspired MI→RF-RFE. Methods are first checked on the small reference table and then applied to one common UPenn-GBM structural-radiomics dataset.",
        "",
        "## Reference-data results",
        "",
        _markdown_table(reference),
        "",
        "**Interpretation note.** Le and Do are reproduced/checked on the shared 53-patient TCGA table. The Calabrese pipeline is a method sanity check on that table because the original UCSF feature matrix is not bundled here.",
        "",
        "## UPenn dataset",
        "",
    ]

    if upenn_meta:
        lines.extend(
            [
                f"- Patients: {upenn_meta.get('patients', 'n/a')}",
                f"- Methylated: {upenn_meta.get('methylated', 'n/a')}",
                f"- Unmethylated: {upenn_meta.get('unmethylated', 'n/a')}",
                f"- Clean radiomics features: {upenn_meta.get('features', 'n/a')}",
                "",
            ]
        )

    lines.extend(
        [
            "## UPenn feature-selection results",
            "",
            _markdown_table(upenn),
            "",
            "The Do `internal_score` is the GA Random-Forest cross-validation fitness used by the selector; it is not an external generalization metric.",
            "",
            "## Pairwise overlap on UPenn",
            "",
            _markdown_table(pairwise),
            "",
            "## Selected-feature composition on UPenn",
            "",
        ]
    )

    for dimension in ["family", "modality", "region"]:
        part = composition[composition["dimension"] == dimension].copy()
        lines.extend([f"### {dimension.title()}", "", _markdown_table(part), ""])

    lines.extend(
        [
            "## What to discuss in the report",
            "",
            "1. How aggressively each method reduces the feature space.",
            "2. Which feature families, MRI modalities and tumor regions dominate each selected subset.",
            "3. How much the three methods overlap on the same UPenn cohort.",
            "4. Differences between the small reference dataset and the larger UPenn dataset.",
            "5. Method limitations: small original cohorts, stochastic GA behavior, and the fact that feature selection here is the primary object of study rather than a fully validated clinical classifier.",
            "",
        ]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated: {OUT}")


if __name__ == "__main__":
    main()
