from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REFERENCE = ROOT / "results" / "reference"
UPENN = ROOT / "results" / "upenn"
OUT = ROOT / "docs" / "results_summary.md"


def _format_value(value):
    """Format values exactly enough for a compact Markdown report."""
    if pd.isna(value):
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a simple aligned Markdown table."""
    if df.empty:
        return "_Нет данных._"

    headers = [str(column) for column in df.columns]

    rows = [
        headers,
        *[
            [_format_value(value) for value in row]
            for row in df.itertuples(index=False, name=None)
        ],
    ]

    widths = [
        max(len(row[i]) for row in rows)
        for i in range(len(headers))
    ]

    def fmt(row):
        return (
            "| "
            + " | ".join(
                row[i].ljust(widths[i])
                for i in range(len(headers))
            )
            + " |"
        )

    separator = (
        "| "
        + " | ".join(
            "-" * widths[i]
            for i in range(len(headers))
        )
        + " |"
    )

    return "\n".join(
        [fmt(headers), separator]
        + [fmt(row) for row in rows[1:]]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден обязательный файл: {path}"
        )

    return pd.read_csv(path)


def _selected_features_section(
    base_dir: Path,
    dataset_title: str,
) -> list[str]:
    """
    Build the detailed selected-feature section for one dataset.

    F-score:
        ranked top-9 with F-score.

    Do:
        top-10 XGBoost candidates by gain +
        complete final GA subset.

    Calabrese:
        final 32 features with final/average/MI ranks.
    """
    lines = [
        f"## Выбранные признаки — {dataset_title}",
        "",
        (
            "Ниже приведены итоговые признаки, которые были переданы дальше "
            "как результат каждого метода feature selection."
        ),
        "",
        (
            "**Важно:** F-score и MI→RF-RFE формируют ранжирование признаков. "
            "Genetic Algorithm возвращает лучший найденный subset целиком, "
            "поэтому внутри финального GA-набора нет отдельного рейтинга "
            "«1-е, 2-е, 3-е место»."
        ),
        "",
    ]

    # ------------------------------------------------------------------
    # Le et al. — F-score
    # ------------------------------------------------------------------
    le = _read_csv(
        base_dir / "le" / "selected_features.csv"
    )

    le_columns = [
        column
        for column in ["rank", "feature", "f_score"]
        if column in le.columns
    ]

    lines.extend(
        [
            "### Le et al. — F-score",
            "",
            (
                f"Выбрано **{len(le)}** признаков. "
                "Они отсортированы по убыванию F-score."
            ),
            "",
            _markdown_table(le[le_columns]),
            "",
        ]
    )

    # ------------------------------------------------------------------
    # Do et al. — XGBoost gain → GA-RF
    # ------------------------------------------------------------------
    xgb_candidates = _read_csv(
        base_dir / "do" / "xgb_candidates.csv"
    )

    xgb_columns = [
        column
        for column in ["rank", "feature", "gain"]
        if column in xgb_candidates.columns
    ]

    do_selected = _read_csv(
        base_dir / "do" / "selected_features.csv"
    )

    lines.extend(
        [
            "### Do et al. — XGBoost gain → GA-RF",
            "",
            (
                "Сначала XGBoost формирует ranking кандидатов по `gain`. "
                "Первые 10 кандидатов первого этапа:"
            ),
            "",
            _markdown_table(
                xgb_candidates[xgb_columns].head(10)
            ),
            "",
            (
                f"После Genetic Algorithm в итоговый subset вошло "
                f"**{len(do_selected)}** признаков."
            ),
            "",
            (
                "GA оптимизирует комбинацию признаков целиком по внутренней "
                "5-fold CV accuracy Random Forest. Порядок строк ниже "
                "**не является рейтингом важности**."
            ),
            "",
            "<details>",
            (
                f"<summary>Показать итоговый GA subset "
                f"({len(do_selected)} признаков)</summary>"
            ),
            "",
            _markdown_table(do_selected[["feature"]]),
            "",
            "</details>",
            "",
        ]
    )

    # ------------------------------------------------------------------
    # Calabrese-inspired — MI → RF-RFE
    # ------------------------------------------------------------------
    cal = _read_csv(
        base_dir / "calabrese" / "selected_features.csv"
    )

    cal_columns = [
        column
        for column in [
            "final_rank",
            "feature",
            "average_rank",
            "rank_std",
            "mi_rank",
            "mi_score",
        ]
        if column in cal.columns
    ]

    lines.extend(
        [
            "### Calabrese-inspired — Mutual Information → RF-RFE",
            "",
            (
                f"Выбрано **{len(cal)}** признаков. "
                "Основной итоговый порядок задаётся `final_rank`; "
                "он строится по среднему RFE-rank между folds, "
                "а MI-rank используется как дополнительный критерий "
                "при равенстве."
            ),
            "",
            _markdown_table(cal[cal_columns]),
            "",
        ]
    )

    return lines


def main():
    required = [
        REFERENCE / "summary.csv",
        REFERENCE / "le" / "selected_features.csv",
        REFERENCE / "do" / "xgb_candidates.csv",
        REFERENCE / "do" / "selected_features.csv",
        REFERENCE / "calabrese" / "selected_features.csv",

        UPENN / "summary.csv",
        UPENN / "metadata.json",
        UPENN / "le" / "selected_features.csv",
        UPENN / "do" / "xgb_candidates.csv",
        UPENN / "do" / "selected_features.csv",
        UPENN / "calabrese" / "selected_features.csv",

        UPENN / "comparison" / "pairwise_overlap.csv",
        UPENN / "comparison" / "feature_composition.csv",
        UPENN / "comparison" / "predictive_comparison.csv",
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Для генерации полного results_summary.md "
            "не хватает файлов:\n"
            + "\n".join(missing)
            + "\n\nСначала запустите полный pipeline "
              "или нужные отдельные scripts."
        )

    # ------------------------------------------------------------------
    # Main summary tables
    # ------------------------------------------------------------------
    reference = pd.read_csv(
        REFERENCE / "summary.csv"
    )

    # The current hand-edited results_summary does not show internal_score.
    if "internal_score" in reference.columns:
        reference = reference.drop(
            columns=["internal_score"]
        )

    # Keep the wording already used in the current results_summary.
    if "note" in reference.columns:
        note_map = {
            "le_fscore": (
                "Le top-9 F-score reproduction; "
                "exact published top-9 match"
            ),
            "do_xgb_ga": (
                "XGBoost gain > 0 followed by GA-RF"
            ),
            "calabrese_mi_rfe": (
                "Calabrese-inspired method sanity check on shared "
                "TCGA table; original UCSF cohort not bundled"
            ),
        }

        reference["note"] = reference.apply(
            lambda row: note_map.get(
                row["method"],
                row["note"],
            ),
            axis=1,
        )

    upenn = pd.read_csv(
        UPENN / "summary.csv"
    )

    # Same appearance as the current edited results_summary.
    if "internal_score" in upenn.columns:
        upenn = upenn.drop(
            columns=["internal_score"]
        )

    pairwise = pd.read_csv(
        UPENN
        / "comparison"
        / "pairwise_overlap.csv"
    )

    composition = pd.read_csv(
        UPENN
        / "comparison"
        / "feature_composition.csv"
    )

    predictive = pd.read_csv(
        UPENN
        / "comparison"
        / "predictive_comparison.csv"
    )

    with open(
        UPENN / "metadata.json",
        encoding="utf-8",
    ) as file:
        upenn_meta = json.load(file)

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    lines = [
        "# MGMT radiomics feature-selection — сводка результатов",
        "",
        "## Цель исследования",
        "",
        (
            "Сравнить три подхода к обработке и отбору радиомических "
            "признаков: F-score по Le, XGBoost→GA-RF по Do и "
            "Calabrese-inspired MI→RF-RFE. Сначала методы проверяются "
            "на небольшом reference dataset, после чего применяются "
            "к одному общему набору structural radiomics UPenn-GBM."
        ),
        "",
        "",
        "## Результаты на reference dataset",
        "",
        _markdown_table(reference),
        "",
        "",
    ]

    # Exact selected features on reference.
    lines.extend(
        _selected_features_section(
            REFERENCE,
            "reference dataset",
        )
    )

    # UPenn dataset summary.
    lines.extend(
        [
            "## UPenn dataset",
            "",
            f"- Пациентов: {upenn_meta.get('patients', 'n/a')}",
            f"- Methylated: {upenn_meta.get('methylated', 'n/a')}",
            f"- Unmethylated: {upenn_meta.get('unmethylated', 'n/a')}",
            (
                "- Очищенных radiomics-признаков: "
                f"{upenn_meta.get('features', 'n/a')}"
            ),
            "",
            "## Результаты feature selection на UPenn",
            "",
            _markdown_table(upenn),
            "",
            "",
        ]
    )

    # Exact selected features on UPenn.
    lines.extend(
        _selected_features_section(
            UPENN,
            "UPenn-GBM",
        )
    )

    # Pairwise overlap.
    lines.extend(
        [
            "## Попарное пересечение выбранных признаков на UPenn",
            "",
            _markdown_table(pairwise),
            "",
            "## Состав выбранных признаков на UPenn",
            "",
        ]
    )

    dimension_titles = {
        "family": "Семейства признаков",
        "modality": "MRI-модальности",
        "region": "Области опухоли",
    }

    for dimension in [
        "family",
        "modality",
        "region",
    ]:
        part = composition[
            composition["dimension"] == dimension
        ].copy()

        lines.extend(
            [
                f"### {dimension_titles[dimension]}",
                "",
                _markdown_table(part),
                "",
            ]
        )

    # Predictive comparison.
    lines.extend(
        [
            "## Дополнительное сравнение по ROC-AUC на UPenn",
            "",
            _markdown_table(predictive),
            "",
            (
                "ROC-AUC в этой таблице используется для внутреннего "
                "сравнения готовых feature subsets. Это не независимая "
                "оценка generalization, поскольку feature selection был "
                "выполнен на полном UPenn dataset до этой CV."
            ),
            "",
        ]
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Generated: {OUT}")


if __name__ == "__main__":
    main()
