# Preprocessing UPenn-GBM перед feature selection

Основной код находится в:

```text
scripts/build_upenn.py
src/mgmt_features/preprocessing.py
```

---

# 1. Исходная clinical table

Используется файл:

```text
data/raw/upenn/clinical/UPENN-GBM_clinical_info_v2.1.csv
```

В нём находится:

```text
671 пациент
```

Распределение `MGMT`:

| MGMT status | Пациентов |
|---|---:|
| Not Available | 348 |
| Unmethylated | 170 |
| Methylated | 121 |
| Indeterminate | 32 |
| **Всего** | **671** |

Для feature selection нам нужен однозначный binary target.

Поэтому оставляются только:

```text
Unmethylated
Methylated
```

В коде:

```python
VALID_MGMT = {
    "Methylated": 1,
    "Unmethylated": 0,
}

clinical = clinical[
    clinical["MGMT"].isin(VALID_MGMT)
].copy()

clinical["target"] = (
    clinical["MGMT"]
    .map(VALID_MGMT)
    .astype(int)
)
```

После этого:

```text
671
↓ удалить Not Available и Indeterminate
291 пациента с известным MGMT
```

Распределение:

```text
170 Unmethylated → target = 0
121 Methylated   → target = 1
```

---

# 2. Нормализация SubjectID

Перед объединением таблиц ID пациентов приводятся к строковому формату и очищаются от пробелов:

```python
def _normalize_id(series):
    return series.astype(str).str.strip()
```

Это необходимо, чтобы одинаковые пациенты корректно совпадали между clinical и radiomics tables.

---

# 3. Какие radiomics используются

В основной pipeline включены только structural MRI:

```text
FLAIR
T1GD
T1
T2
```

Для каждой modality используются три области опухоли:

```text
ED — edema
ET — enhancing tumor
NC — necrotic/core region
```

Таким образом:

```text
4 modalities × 3 regions = 12 radiomics blocks
```

Полный список:

```text
FLAIR_ED
FLAIR_ET
FLAIR_NC

T1GD_ED
T1GD_ET
T1GD_NC

T1_ED
T1_ET
T1_NC

T2_ED
T2_ET
T2_NC
```

Каждый block содержит:

```text
144 radiomics-признака
```

Поэтому потенциально:

```text
12 × 144 = 1728 признаков
```

DTI и DSC в основной pipeline не используются.

---

# 4. Объединение clinical и radiomics tables

В начале есть:

```text
291 пациент с известным MGMT
```

Каждый radiomics block последовательно объединяется с текущей таблицей через:

```python
merged = merged.merge(
    block_df,
    on="SubjectID",
    how="inner",
    validate="one_to_one",
)
```

`how="inner"` означает:

> пациент остаётся только в том случае, если он присутствует и в уже собранной таблице, и в новом radiomics block.

Поэтому число пациентов постепенно уменьшается.

Фактическая последовательность:

| Добавляемый block | До merge | После merge |
|---|---:|---:|
| исходные MGMT-labelled patients | 291 | 291 |
| FLAIR_ED | 291 | 262 |
| FLAIR_ET | 262 | 258 |
| FLAIR_NC | 258 | 256 |
| T1GD_ED | 256 | 256 |
| T1GD_ET | 256 | 256 |
| T1GD_NC | 256 | 256 |
| T1_ED | 256 | 256 |
| T1_ET | 256 | 256 |
| T1_NC | 256 | 256 |
| T2_ED | 256 | 256 |
| T2_ET | 256 | 256 |
| T2_NC | 256 | 256 |

То есть основная потеря пациентов происходит на первых трёх FLAIR blocks.

После объединения всех 12 таблиц:

```text
256 пациентов
```

Распределение MGMT в финальной группе:

```text
148 Unmethylated
108 Methylated
```

Именно эти 256 пациентов используются далее во всех трёх feature-selection методах.

---

# 5. Матрица до очистки признаков

После объединения:

```text
256 пациентов
1728 radiomics-признаков
```

Плюс три служебные колонки:

```text
SubjectID
MGMT
target
```

Metadata не участвуют в feature selection.

В feature matrix `X` попадают только 1728 radiomics columns.

---

# 6. Замена бесконечных значений

В начале preprocessing:

```python
X = X.copy().replace(
    [np.inf, -np.inf],
    np.nan,
)
```

Если где-либо встречаются `+inf` или `-inf`, они переводятся в стандартный `NaN`, после чего обрабатываются единообразно с пропущенными значениями.

---

# 7. Удаление константных признаков

Константный признак имеет одно и то же значение у всех пациентов и не может разделять MGMT-классы.

Код:

```python
constant_features = [
    column
    for column in X.columns
    if X[column].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_features)
```

Было удалено:

```text
84 constant features
```

После этого:

```text
1728 - 84 = 1644 признака
```

---

# 8. Удаление точных дубликатов

Некоторые radiomics columns полностью совпадают друг с другом для всех пациентов.

Например:

```text
Feature_A = [1.2, 3.4, 2.8, ...]
Feature_B = [1.2, 3.4, 2.8, ...]
```

Второй признак не добавляет новой информации.

Код:

```python
duplicate_mask = X.T.duplicated(
    keep="first"
)

duplicate_features = (
    duplicate_mask[duplicate_mask]
    .index.tolist()
)

X = X.drop(
    columns=duplicate_features
)
```

Зачем используется `X.T`:

```text
до транспонирования:
rows    = patients
columns = features

после X.T:
rows = features
```

После этого pandas может искать одинаковые feature vectors как дублирующиеся строки.

Удалено:

```text
164 exact duplicate features
```

Получаем:

```text
1644 - 164 = 1480 признаков
```

---

# 9. Обработка оставшихся пропусков

После удаления constants и duplicates осталось:

```text
4 missing values
в 4 признаках
```

Они заполняются медианой соответствующего признака:

```python
medians = X.median(axis=0)
X = X.fillna(medians)
```

Почему медиана:

- проста;
- не требует удаления пациента;
- менее чувствительна к выбросам, чем среднее значение.

После imputation код дополнительно проверяет:

```python
if X.isna().any().any():
    raise ValueError(
        "NaN values remain after UPenn cleaning."
    )
```

Поэтому итоговая feature matrix не содержит `NaN`.

Важно:

```text
imputation не меняет число признаков
```

Остаётся:

```text
1480
```

---

# 10. Финальный UPenn dataset

Итог preprocessing:

```text
Исходная clinical table:
671 пациент
        ↓
оставить известный MGMT:
291 пациент
        ↓
пересечение с 12 structural radiomics blocks:
256 пациентов
        ↓
1728 raw structural features
        ↓
-84 constants
        ↓
1644
        ↓
-164 exact duplicates
        ↓
1480
        ↓
median imputation 4 values
        ↓
1480 clean features
```

Финальный набор:

| Характеристика | Значение |
|---|---:|
| Пациентов | 256 |
| Unmethylated | 148 |
| Methylated | 108 |
| Structural MRI modalities | 4 |
| Tumor regions | 3 |
| Radiomics blocks | 12 |
| Исходных radiomics-признаков | 1728 |
| Удалено constants | 84 |
| Удалено exact duplicates | 164 |
| Imputed values | 4 |
| **Финальных признаков** | **1480** |

Сохраняется:

```text
data/processed/upenn/upenn_clean.csv
```

Отдельная сводка preprocessing:

```text
data/processed/upenn/summary.json
```

---

# 11. Что именно передаётся в feature selection

Файл `upenn_clean.csv` содержит:

```text
SubjectID
MGMT
target
+ 1480 radiomics features
```

При загрузке:

```python
UPENN_METADATA_COLUMNS = [
    "SubjectID",
    "MGMT",
    "target",
]
```

Эти три колонки исключаются из `X`.

Поэтому все три selectors получают абсолютно одну и ту же матрицу:

```text
X: 256 × 1480
y: 256 binary MGMT labels
```

Дальше pipeline расходится на три ветки:

```text
                         256 × 1480
                              │
             ┌────────────────┼────────────────┐
             │                │                │
          F-score          XGBoost             MI
             │                │                │
          Top-9             gain>0          Top-1024
                              │                │
                              GA              RF-RFE
                              │                │
                              ↓                ↓
             9              136               32
          features        features         features
```

Именно поэтому дальнейшее сравнение является сравнением методов feature selection на одном и том же подготовленном UPenn dataset.

---

# 12. Итоговые UPenn feature-selection results

На финальной матрице `256 × 1480` получены:

```text
F-score:
1480 → 9

XGBoost + GA-RF:
1480 → 285 XGBoost candidates → 136

Mutual Information + RF-RFE:
1480 → 1024 → 32
```

Дополнительное внутреннее downstream-сравнение через одну и ту же Logistic Regression:

| Метод | Features | ROC-AUC |
|---|---:|---:|
| XGBoost + GA-RF | 136 | 0.6062 |
| F-score | 9 | 0.5968 |
| MI + RF-RFE | 32 | 0.5835 |

Эти ROC-AUC используются только как сравнительная оценка выбранных feature subsets: feature selection выполнялся на полном UPenn dataset до downstream cross-validation.
