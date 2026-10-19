# Preprocessing UPenn-GBM перед feature selection

Основной код находится в:

```text
scripts/build_upenn.py
src/mgmt_features/preprocessing.py
```

## Два выхода для разных целей

`scripts/build_upenn.py` сохраняет одну и ту же когорту после объединения
12 блоков в два локальных файла (оба игнорируются Git):

- `data/processed/upenn/upenn_structural_raw.csv` — 256 пациентов,
  1728 признаков **до** удаления констант, дубликатов и заполнения пропусков;
  вход для fold-safe оценки.
- `data/processed/upenn/upenn_clean.csv` — прежняя матрица 256 × 1480;
  используется для воспроизведения уже выполненного exploratory-сравнения.

В `scripts/run_upenn_nested_cv.py` cleaner находится внутри `Pipeline`:
на каждом внешнем и внутреннем обучающем фолде он определяет константы,
точные дубликаты и train-медианы. На соответствующей validation-части
применяются только зафиксированные правила. Существующие `run_upenn.py`
и `evaluate_upenn_selected.py` сохраняют прежний exploratory-протокол.

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

Разделы 6–10 ниже фиксируют очистку **всей** матрицы для раннего
ознакомительного анализа и объясняют происхождение числа 1480. В основной
nested CV те же правила очистки обучаются заново **внутри каждого обучающего
фолда** на исходных 1728 признаках; число оставшихся признаков может
различаться между фолдами.

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

# 10. Полностью очищенная матрица для раннего анализа

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

Показатели очистки всей когорты:

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
| **Признаков после очистки всей когорты** | **1480** |

Сохраняется:

```text
data/processed/upenn/upenn_clean.csv
```

Отдельная сводка preprocessing:

```text
data/processed/upenn/summary.json
```

---

# 11. Два входа для разных протоколов оценки

Оба файла содержат 256 пациентов, метаданные `SubjectID`, `MGMT`, `target`
и радиомические признаки. Эти три служебные колонки исключаются из матрицы
признаков `X`; бинарная метка `target` используется как `y`.

| Протокол | Входной файл | Что происходит с признаками |
|---|---|---|
| Основная nested CV | `upenn_structural_raw.csv`: 1728 исходных признаков | На каждой обучающей части внешнего и внутреннего фолда заново удаляются константы и точные дубликаты, вычисляются медианы; затем выполняется отбор. Проверочная часть не влияет на эти шаги. |
| Ранний ознакомительный анализ | `upenn_clean.csv`: 1480 признаков | Очистка и отбор выполнялись на всех 256 пациентах до downstream CV; эта оценка подвержена утечке информации из проверочных фолдов. |

В основном сравнении все методы получают **одни и те же внешние фолды**
исходной матрицы 256 × 1728. После обучения очистки на конкретном фолде
число оставшихся признаков не обязано равняться 1480. Затем F-score
выбирает 9 признаков, MI → RF-RFE — 32, а GA-RF ищет подмножество среди
кандидатов XGBoost; его размер также меняется от фолда к фолду.

---

# 12. Как читать ранние и основные результаты

В раннем ознакомительном прогоне на заранее очищенной матрице
`256 × 1480` были получены такие оценки downstream Logistic Regression:

| Метод | Выбрано признаков | Ранний ROC-AUC |
|---|---:|---:|
| XGBoost + GA-RF | 136 | 0.6062 |
| F-score | 9 | 0.5968 |
| MI + RF-RFE | 32 | 0.5835 |

Здесь признаки выбирались с использованием **всех 256 пациентов до CV**.
Эти числа сохранены для истории проекта и не считаются оценкой качества
на новых пациентах.

Основная оценка запускается через `scripts/run_upenn_nested_cv.py` на
`upenn_structural_raw.csv`. Очистка и отбор обучаются внутри каждого
обучающего фолда; получены следующие AUROC по объединённым внешним
out-of-fold (OOF) прогнозам для заранее фиксированного первого повтора:

| Метод | Основной OOF AUROC |
|---|---:|
| F-score | 0.423 |
| MI → RF-RFE | 0.462 |
| XGBoost → GA-RF | 0.464 |
| Все признаки после очистки внутри фолда, L2 Logistic Regression | 0.434 |

Это разные протоколы: изменение ROC-AUC между таблицами нельзя приписать
только выбору метода или «ухудшению» модели. Описание внешних фолдов,
доверительных интервалов, случайных бейзлайнов и пяти сидов GA приведено
в [README](../README.md). Независимого тестового набора в этом сравнении нет.
