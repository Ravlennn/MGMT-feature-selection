# MGMT Radiomics Feature Selection

Cравнение трёх подходов к отбору радиомических признаков для анализа статуса метилирования промотора **MGMT**:

- **Le et al.** — ранжирование по F-score.
- **Do et al.** — XGBoost gain → Genetic Algorithm с Random Forest в качестве fitness-функции.
- **Calabrese-inspired** — Mutual Information → Random-Forest RFE.

Нынешняя цель — **сравнение подходов к feature selection**.

## Общая схема проекта

```text
reference / авторские данные
        ↓
проверка 3 методов feature selection
        ↓
сравнение с результатами статей

открытые данные UPenn-GBM radiomics
        ↓
общий preprocessing
        ↓
3 метода feature selection
        ↓
сравнение выбранных наборов признаков
        ↓
дополнительная оценка через Logistic Regression
        ↓
таблицы и материалы для отчёта
```

## Структура репозитория

```text
MGMT-feature-selection/
├── data/
│   ├── raw/
│   │   ├── do_2022/
│   │   └── upenn/
│   └── processed/
│       └── upenn/
├── src/mgmt_features/
│   ├── analysis.py
│   ├── config.py
│   ├── data.py
│   ├── pipelines.py
│   ├── preprocessing.py
│   └── selectors/
│       ├── fscore.py
│       ├── xgb_importance.py
│       ├── genetic_rf.py
│       └── mi_rf_rfe.py
├── scripts/
│   ├── build_upenn.py
│   ├── run_reference.py
│   ├── run_upenn.py
│   ├── evaluate_upenn_selected.py
│   ├── build_report.py
│   └── run_all.py
├── results/
│   ├── reference/
│   └── upenn/
└── docs/
    ├── SOURCES.md
    └── results_summary.md
```

## Установка

Из корня репозитория:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Поддерживается Python 3.11+.

Основные зависимости:

- NumPy
- pandas
- scikit-learn
- XGBoost

---

# Данные

## Reference dataset

Для проверки реализации методов используется датасет:

```text
data/raw/do_2022/Training dataset.csv
```

В ней:

- 53 пациента;
- 724 исходных radiomics-признака;
- 27 Unmethylated;
- 26 Methylated.

На этапе preprocessing удаляются все признаки, содержащие хотя бы один `NaN`:

```text
724 исходных признака
→ удалить 20 признаков с пропусками
→ 704 признака
```

После этого один и тот же датасет из 704 признаков передаётся всем трём методам.

## UPenn-GBM

Для основного сравнения используется официальный датасет UPenn-GBM.

Подробный preprocessing вынесен в отдельный файл:

```text
README_UPENN_PREPROCESSING.md
```

Кратко итоговая последовательность выглядит так:

```text
671 пациент в clinical table
        ↓
оставляем только известный MGMT
        ↓
291 пациент
        ↓
inner join 12 structural radiomics blocks
        ↓
256 пациентов
        ↓
1728 radiomics-признаков
        ↓
- 84 константных
- 164 точных дубликата
+ median imputation 4 пропущенных значений
        ↓
1480 clean radiomics-признаков
```

Финальная выборка:

```text
256 пациентов
148 Unmethylated
108 Methylated
1480 radiomics-признаков
```

Результат сохраняется в:

```text
data/processed/upenn/upenn_clean.csv
```

---

# Почему используются только structural MRI

Основной эксперимент использует только структурные MRI radiomics:

```text
FLAIR × ED / ET / NC
T1GD  × ED / ET / NC
T1    × ED / ET / NC
T2    × ED / ET / NC
```

Всего:

```text
4 modalities × 3 tumor regions = 12 radiomics blocks
```

---

# Построение UPenn dataset

Запуск:

```powershell
python scripts/build_upenn.py
```

Ожидаемый результат:

```text
671 clinical patients
→ 291 с известным MGMT
→ 256 с полным набором 12 structural radiomics blocks

1728 raw structural features
→ удалить 84 constants
→ удалить 164 exact duplicates
→ median-impute 4 residual missing values
= 1480 clean radiomics features
```

Создаются:

```text
data/processed/upenn/upenn_clean.csv
data/processed/upenn/summary.json
```

---

# Метод 1: Le et al. — F-score

Для каждого признака независимо рассчитывается F-score:

```text
различие средних между Methylated и Unmethylated
------------------------------------------------
       внутригрупповой разброс признака
```

Чем сильнее различаются две группы и чем меньше разброс внутри групп, тем выше F-score.

После расчёта все признаки сортируются по убыванию F-score и выбирается фиксированный top-9:

```text
reference:
704 → 9

UPenn:
1480 → 9
```

На reference dataset опубликованный top-9 Le et al. был воспроизведён точно.

---

# Метод 2: Do et al. — XGBoost gain → GA-RF

Метод состоит из двух этапов.

## Этап 1. XGBoost

XGBoost обучается на всех признаках, после чего рассчитывается `gain importance`.

Оставляются признаки:

```text
gain > 0
```

Результаты:

```text
reference:
704 → 38 XGBoost candidates

UPenn:
1480 → 285 XGBoost candidates
```

## Этап 2. Genetic Algorithm + Random Forest

Каждая возможная комбинация признаков кодируется бинарной хромосомой:

```text
1 0 1 1 0 ...
```

где:

- `1` — признак используется;
- `0` — признак исключён.

Fitness каждой хромосомы — средняя `accuracy` Random Forest в stratified 5-fold cross-validation.

GA использует:

- selection;
- crossover;
- mutation;
- elitism.

При одинаковой accuracy предпочтение отдаётся более компактному набору признаков.

Результаты:

```text
reference:
38 → 21
internal RF CV accuracy ≈ 0.9255

UPenn:
285 → 136
internal RF CV accuracy ≈ 0.6290
```

`internal_score` является fitness-функцией GA, а не независимой внешней оценкой качества модели.

---

# Метод 3: Calabrese-inspired — Mutual Information → RF-RFE

## Этап 1. Mutual Information

Для каждого признака рассчитывается Mutual Information с MGMT target.

Затем выбираются максимум 1024 лучших признака.

Поэтому:

```text
reference:
704 → 704
```

На reference признаков меньше 1024, поэтому MI только ранжирует их.

На UPenn:

```text
1480 → 1024
```

## Этап 2. Random Forest RFE

Далее выполняется Recursive Feature Elimination:

```text
обучить Random Forest
→ определить менее важные признаки
→ удалить 16 признаков
→ повторить
```

Процедура выполняется в 5 folds.

Для каждого признака вычисляется средний RFE-rank между folds, после чего выбираются 32 лучших признака.

Итог:

```text
reference:
704 → 704 → 32

UPenn:
1480 → 1024 → 32
```

Для Calabrese это method sanity check / адаптация метода, поскольку оригинальная UCSF feature matrix в проекте отсутствует.

---


# Как используется cross-validation


## 1. F-score

F-score **не использует cross-validation**.

---

## 2. XGBoost → Genetic Algorithm + Random Forest

На первом этапе XGBoost обучается на текущем датасете и оставляет признаки с:

```text
gain > 0
```

Cross-validation появляется внутри Genetic Algorithm.

Каждая хромосома GA задаёт отдельный subset признаков:

```text
1 0 1 1 0 ...
```

Для оценки этой хромосомы используется:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

`Stratified` означает, что в каждом fold стараются сохранить примерно то же соотношение `Methylated / Unmethylated`, что и во всём датасете.

Для каждого из 5 folds:

```text
4 folds → обучение Random Forest
1 fold  → validation

повторить 5 раз
```

Random Forest содержит:

```text
100 деревьев
```

На каждом fold считается `accuracy`, после чего берётся среднее:

```text
accuracy_1
accuracy_2
accuracy_3
accuracy_4
accuracy_5
     ↓
mean accuracy
     ↓
fitness хромосомы
```

Именно это значение GA использует, чтобы решить, какая комбинация признаков лучше.

Поэтому значения:

```text
reference: ≈ 0.9255
UPenn:     ≈ 0.6290
```

— это **internal 5-fold CV fitness Genetic Algorithm**, а не независимая test accuracy.

На UPenn при 256 пациентах в каждом таком разбиении получается примерно:

```text
204–205 пациентов → train
51–52 пациента    → validation
```

На reference dataset из 53 пациентов:

```text
примерно 42–43 → train
примерно 10–11 → validation
```

---

## 3. Mutual Information → RF-RFE

В методе Calabrese-inspired cross-validation используется по-другому.

Сначала Mutual Information рассчитывается на текущем датасете и формирует набор кандидатов:

```text
reference:
704 → 704

UPenn:
1480 → 1024
```

После этого создаётся:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

Для каждого fold RF-RFE обучается **только на train-части**:

```text
fold 1 train → RF-RFE ranking 1
fold 2 train → RF-RFE ranking 2
fold 3 train → RF-RFE ranking 3
fold 4 train → RF-RFE ranking 4
fold 5 train → RF-RFE ranking 5
```

Внутри RFE используется Random Forest:

```text
1000 деревьев
```

На каждой итерации удаляется:

```text
16 признаков
```

пока не останется 32 признака.


После пяти запусков для каждого признака вычисляется:

```text
average_rank = средний RFE-rank по 5 folds
rank_std     = разброс rank между folds
```

И затем признаки сортируются по `average_rank`. Если средний rank совпадает, дополнительным критерием служит `mi_rank`.

Итог:

```text
5 RFE rankings
      ↓
average rank
      ↓
final ranking
      ↓
top-32
```

---

## 4. Дополнительная оценка selected features по ROC-AUC

После того как каждый метод уже выбрал свой набор признаков, выполняется отдельное сравнение:

```text
F-score subset        → 9 признаков
XGBoost + GA-RF       → 136 признаков
MI + RF-RFE           → 32 признака
```

Для всех трёх используется **одинаковый downstream pipeline**:

```text
StandardScaler
→ Logistic Regression
→ Stratified 5-fold cross-validation
```

Разбиение:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

`StandardScaler` и `LogisticRegression` находятся внутри одного `Pipeline`, поэтому на каждом fold scaler обучается только на train-части и затем применяется к validation-части.

Через:

```python
cross_val_predict(
    ...,
    method="predict_proba",
)
```

каждый пациент получает out-of-fold вероятность от модели, которая не использовала этого пациента при обучении.

После объединения predictions со всех пяти folds считаются:

```text
ROC-AUC
Balanced Accuracy
Accuracy
```

Текущие результаты:

| Метод | Признаков | ROC-AUC | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|
| XGBoost + GA-RF | 136 | 0.6062 | 0.5651 | 0.5781 |
| F-score | 9 | 0.5968 | 0.5513 | 0.5781 |
| MI + RF-RFE | 32 | 0.5835 | 0.5639 | 0.5898 |

### Ограничение этой оценки

Feature selection выполняется **до** данной downstream cross-validation на полном UPenn dataset.

То есть схема сейчас:

```text
полный UPenn
     ↓
feature selection
     ↓
готовый subset
     ↓
5-fold CV Logistic Regression
```

а не строгая nested схема:

```text
outer train
     ↓
feature selection только внутри train
     ↓
model
     ↓
outer validation
```

Поэтому ROC-AUC в проекте используется как **внутренняя сравнительная метрика готовых feature subsets**, а не как независимая оценка generalization или внешняя clinical validation.

---

# Быстрый путь к основным результатам

Валидированные reference-результаты уже сохранены в `results/reference/`.

Поэтому можно запустить:

```powershell
python scripts/run_all.py --skip-reference
```

Команда:

1. заново создаст `upenn_clean.csv`;
2. применит Le, Do и Calabrese-inspired к UPenn;
3. сравнит выбранные признаки;
4. создаст `docs/results_summary.md`.

Do GA-RF и Calabrese RF-RFE являются наиболее вычислительно тяжёлыми этапами.

---

# Полное воспроизведение с нуля

Чтобы заново запустить reference и UPenn:

```powershell
python scripts/run_all.py
```

Для быстрого smoke-test:

```powershell
python scripts/run_all.py --quick
```

`--quick` использует уменьшенные GA/RF-RFE параметры только для проверки работоспособности кода.

**Результаты `--quick` нельзя использовать как финальные результаты эксперимента.**

---

# Запуск отдельных этапов

Построить UPenn dataset:

```powershell
python scripts/build_upenn.py
```

Перезапустить reference experiments:

```powershell
python scripts/run_reference.py
```

Запустить все три метода на UPenn:

```powershell
python scripts/run_upenn.py
```

Собрать Markdown-сводку результатов:

```powershell
python scripts/build_report.py
```

Отдельно сравнить выбранные UPenn feature subsets через единый downstream-классификатор:

```powershell
python scripts/evaluate_upenn_selected.py
```

---

# Reference results

Текущие валидированные результаты:

```text
Le:
704 → 9

Do:
704 → 38 XGB candidates → 21 GA features

Calabrese:
704 → 704 → 32
```

Для Do:

```text
GA internal 5-fold RF accuracy ≈ 0.9255
```

Для Le опубликованный top-9 был воспроизведён точно.

Calabrese pipeline на reference dataset является проверкой логики метода, а не точным воспроизведением оригинального UCSF cohort.

---

# UPenn feature-selection results

После полного запуска:

```text
Le:
1480 → 9

Do:
1480 → 285 → 136

Calabrese:
1480 → 1024 → 32
```

Результаты сохраняются в:

```text
results/upenn/
├── summary.csv
├── le/
├── do/
├── calabrese/
└── comparison/
    ├── pairwise_overlap.csv
    ├── feature_membership.csv
    ├── common_all_methods.csv
    ├── feature_composition.csv
    └── predictive_comparison.csv
```

`feature_composition.csv` позволяет сравнивать выбранные признаки по:

- семейству признака: `GLCM`, `GLRLM`, `GLSZM`, Histogram, Intensity, Morphologic и т. д.;
- MRI modality: `FLAIR`, `T1`, `T1GD`, `T2`;
- tumor region: `ED`, `ET`, `NC`.

Morphologic/volumetric признаки помечаются как `SHAPE_SHARED`, поскольку они определяются геометрией сегментации, а точные дубликаты между modalities удаляются во время preprocessing.

---

# Дополнительное сравнение по ROC-AUC

После feature selection выбранные наборы признаков можно сравнить на одном и том же downstream-классификаторе:

```text
StandardScaler
→ Logistic Regression
→ stratified 5-fold CV
```

Запуск:

```powershell
python scripts/evaluate_upenn_selected.py
```

Текущий результат:

| Метод | Признаков | ROC-AUC | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|
| XGBoost + GA-RF | 136 | 0.6062 | 0.5651 | 0.5781 |
| F-score | 9 | 0.5968 | 0.5513 | 0.5781 |
| MI + RF-RFE | 32 | 0.5835 | 0.5639 | 0.5898 |

Важно: feature selection выполнялся до этой downstream CV на полном UPenn dataset, поэтому ROC-AUC используется как **внутренняя сравнительная метрика выбранных feature subsets**, а не как независимая внешняя оценка generalization.

