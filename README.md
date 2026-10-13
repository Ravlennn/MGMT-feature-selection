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
│   ├── nested_cv.py
│   ├── nested_statistics.py
│   ├── selection_stability.py
│   ├── pipelines.py
│   ├── preprocessing.py
│   └── selectors/
│       ├── fscore.py
│       ├── xgb_importance.py
│       ├── genetic_rf.py
│       ├── mi_rf_rfe.py
│       └── cv_adapters.py
├── scripts/
│   ├── build_upenn.py
│   ├── run_reference.py
│   ├── run_upenn.py
│   ├── evaluate_upenn_selected.py
│   ├── run_upenn_nested_cv.py
│   ├── run_upenn_selection_stability.py
│   ├── build_report.py
│   └── run_all.py
├── results/
│   ├── reference/
│   └── upenn/
└── docs/
    ├── SOURCES.md
    ├── SELECTION_STABILITY.md
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
docs/UPENN_PREPROCESSING.md
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
data/processed/upenn/upenn_structural_raw.csv
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

Результаты раннего exploratory-прогона (отбор признаков на всей когорте):

| Метод | Признаков | ROC-AUC | Balanced accuracy | Accuracy |
|---|---:|---:|---:|---:|
| XGBoost + GA-RF | 136 | 0.6062 | 0.5651 | 0.5781 |
| F-score | 9 | 0.5968 | 0.5513 | 0.5781 |
| MI + RF-RFE | 32 | 0.5835 | 0.5639 | 0.5898 |

### Ограничение этой оценки

Feature selection выполняется **до** данной downstream cross-validation на полном UPenn dataset.

То есть схема этого раннего прогона:

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

Поэтому эти ранние ROC-AUC являются **внутренней сравнительной метрикой готовых feature subsets**, а не независимой оценкой generalization или внешней clinical validation. Результаты корректной fold-safe оценки приведены ниже.

---

## Fold-safe оценка трёх методов и бейзлайнов

Отдельный эксперимент сравнивает Le/F-score, Do/XGBoost→GA-RF и
Calabrese-inspired MI→RF-RFE. Бейзлайны: `all_features` с L2 Logistic
Regression без отбора, `random_9`, `random_32` и `random_do_k`. Последний
получает фактическое K признаков, выбранных Do на том же внешнем обучающем
фолде. Для каждого random-бейзлайна по умолчанию выполняется 50 повторов;
`random_do_k` по очереди сопоставляется с пятью GA seeds.

Все ветки используют одинаковые внешние 5 фолдов. В каждом outer train
внутренний 5-fold `GridSearchCV` выбирает `C` из
`[0.001, 0.01, 0.1, 1, 10, 100]` по AUROC; GA запускается с пятью заранее
фиксированными seeds `11, 22, 33, 44, 55`. Выбор лучшего GA seed по внешнему
AUROC не выполняется.

`FoldSafeRadiomicsCleaner`, отбор признаков и `StandardScaler` входят в
`Pipeline`: они обучаются на train-части каждого внутреннего фолда, затем
заново на полном outer train. Оставшийся outer validation используется только
для прогноза. Для всех признаков применяется тот же L2-классификатор и та же
сетка `C`. Кэш Pipeline не пересчитывает дорогой selector для каждого нового
значения `C` на одном и том же обучающем фолде.

Для nested CV предусмотрены два заранее фиксированных профиля селекторов:

| Профиль | Do GA-RF | Calabrese-inspired RF-RFE |
|---|---|---|
| `practical` (по умолчанию) | population 12, generations 5, RF 50, internal CV 3 | MI 1024, top-32, RF 100, step 64, internal CV 3 |
| `legacy` | UPenn GA population 50, generations 20, RF 100, internal CV 5 | MI 1024, top-32, RF 1000, step 16, internal CV 5 |

`practical` сохраняет этапы алгоритмов, но уменьшает вычислительные параметры;
его результаты нельзя выдавать за запуск прежних настроек. `legacy` берёт
настройки из текущего UPenn-кода проекта, а не гарантирует точного
воспроизведения гиперпараметров статей; он может работать очень долго даже
с кэшем. `--quick` дополнительно уменьшает все
тяжёлые параметры и предназначен только для проверки кода.

```powershell
python scripts/build_upenn.py
python scripts/run_upenn_nested_cv.py --quick
python scripts/run_upenn_nested_cv.py
```

Можно запускать методы отдельно, а затем повторить команду без `--methods`,
чтобы использовать готовые checkpoints и собрать общую таблицу:

```powershell
python scripts/run_upenn_nested_cv.py --methods le_fscore all_features random_9 random_32
python scripts/run_upenn_nested_cv.py --methods calabrese_mi_rfe
python scripts/run_upenn_nested_cv.py --methods do_xgb_ga random_do_k
python scripts/run_upenn_nested_cv.py
```

Если GA пропускается из-за вычислительной стоимости, можно получить
**отдельный сокращённый отчёт** по Le и Calabrese без повторного расчёта уже
сохранённых фолдов:

```powershell
python scripts/run_upenn_nested_cv.py --methods random_9 random_32
python scripts/run_upenn_nested_cv.py --skip-ga
```

По умолчанию выполняется 50 повторов каждого случайного бейзлайна. Это
выбранная настройка эксперимента, а не требование руководителя. Если расчёт
уже начат и достаточно 20 повторов, остановите его после завершённого фолда
и подготовьте отдельный каталог с готовыми checkpoints:

```powershell
python scripts/prepare_upenn_nested_cv_repeats.py --random-repeats 20
python scripts/run_upenn_nested_cv.py --skip-ga --random-repeats 20 --output results/upenn/nested_cv/repeats_20
```

Первая команда копирует только завершённые фолды Le, Calabrese,
`all_features`, `random_9` и `random_32`, для каждого случайного метода —
только повторы 0–19. Выходной каталог должен быть новым. Вторая команда
досчитывает недостающие фолды и создаёт итоговые CSV. Старый каталог
`full/` не меняется; повторный запуск второй команды использует checkpoints.

`--skip-ga` использует checkpoints из указанного каталога;
при `--output` можно выбрать отдельный каталог. Сокращённый отчёт
содержит пять методов и пять парных сравнений. Он сохранён как промежуточный
вариант; **полный прогон с пятью сидами GA и `random_do_k` уже выполнен**
в `results/upenn/nested_cv/repeats_20/`. Результаты `--quick` не являются
частью итоговой оценки.

Команды итогового сбора не пересчитывают завершённые фолды. Они
сохраняют `summary.csv`, `fold_metrics.csv`, `oof_predictions.csv`,
`selected_features_by_fold.csv`, `method_inference.csv` и
`paired_comparisons.csv` в указанном `--output` (по умолчанию
`results/upenn/nested_cv/full/`). После
прерывания завершённые фолды читаются из `checkpoints/`; при изменении
исходного CSV, кода отбора или настроек требуется другая выходная папка.

Профиль `legacy` запускается в отдельную папку, например через
`--selector-profile legacy --output results/upenn/nested_cv/legacy`.
Можно увеличить число bootstrap-повторов статистического отчёта через
`--bootstrap-samples 5000`, не меняя эксперимент с моделями.

`summary.csv` содержит AUROC каждого повтора: среднее ± SD по пяти внешним
фолдам и AUROC объединённых OOF-предсказаний. В `method_inference.csv`
основная оценка для каждого метода заранее зафиксирована на `repeat=0`:
AUROC и 95% percentile CI по 2000 стратифицированным bootstrap-выборкам
пациентов. Там же показаны среднее, SD, минимум и максимум AUROC **между
повторами**: пять запусков GA и выбранное число случайных наборов признаков
(20 в итоговом эксперименте) не считаются
дополнительными пациентами. Для методов без повторов SD оставляется пустым.

`paired_comparisons.csv` в полном протоколе содержит девять сравнений: каждый
селектор против случайного K и `all_features`, затем три попарных сравнения
селекторов. `delta_auc_a_minus_b > 0` означает преимущество `method_a`;
95% CI разницы получен из **тех же** bootstrap-выборок пациентов.
`delong_p_two_sided` — парный тест DeLong для AUROC по OOF-прогнозам;
`delong_p_holm` корректирует p-values методом Holm по числу сравнений,
указанному в `n_holm_comparisons` (девять или пять при `--skip-ga`).
В сравнениях используются одинаковые пациенты/внешние фолды и только
заранее выбранный `repeat=0` (для GA — seed 11, для `random_do_k` —
соответствующий GA seed). Случайные повторы не усредняются в ансамбль.

Интервалы описывают разброс при переотборе **этих пациентов с уже
полученными OOF-прогнозами**, а не повторном обучении всей схемы на новой
когорте. DeLong на OOF-прогнозах разных обученных моделей трактуется как
приближённый, исследовательский тест; основной акцент — AUROC, парная
разница и её пациентский bootstrap CI. Никакого внешнего независимого
тестового набора в этом эксперименте нет. `--quick` создаёт только
проверочный отчёт без научной интерпретации. Прежние AUROC
0.606/0.597/0.583 из exploratory-сравнения не заменяются и не должны
сопоставляться с этой новой оценкой как результаты одного протокола.

### Итоги полного fold-safe прогона на UPenn

Для сравнения использована одна когорта из 256 пациентов, исходные 1728
структурных радиомических признаков и одинаковая Logistic Regression с
L2-регуляризацией. Очистка, отбор, масштабирование и выбор `C` выполняются
только на обучающих пациентах каждого фолда. Числа ниже получены из полного
прогона `results/upenn/nested_cv/repeats_20/`; сами результаты хранятся
локально и игнорируются Git.

| Метод | Признаков в среднем | OOF AUROC, повтор 0 | 95% ДИ | Средний OOF AUROC по повторам |
|---|---:|---:|---:|---:|
| Le F-score | 9 | 0.423 | 0.355–0.490 | — |
| Calabrese-inspired MI → RF-RFE | 32 | 0.462 | 0.391–0.531 | — |
| Do XGBoost → GA-RF | 116.2 | 0.464 | 0.394–0.534 | 0.491 ± 0.019 (5 сидов) |
| Все признаки, без отбора | 1479.2 | 0.434 | 0.367–0.506 | — |
| Случайные 9 признаков | 9 | 0.465 | 0.394–0.538 | 0.459 ± 0.036 (20 повторов) |
| Случайные 32 признака | 32 | 0.473 | 0.405–0.544 | 0.456 ± 0.029 (20 повторов) |
| Случайное число признаков как у GA | 116.2 | 0.479 | 0.405–0.547 | 0.459 ± 0.046 (20 повторов) |

**Интерпретация качества.** `repeat=0` зафиксирован для основных парных
сравнений; среднее ± SD справа описывает разброс по пяти сидам GA или по
20 случайным запускам и не заменяет оценку `repeat=0`. Лучший отдельный
сид GA нельзя выбирать задним числом как основной результат. В варианте
«без отбора» после очистки на обучающих фолдах остаётся в среднем 1479.2
признака, поэтому это не строка с 1728 необработанными столбцами.
Доверительные интервалы условны относительно уже полученных OOF-прогнозов;
отдельного независимого test set здесь нет.

Каждый метод отбора также сравнивался на тех же пациентах с соответствующим
случайным K и с моделью без отбора. Ниже положительная разница означает
больший AUROC метода отбора; интервалы рассчитаны для парных разниц.

| Метод | Δ AUROC к случайному K (95% ДИ) | Δ AUROC к модели без отбора (95% ДИ) |
|---|---:|---:|
| F-score | −0.042 [−0.128; 0.046] | −0.012 [−0.095; 0.069] |
| MI → RF-RFE | −0.010 [−0.095; 0.076] | 0.028 [−0.055; 0.112] |
| XGBoost → GA-RF | −0.015 [−0.113; 0.086] | 0.029 [−0.049; 0.106] |

**Интерпретация сравнений.** Все показанные интервалы включают ноль;
во всех девяти предусмотренных парных сравнениях p-value DeLong после
поправки Holm равен 1.0. Убедительного преимущества метода отбора над
случайным набором того же размера или моделью без отбора не обнаружено.
Это не доказывает эквивалентность методов: интервалы допускают как
положительные, так и отрицательные различия.

### Стабильность выбранных признаков

Каждый селектор был заново обучен на 20 стратифицированных бутстрэп-выборках
пациентов с возвращением. На каждой выборке очистка и отбор выполнялись
заново; сид самого селектора не менялся. Jaccard вычислен для всех 190 пар
наборов каждого метода. Это отдельный эксперимент, не бутстрэп OOF-прогнозов
для доверительных интервалов AUROC.

| Метод | Признаков за бутстрэп | Средний Jaccard | Медиана Jaccard | Разных признаков за 20 выборок |
|---|---:|---:|---:|---:|
| F-score | 9 | 0.027 | 0.000 | 132 |
| MI → RF-RFE | 32 | 0.047 | 0.049 | 370 |
| XGBoost → GA-RF | 98.9 в среднем | 0.070 | 0.068 | 865 |

**Интерпретация стабильности.** У F-score медианный Jaccard равен нулю:
не менее половины пар запусков не имеют общих признаков. Для MI → RF-RFE
и GA пересечение также невелико. GA выбирает значительно больше признаков,
поэтому его более высокий Jaccard сам по себе не означает более устойчивый
биомаркерный набор. 190 пар не являются независимыми наблюдениями.
Высокая внутренняя accuracy GA на бутстрэпах (примерно 0.8–0.9) не
оценивает качество на новых пациентах: копии одного пациента могут попасть
в разные внутренние фолды селектора. Для предсказательной оценки используется
внешняя CV из таблицы выше.

**Общий вывод:** на UPenn не обнаружено убедительной пользы трёх методов
отбора для AUROC, а выбранные ими признаки чувствительны к составу пациентов.
Сопоставление конкретных признаков reference ↔ UPenn требует проверенного
словаря соответствий между схемами извлечения и пока не выполнено. Протокол
стабильности, команды запуска и ограничения описаны в
[`docs/SELECTION_STABILITY.md`](docs/SELECTION_STABILITY.md).

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
