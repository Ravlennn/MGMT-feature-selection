# MGMT radiomics feature-selection — сводка результатов

## Цель исследования

Сравнить три подхода к обработке и отбору радиомических признаков, взятых из статей: F-score по Le, XGBoost→GA-RF по Do и Calabrese-inspired MI→RF-RFE. Сначала методы проверяются на небольшой reference-таблице, после чего применяются к одному общему набору structural radiomics UPenn-GBM.

## Результаты на reference dataset

| method           | input_features | stage1_features | selected_features | note                                                                                          |
| ---------------- | -------------- | --------------- | ----------------- | --------------------------------------------------------------------------------------------- |
| le_fscore        | 704            | 704             | 9                 | Воспроизведение Le top-9 по F-score; опубликованный top-9 совпал точно                        |
| do_xgb_ga        | 704            | 38              | 21                | XGBoost gain > 0, после чего GA-RF                                                            |
| calabrese_mi_rfe | 704            | 704             | 32                | Проверка логики метода Calabrese на общей TCGA-таблице; оригинальный UCSF cohort не включён   |


## UPenn dataset

- Пациентов: 256
- Methylated: 108
- Unmethylated: 148
- Очищенных radiomics-признаков: 1480

## Результаты feature selection на UPenn

| method           | input_features | stage1_features | selected_features |
| ---------------- | -------------- | --------------- | ----------------- |
| le_fscore        | 1480           | 1480            | 9                 |
| do_xgb_ga        | 1480           | 285             | 136               |
| calabrese_mi_rfe | 1480           | 1024            | 32                |


## Попарное пересечение выбранных признаков на UPenn

| method_a  | method_b         | intersection | union | jaccard            |
| --------- | ---------------- | ------------ | ----- | ------------------ |
| le_fscore | do_xgb_ga        | 1            | 144   | 0.0069444444444444 |
| le_fscore | calabrese_mi_rfe | 4            | 37    | 0.1081081081081081 |
| do_xgb_ga | calabrese_mi_rfe | 12           | 156   | 0.0769230769230769 |

## Состав выбранных признаков на UPenn

### Семейства признаков

| method           | dimension | value       | count | fraction           |
| ---------------- | --------- | ----------- | ----- | ------------------ |
| le_fscore        | family    | HISTOGRAM   | 4     | 0.4444444444444444 |
| le_fscore        | family    | INTENSITY   | 2     | 0.2222222222222222 |
| le_fscore        | family    | GLSZM       | 2     | 0.2222222222222222 |
| le_fscore        | family    | MORPHOLOGIC | 1     | 0.1111111111111111 |
| do_xgb_ga        | family    | HISTOGRAM   | 55    | 0.4044117647058823 |
| do_xgb_ga        | family    | INTENSITY   | 30    | 0.2205882352941176 |
| do_xgb_ga        | family    | GLSZM       | 27    | 0.1985294117647058 |
| do_xgb_ga        | family    | GLCM        | 10    | 0.0735294117647058 |
| do_xgb_ga        | family    | GLRLM       | 7     | 0.0514705882352941 |
| do_xgb_ga        | family    | MORPHOLOGIC | 4     | 0.0294117647058823 |
| do_xgb_ga        | family    | NGTDM       | 3     | 0.0220588235294117 |
| calabrese_mi_rfe | family    | HISTOGRAM   | 16    | 0.5                |
| calabrese_mi_rfe | family    | INTENSITY   | 7     | 0.21875            |
| calabrese_mi_rfe | family    | MORPHOLOGIC | 4     | 0.125              |
| calabrese_mi_rfe | family    | GLSZM       | 3     | 0.09375            |
| calabrese_mi_rfe | family    | GLCM        | 1     | 0.03125            |
| calabrese_mi_rfe | family    | GLRLM       | 1     | 0.03125            |

### MRI-модальности

| method           | dimension | value        | count | fraction           |
| ---------------- | --------- | ------------ | ----- | ------------------ |
| le_fscore        | modality  | T1GD         | 3     | 0.3333333333333333 |
| le_fscore        | modality  | T2           | 3     | 0.3333333333333333 |
| le_fscore        | modality  | T1           | 2     | 0.2222222222222222 |
| le_fscore        | modality  | SHAPE_SHARED | 1     | 0.1111111111111111 |
| do_xgb_ga        | modality  | T2           | 39    | 0.2867647058823529 |
| do_xgb_ga        | modality  | T1GD         | 36    | 0.2647058823529412 |
| do_xgb_ga        | modality  | T1           | 30    | 0.2205882352941176 |
| do_xgb_ga        | modality  | FLAIR        | 27    | 0.1985294117647058 |
| do_xgb_ga        | modality  | SHAPE_SHARED | 4     | 0.0294117647058823 |
| calabrese_mi_rfe | modality  | T2           | 14    | 0.4375             |
| calabrese_mi_rfe | modality  | T1GD         | 7     | 0.21875            |
| calabrese_mi_rfe | modality  | T1           | 5     | 0.15625            |
| calabrese_mi_rfe | modality  | SHAPE_SHARED | 4     | 0.125              |
| calabrese_mi_rfe | modality  | FLAIR        | 2     | 0.0625             |

### Области опухоли

| method           | dimension | value | count | fraction           |
| ---------------- | --------- | ----- | ----- | ------------------ |
| le_fscore        | region    | ED    | 5     | 0.5555555555555556 |
| le_fscore        | region    | NC    | 3     | 0.3333333333333333 |
| le_fscore        | region    | ET    | 1     | 0.1111111111111111 |
| do_xgb_ga        | region    | NC    | 49    | 0.3602941176470588 |
| do_xgb_ga        | region    | ED    | 47    | 0.3455882352941176 |
| do_xgb_ga        | region    | ET    | 40    | 0.2941176470588235 |
| calabrese_mi_rfe | region    | ET    | 12    | 0.375              |
| calabrese_mi_rfe | region    | NC    | 11    | 0.34375            |
| calabrese_mi_rfe | region    | ED    | 9     | 0.28125            |


