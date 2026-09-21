# MGMT radiomics feature-selection — generated results summary

## Study goal

Compare three article-derived radiomics feature-processing approaches: Le F-score, Do XGBoost→GA-RF, and Calabrese-inspired MI→RF-RFE. Methods are first checked on the small reference table and then applied to one common UPenn-GBM structural-radiomics dataset.

## Reference-data results

| method           | input_features | stage1_features | selected_features | internal_score     | note                                                                                          |
| ---------------- | -------------- | --------------- | ----------------- | ------------------ | --------------------------------------------------------------------------------------------- |
| le_fscore        | 704            | 704             | 9                 | nan                | Le top-9 F-score reproduction; exact published top-9 match                                    |
| do_xgb_ga        | 704            | 38              | 21                | 0.9254545454545454 | XGBoost gain > 0 followed by GA-RF                                                            |
| calabrese_mi_rfe | 704            | 704             | 32                | nan                | Calabrese-inspired method sanity check on shared TCGA table; original UCSF cohort not bundled |

**Interpretation note.** Le and Do are reproduced/checked on the shared 53-patient TCGA table. The Calabrese pipeline is a method sanity check on that table because the original UCSF feature matrix is not bundled here.

## UPenn dataset

- Patients: 256
- Methylated: 108
- Unmethylated: 148
- Clean radiomics features: 1480

## UPenn feature-selection results

| method           | input_features | stage1_features | selected_features | internal_score     |
| ---------------- | -------------- | --------------- | ----------------- | ------------------ |
| le_fscore        | 1480           | 1480            | 9                 | nan                |
| do_xgb_ga        | 1480           | 285             | 136               | 0.6290346907993967 |
| calabrese_mi_rfe | 1480           | 1024            | 32                | nan                |

The Do `internal_score` is the GA Random-Forest cross-validation fitness used by the selector; it is not an external generalization metric.

## Pairwise overlap on UPenn

| method_a  | method_b         | intersection | union | jaccard            |
| --------- | ---------------- | ------------ | ----- | ------------------ |
| le_fscore | do_xgb_ga        | 1            | 144   | 0.0069444444444444 |
| le_fscore | calabrese_mi_rfe | 4            | 37    | 0.1081081081081081 |
| do_xgb_ga | calabrese_mi_rfe | 12           | 156   | 0.0769230769230769 |

## Selected-feature composition on UPenn

### Family

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

### Modality

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

### Region

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

## What to discuss in the report

1. How aggressively each method reduces the feature space.
2. Which feature families, MRI modalities and tumor regions dominate each selected subset.
3. How much the three methods overlap on the same UPenn cohort.
4. Differences between the small reference dataset and the larger UPenn dataset.
5. Method limitations: small original cohorts, stochastic GA behavior, and the fact that feature selection here is the primary object of study rather than a fully validated clinical classifier.
