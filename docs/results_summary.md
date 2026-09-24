# MGMT radiomics feature-selection — сводка результатов

## Цель исследования

Сравнить три подхода к обработке и отбору радиомических признаков: F-score по Le, XGBoost→GA-RF по Do и Calabrese-inspired MI→RF-RFE. Сначала методы проверяются на небольшом reference dataset, после чего применяются к одному общему набору structural radiomics UPenn-GBM.


## Результаты на reference dataset

| method           | input_features | stage1_features | selected_features | note                                                                                          |
| ---------------- | -------------- | --------------- | ----------------- | --------------------------------------------------------------------------------------------- |
| le_fscore        | 704            | 704             | 9                 | Le top-9 F-score reproduction; exact published top-9 match                                    |
| do_xgb_ga        | 704            | 38              | 21                | XGBoost gain > 0 followed by GA-RF                                                            |
| calabrese_mi_rfe | 704            | 704             | 32                | Calabrese-inspired method sanity check on shared TCGA table; original UCSF cohort not bundled |


## Выбранные признаки — reference dataset

Ниже приведены итоговые признаки, которые были переданы дальше как результат каждого метода feature selection.

**Важно:** F-score и MI→RF-RFE формируют ранжирование признаков. Genetic Algorithm возвращает лучший найденный subset целиком, поэтому внутри финального GA-набора нет отдельного рейтинга «1-е, 2-е, 3-е место».

### Le et al. — F-score

Выбрано **9** признаков. Они отсортированы по убыванию F-score.

| rank | feature                     | f_score  |
| ---- | --------------------------- | -------- |
| 1    | HISTO_ET_T2_Bin6            | 0.259136 |
| 2    | TEXTURE_GLRLM_ED_T2_GLV     | 0.199733 |
| 3    | TEXTURE_GLSZM_NET_FLAIR_ZP  | 0.183226 |
| 4    | TEXTURE_GLSZM_NET_FLAIR_SZE | 0.167387 |
| 5    | TEXTURE_GLSZM_NET_FLAIR_ZSN | 0.167230 |
| 6    | TEXTURE_GLSZM_NET_T1_ZSN    | 0.166896 |
| 7    | TEXTURE_GLSZM_NET_T1_SZE    | 0.161356 |
| 8    | HISTO_ED_T2_Bin5            | 0.157338 |
| 9    | TEXTURE_GLSZM_NET_T1_ZP     | 0.152634 |

### Do et al. — XGBoost gain → GA-RF

Сначала XGBoost формирует ranking кандидатов по `gain`. Первые 10 кандидатов первого этапа:

| rank | feature                        | gain     |
| ---- | ------------------------------ | -------- |
| 1    | TEXTURE_GLSZM_ET_T2_GLV        | 9.714305 |
| 2    | TEXTURE_GLSZM_NET_T1_SZE       | 7.529892 |
| 3    | TEXTURE_GLRLM_NET_T2_RLV       | 7.397038 |
| 4    | TEXTURE_GLRLM_ED_T2_GLV        | 4.816936 |
| 5    | HISTO_ED_T2_Bin8               | 4.166760 |
| 6    | TEXTURE_GLCM_NET_FLAIR_Energy  | 3.997004 |
| 7    | HISTO_NET_FLAIR_Bin4           | 2.941544 |
| 8    | TEXTURE_GLCM_ED_T2_Homogeneity | 2.077134 |
| 9    | TEXTURE_GLCM_ED_FLAIR_Entropy  | 1.988917 |
| 10   | HISTO_ET_T2_Bin6               | 1.971020 |

После Genetic Algorithm в итоговый subset вошло **21** признаков.

GA оптимизирует комбинацию признаков целиком по внутренней 5-fold CV accuracy Random Forest. Порядок строк ниже **не является рейтингом важности**.

<details>
<summary>Показать итоговый GA subset (21 признаков)</summary>

| feature                            |
| ---------------------------------- |
| TEXTURE_GLSZM_NET_T1_SZE           |
| TEXTURE_GLRLM_ED_T2_GLV            |
| TEXTURE_GLCM_NET_FLAIR_Energy      |
| HISTO_NET_FLAIR_Bin4               |
| TEXTURE_GLCM_ED_T2_Homogeneity     |
| TEXTURE_GLCM_ED_FLAIR_Entropy      |
| HISTO_ET_T2_Bin6                   |
| SPATIAL_Temporal                   |
| TEXTURE_GLRLM_NET_T2_SRLGE         |
| TEXTURE_GLSZM_ED_T2_GLN            |
| TEXTURE_GLRLM_NET_T1_GLV           |
| HISTO_NET_FLAIR_Bin7               |
| TEXTURE_NGTDM_NET_FLAIR_Contrast   |
| HISTO_ET_T1Gd_Bin7                 |
| TEXTURE_GLSZM_NET_T2_SZHGE         |
| TEXTURE_GLSZM_ET_T1_ZSV            |
| INTENSITY_Mean_ET_T1               |
| TEXTURE_GLOBAL_ET_T2_Kurtosis      |
| TEXTURE_GLCM_NET_FLAIR_Correlation |
| TEXTURE_GLOBAL_ED_T2_Skewness      |
| VOLUME_ET_OVER_ED                  |

</details>

### Calabrese-inspired — Mutual Information → RF-RFE

Выбрано **32** признаков. Основной итоговый порядок задаётся `final_rank`; он строится по среднему RFE-rank между folds, а MI-rank используется как дополнительный критерий при равенстве.

| final_rank | feature                            | average_rank | rank_std | mi_rank | mi_score |
| ---------- | ---------------------------------- | ------------ | -------- | ------- | -------- |
| 1          | HISTO_ED_T2_Bin5                   | 1.000000     | 0.000000 | 1       | 0.238491 |
| 2          | TEXTURE_GLSZM_NET_FLAIR_SZE        | 1.000000     | 0.000000 | 12      | 0.186783 |
| 3          | TEXTURE_GLRLM_NET_T1_GLV           | 1.000000     | 0.000000 | 27      | 0.137030 |
| 4          | TEXTURE_GLRLM_ED_T2_GLV            | 1.000000     | 0.000000 | 43      | 0.125478 |
| 5          | HISTO_ET_T2_Bin6                   | 1.200000     | 0.447214 | 2       | 0.230203 |
| 6          | TEXTURE_GLSZM_NET_FLAIR_ZSN        | 1.200000     | 0.447214 | 5       | 0.204236 |
| 7          | TEXTURE_GLSZM_NET_FLAIR_ZP         | 1.200000     | 0.447214 | 32      | 0.133918 |
| 8          | TEXTURE_GLRLM_NET_T2_SRLGE         | 1.400000     | 0.894427 | 175     | 0.050615 |
| 9          | TEXTURE_GLSZM_ED_T2_HGZE           | 1.400000     | 0.547723 | 197     | 0.043732 |
| 10         | TEXTURE_GLSZM_NET_T1_SZE           | 1.400000     | 0.547723 | 255     | 0.029678 |
| 11         | TEXTURE_GLCM_ED_T2_Energy          | 1.800000     | 0.836660 | 44      | 0.125043 |
| 12         | VOLUME_BRAIN                       | 1.800000     | 0.836660 | 122     | 0.067349 |
| 13         | TEXTURE_GLSZM_ED_T2_GLN            | 1.800000     | 1.788854 | 183     | 0.048261 |
| 14         | TEXTURE_GLRLM_ED_T2_RLV            | 1.800000     | 1.303840 | 239     | 0.033250 |
| 15         | TEXTURE_GLRLM_NET_FLAIR_RLN        | 2.200000     | 1.303840 | 46      | 0.121019 |
| 16         | TEXTURE_GLSZM_NET_T1_ZSN           | 2.200000     | 1.303840 | 229     | 0.034433 |
| 17         | TEXTURE_GLRLM_NET_FLAIR_RP         | 2.800000     | 2.683282 | 34      | 0.133287 |
| 18         | TEXTURE_GLRLM_NET_FLAIR_SRE        | 2.800000     | 2.049390 | 36      | 0.132781 |
| 19         | TEXTURE_GLRLM_ED_T1Gd_SRLGE        | 3.200000     | 1.788854 | 7       | 0.196217 |
| 20         | TEXTURE_GLSZM_ED_FLAIR_SZE         | 3.200000     | 3.346640 | 184     | 0.047872 |
| 21         | TEXTURE_GLCM_ED_FLAIR_Correlation  | 3.200000     | 2.167948 | 258     | 0.028409 |
| 22         | TEXTURE_GLCM_ED_T2_AutoCorrelation | 3.400000     | 4.277850 | 181     | 0.049137 |
| 23         | TEXTURE_GLSZM_ED_FLAIR_ZSN         | 3.400000     | 4.335897 | 290     | 0.019256 |
| 24         | TEXTURE_GLRLM_ED_FLAIR_LRLGE       | 3.400000     | 2.607681 | 665     | 0.000000 |
| 25         | TEXTURE_GLSZM_ED_T1_LGZE           | 3.600000     | 2.073644 | 40      | 0.128330 |
| 26         | TEXTURE_GLSZM_ED_T2_SZHGE          | 3.600000     | 4.219005 | 126     | 0.065996 |
| 27         | TEXTURE_GLSZM_NET_FLAIR_LZHGE      | 4.000000     | 5.612486 | 243     | 0.032652 |
| 28         | TEXTURE_GLRLM_NET_T1_RLN           | 4.600000     | 4.098780 | 24      | 0.146231 |
| 29         | TEXTURE_GLSZM_ED_T1Gd_LGZE         | 4.600000     | 3.361547 | 37      | 0.131468 |
| 30         | TEXTURE_GLSZM_NET_T1_ZP            | 4.600000     | 7.503333 | 101     | 0.079491 |
| 31         | TEXTURE_GLSZM_NET_T2_LGZE          | 4.800000     | 4.086563 | 218     | 0.037570 |
| 32         | TEXTURE_GLSZM_NET_T2_SZLGE         | 4.800000     | 6.833740 | 269     | 0.026024 |

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


## Выбранные признаки — UPenn-GBM

Ниже приведены итоговые признаки, которые были переданы дальше как результат каждого метода feature selection.

**Важно:** F-score и MI→RF-RFE формируют ранжирование признаков. Genetic Algorithm возвращает лучший найденный subset целиком, поэтому внутри финального GA-набора нет отдельного рейтинга «1-е, 2-е, 3-е место».

### Le et al. — F-score

Выбрано **9** признаков. Они отсортированы по убыванию F-score.

| rank | feature                                                       | f_score  |
| ---- | ------------------------------------------------------------- | -------- |
| 1    | T2_ED_Histogram_Bins-16_Bins-16_Bin-8_Probability             | 0.030292 |
| 2    | FLAIR_ET_Morphologic_Eccentricity                             | 0.027273 |
| 3    | T1GD_NC_Intensity_Maximum                                     | 0.025217 |
| 4    | T2_ED_Histogram_Bins-16_Bins-16_CoefficientOfVariation        | 0.024664 |
| 5    | T2_ED_Histogram_Bins-16_Bins-16_Bin-7_Probability             | 0.024513 |
| 6    | T1GD_NC_Intensity_Range                                       | 0.024431 |
| 7    | T1_NC_GLSZM_Bins-16_Radius-1_LowGreyLevelEmphasis             | 0.023973 |
| 8    | T1GD_ED_Histogram_Bins-16_Bins-16_Bin-10_Probability          | 0.022145 |
| 9    | T1_ED_GLSZM_Bins-16_Radius-1_GreyLevelNonUniformityNormalized | 0.021487 |

### Do et al. — XGBoost gain → GA-RF

Сначала XGBoost формирует ranking кандидатов по `gain`. Первые 10 кандидатов первого этапа:

| rank | feature                                                          | gain     |
| ---- | ---------------------------------------------------------------- | -------- |
| 1    | FLAIR_ET_Histogram_Bins-16_Bins-16_MedianAbsoluteDeviation       | 9.474392 |
| 2    | FLAIR_ET_GLSZM_Bins-16_Radius-1_GreyLevelNonUniformityNormalized | 8.309825 |
| 3    | T2_NC_GLSZM_Bins-16_Radius-1_SmallZoneEmphasis                   | 7.699777 |
| 4    | T1GD_NC_Intensity_Range                                          | 7.117986 |
| 5    | T1GD_NC_Intensity_Mode                                           | 6.856683 |
| 6    | T1GD_ED_Histogram_Bins-16_Bins-16_Bin-3_Probability              | 6.833370 |
| 7    | T1_ET_GLSZM_Bins-16_Radius-1_SmallZoneLowGreyLevelEmphasis       | 6.618685 |
| 8    | T2_ET_Histogram_Bins-16_Bins-16_Bin-5_Frequency                  | 6.590713 |
| 9    | T1_NC_Intensity_Skewness                                         | 6.205830 |
| 10   | T2_NC_Intensity_Energy                                           | 5.749543 |

После Genetic Algorithm в итоговый subset вошло **136** признаков.

GA оптимизирует комбинацию признаков целиком по внутренней 5-fold CV accuracy Random Forest. Порядок строк ниже **не является рейтингом важности**.

<details>
<summary>Показать итоговый GA subset (136 признаков)</summary>

| feature                                                            |
| ------------------------------------------------------------------ |
| FLAIR_ET_GLSZM_Bins-16_Radius-1_GreyLevelNonUniformityNormalized   |
| T1GD_NC_Intensity_Range                                            |
| T1GD_NC_Intensity_Mode                                             |
| T1_ET_GLSZM_Bins-16_Radius-1_SmallZoneLowGreyLevelEmphasis         |
| T2_NC_Intensity_Energy                                             |
| FLAIR_ED_Histogram_Bins-16_Bins-16_Bin-6_Probability               |
| T1_ET_Intensity_Range                                              |
| T1GD_ED_Histogram_Bins-16_Bins-16_Bin-15_Probability               |
| T2_NC_Intensity_Mode                                               |
| T2_NC_Histogram_Bins-16_Bins-16_Bin-15_Frequency                   |
| FLAIR_NC_GLSZM_Bins-16_Radius-1_GreyLevelMean                      |
| T1_NC_Intensity_NinetiethPercentile                                |
| T1_ED_Histogram_Bins-16_Bins-16_MeanAbsoluteDeviation              |
| FLAIR_ET_Histogram_Bins-16_Bins-16_Bin-13_Probability              |
| FLAIR_ET_Histogram_Bins-16_Bins-16_Bin-2_Frequency                 |
| T2_NC_Histogram_Bins-16_Bins-16_Bin-1_Probability                  |
| T1GD_NC_Histogram_Bins-16_Bins-16_Bin-6_Probability                |
| T1_ET_GLCM_Bins-16_Radius-1_Homogeneity                            |
| T1GD_NC_GLSZM_Bins-16_Radius-1_GreyLevelNonUniformityNormalized    |
| T2_NC_Histogram_Bins-16_Bins-16_Bin-15_Probability                 |
| T1GD_ED_Histogram_Bins-16_Bins-16_Bin-5_Frequency                  |
| T1_ET_Histogram_Bins-16_Bins-16_Bin-15_Probability                 |
| FLAIR_ET_Morphologic_Elongation                                    |
| FLAIR_ED_GLSZM_Bins-16_Radius-1_ZoneSizeNonUniformity              |
| T2_ED_GLRLM_Bins-16_Radius-1_LowGreyLevelRunEmphasis               |
| FLAIR_NC_GLCM_Bins-16_Radius-1_AutoCorrelation                     |
| T1GD_ED_Histogram_Bins-16_Bins-16_Bin-6_Probability                |
| T2_ED_Intensity_MedianAbsoluteDeviation                            |
| T1GD_NC_Histogram_Bins-16_Bins-16_Bin-0_Frequency                  |
| T1GD_ED_GLCM_Bins-16_Radius-1_Homogeneity                          |
| T2_ET_Histogram_Bins-16_Bins-16_Mean                               |
| T1GD_ET_GLSZM_Bins-16_Radius-1_ZoneSizeNoneUniformityNormalized    |
| T2_NC_Intensity_Minimum                                            |
| T1GD_NC_Intensity_Kurtosis                                         |
| T2_NC_GLCM_Bins-16_Radius-1_AutoCorrelation                        |
| T1GD_ET_GLSZM_Bins-16_Radius-1_SmallZoneEmphasis                   |
| FLAIR_ED_Histogram_Bins-16_Bins-16_Bin-15_Probability              |
| FLAIR_ET_Morphologic_Flatness                                      |
| T1_NC_GLCM_Bins-16_Radius-1_ClusterProminence                      |
| T1_NC_Histogram_Bins-16_Bins-16_Bin-7_Frequency                    |
| T1_ET_GLSZM_Bins-16_Radius-1_ZoneSizeEntropy                       |
| T1_ET_GLCM_Bins-16_Radius-1_ClusterProminence                      |
| FLAIR_ET_Intensity_Minimum                                         |
| T1GD_ET_Histogram_Bins-16_Bins-16_Bin-8_Frequency                  |
| FLAIR_ED_GLRLM_Bins-16_Radius-1_LongRunHighGreyLevelEmphasis       |
| T1_NC_Intensity_StandardDeviation                                  |
| FLAIR_NC_Intensity_QuartileCoefficientOfVariation                  |
| T1GD_ED_Intensity_CoefficientOfVariation                           |
| T1GD_NC_Intensity_QuartileCoefficientOfVariation                   |
| T2_ED_Histogram_Bins-16_Bins-16_Bin-15_Probability                 |
| T1GD_NC_GLRLM_Bins-16_Radius-1_LongRunLowGreyLevelEmphasis         |
| FLAIR_ED_NGTDM_Coarsness                                           |
| T1_ET_Histogram_Bins-16_Bins-16_MedianAbsoluteDeviation            |
| T1_ED_Histogram_Bins-16_Bins-16_Bin-2_Probability                  |
| T1GD_ET_Intensity_Skewness                                         |
| T1_ED_Intensity_Skewness                                           |
| T2_ET_GLSZM_Bins-16_Radius-1_GreyLevelVariance                     |
| T1GD_NC_Histogram_Bins-16_Bins-16_Bin-11_Frequency                 |
| FLAIR_NC_Intensity_MeanAbsoluteDeviation                           |
| T1_ET_Histogram_Bins-16_Bins-16_Bin-13_Frequency                   |
| T1_ET_Histogram_Bins-16_Bins-16_Uniformity                         |
| T2_ED_Histogram_Bins-16_Bins-16_Bin-13_Frequency                   |
| FLAIR_NC_Histogram_Bins-16_Bins-16_Skewness                        |
| T1GD_ED_GLCM_Bins-16_Radius-1_Entropy                              |
| FLAIR_ET_Intensity_MeanAbsoluteDeviation                           |
| T1GD_NC_GLSZM_Bins-16_Radius-1_ZoneSizeEntropy                     |
| T1GD_ET_GLRLM_Bins-16_Radius-1_HighGreyLevelRunEmphasis            |
| T1GD_ED_GLSZM_Bins-16_Radius-1_SmallZoneEmphasis                   |
| T1_ET_GLSZM_Bins-16_Radius-1_GreyLevelNonUniformityNormalized      |
| FLAIR_ET_Histogram_Bins-16_Bins-16_RobustMeanAbsoluteDeviation1090 |
| T2_ED_Intensity_QuartileCoefficientOfVariation                     |
| T2_ED_Histogram_Bins-16_Bins-16_Bin-15_Frequency                   |
| T1_NC_Histogram_Bins-16_Bins-16_QuartileCoefficientOfVariation     |
| T2_ET_Histogram_Bins-16_Bins-16_Bin-15_Frequency                   |
| T2_NC_GLSZM_Bins-16_Radius-1_SmallZoneLowGreyLevelEmphasis         |
| T2_ED_Histogram_Bins-16_Bins-16_Bin-14_Frequency                   |
| FLAIR_ET_Histogram_Bins-16_Bins-16_Bin-13_Frequency                |
| T1GD_ET_Histogram_Bins-16_Bins-16_Bin-5_Probability                |
| T1GD_ED_Histogram_Bins-16_Bins-16_MedianAbsoluteDeviation          |
| T1_ED_Intensity_MedianAbsoluteDeviation                            |
| T1GD_ET_GLSZM_Bins-16_Radius-1_HighGreyLevelEmphasis               |
| T1_NC_GLCM_Bins-16_Radius-1_Homogeneity                            |
| FLAIR_ET_Histogram_Bins-16_Bins-16_Bin-3_Probability               |
| T2_ED_GLSZM_Bins-16_Radius-1_ZoneSizeEntropy                       |
| T1GD_ET_Histogram_Bins-16_Bins-16_Bin-13_Frequency                 |
| T1_ED_Histogram_Bins-16_Bins-16_Mode                               |
| T2_NC_Histogram_Bins-16_Bins-16_Bin-7_Probability                  |
| T1_NC_Intensity_Maximum                                            |
| T2_ED_Intensity_RootMeanSquare                                     |
| T2_NC_Intensity_QuartileCoefficientOfVariation                     |
| T1GD_ED_Intensity_MeanAbsoluteDeviation                            |
| T1_ED_Histogram_Bins-16_Bins-16_Bin-4_Frequency                    |
| T2_ED_Histogram_Bins-16_Bins-16_Bin-1_Probability                  |
| T2_ED_Histogram_Bins-16_Bins-16_Bin-13_Probability                 |
| T1GD_ET_GLSZM_Bins-16_Radius-1_SmallZoneLowGreyLevelEmphasis       |
| FLAIR_ED_GLRLM_Bins-16_Radius-1_LongRunEmphasis                    |
| T2_NC_Histogram_Bins-16_Bins-16_Bin-10_Probability                 |
| T2_NC_Histogram_Bins-16_Bins-16_Bin-12_Probability                 |
| T1GD_ET_GLSZM_Bins-16_Radius-1_LowGreyLevelEmphasis                |
| T1GD_ET_Intensity_InterQuartileRange                               |
| FLAIR_ED_Intensity_Energy                                          |
| T2_NC_Histogram_Bins-16_Bins-16_Bin-8_Probability                  |
| T2_NC_GLSZM_Bins-16_Radius-1_ZoneSizeEntropy                       |
| T2_NC_GLSZM_Bins-16_Radius-1_LargeZoneHighGreyLevelEmphasis        |
| T1_ED_GLSZM_Bins-16_Radius-1_SmallZoneEmphasis                     |
| T2_ED_GLCM_Bins-16_Radius-1_Homogeneity                            |
| T2_ED_Intensity_Mode                                               |
| T1_ED_Histogram_Bins-16_Bins-16_Bin-9_Probability                  |
| T2_NC_GLSZM_Bins-16_Radius-1_LowGreyLevelEmphasis                  |
| T2_ET_NGTDM_Complexity                                             |
| T1_ED_GLSZM_Bins-16_Radius-1_LowGreyLevelEmphasis                  |
| FLAIR_ET_NGTDM_Contrast                                            |
| T2_NC_GLSZM_Bins-16_Radius-1_ZoneSizeNonUniformity                 |
| T2_NC_Intensity_InterQuartileRange                                 |
| T1_ED_Histogram_Bins-16_Bins-16_Bin-15_Frequency                   |
| FLAIR_ED_Morphologic_Elongation                                    |
| T1GD_ET_Histogram_Bins-16_Bins-16_Bin-15_Probability               |
| FLAIR_ED_Intensity_MeanAbsoluteDeviation                           |
| T2_NC_Histogram_Bins-16_Bins-16_NinetyFifthPercentileMean          |
| T1GD_NC_Histogram_Bins-16_Bins-16_Bin-15_Probability               |
| T1_ED_Intensity_Mode                                               |
| FLAIR_NC_GLSZM_Bins-16_Radius-1_SmallZoneLowGreyLevelEmphasis      |
| T1GD_ED_Histogram_Bins-16_Bins-16_CoefficientOfVariation           |
| T2_NC_GLRLM_Bins-16_Radius-1_LongRunEmphasis                       |
| T1GD_ED_Histogram_Bins-16_Bins-16_MeanAbsoluteDeviation            |
| T1_NC_GLCM_Bins-16_Radius-1_Contrast                               |
| T1GD_ET_GLSZM_Bins-16_Radius-1_GreyLevelVariance                   |
| FLAIR_ED_Morphologic_EllipseDiameter_Axis-0                        |
| FLAIR_NC_Histogram_Bins-16_Bins-16_Kurtosis                        |
| T1_NC_Histogram_Bins-16_Bins-16_Bin-10_Probability                 |
| T2_ED_GLSZM_Bins-16_Radius-1_SmallZoneLowGreyLevelEmphasis         |
| T1GD_ET_Histogram_Bins-16_Bins-16_Bin-6_Frequency                  |
| T2_ED_GLSZM_Bins-16_Radius-1_SmallZoneEmphasis                     |
| FLAIR_ET_Histogram_Bins-16_Bins-16_Mean                            |
| FLAIR_NC_GLRLM_Bins-16_Radius-1_LongRunLowGreyLevelEmphasis        |
| FLAIR_NC_Histogram_Bins-16_Bins-16_Bin-4_Frequency                 |

</details>

### Calabrese-inspired — Mutual Information → RF-RFE

Выбрано **32** признаков. Основной итоговый порядок задаётся `final_rank`; он строится по среднему RFE-rank между folds, а MI-rank используется как дополнительный критерий при равенстве.

| final_rank | feature                                                       | average_rank | rank_std | mi_rank | mi_score |
| ---------- | ------------------------------------------------------------- | ------------ | -------- | ------- | -------- |
| 1          | T2_NC_Intensity_Mode                                          | 1.000000     | 0.000000 | 9       | 0.080925 |
| 2          | T1GD_ED_Histogram_Bins-16_Bins-16_Bin-6_Probability           | 1.000000     | 0.000000 | 460     | 0.012330 |
| 3          | T1_ED_Histogram_Bins-16_Bins-16_Bin-2_Probability             | 1.200000     | 0.447214 | 77      | 0.050366 |
| 4          | FLAIR_ET_Morphologic_Elongation                               | 1.200000     | 0.447214 | 253     | 0.027060 |
| 5          | T1GD_ET_Histogram_Bins-16_Bins-16_MedianAbsoluteDeviation     | 1.200000     | 0.447214 | 282     | 0.025225 |
| 6          | T1GD_NC_Intensity_Range                                       | 1.400000     | 0.547723 | 24      | 0.069844 |
| 7          | T1_ET_Histogram_Bins-16_Bins-16_MedianAbsoluteDeviation       | 1.400000     | 0.894427 | 547     | 0.007717 |
| 8          | FLAIR_ET_Morphologic_Eccentricity                             | 1.400000     | 0.547723 | 844     | 0.000000 |
| 9          | T2_NC_Histogram_Bins-16_Bins-16_Bin-15_Frequency              | 1.800000     | 1.303840 | 58      | 0.055826 |
| 10         | T2_ET_Histogram_Bins-16_Bins-16_Bin-14_Frequency              | 2.000000     | 1.732051 | 312     | 0.022962 |
| 11         | T2_ED_Histogram_Bins-16_Bins-16_Bin-7_Probability             | 2.000000     | 1.414214 | 390     | 0.017386 |
| 12         | FLAIR_ET_Morphologic_Flatness                                 | 2.200000     | 1.303840 | 847     | 0.000000 |
| 13         | T1_ED_GLCM_Bins-16_Radius-1_ClusterShade                      | 2.400000     | 0.894427 | 670     | 0.001634 |
| 14         | T2_ET_Intensity_CoefficientOfVariation                        | 3.000000     | 1.414214 | 1       | 0.091800 |
| 15         | T2_NC_GLRLM_Bins-16_Radius-1_LongRunLowGreyLevelEmphasis      | 3.000000     | 2.915476 | 342     | 0.020882 |
| 16         | T1GD_NC_Histogram_Bins-16_Bins-16_Bin-0_Frequency             | 3.000000     | 3.464102 | 412     | 0.015728 |
| 17         | T1GD_NC_Histogram_Bins-16_Bins-16_Bin-0_Probability           | 3.000000     | 3.391165 | 596     | 0.005623 |
| 18         | T1GD_NC_Intensity_Maximum                                     | 3.400000     | 4.335897 | 283     | 0.025219 |
| 19         | FLAIR_NC_Intensity_InterQuartileRange                         | 3.600000     | 2.607681 | 159     | 0.035453 |
| 20         | T1GD_ET_GLSZM_Bins-16_Radius-1_HighGreyLevelEmphasis          | 3.600000     | 3.209361 | 988     | 0.000000 |
| 21         | FLAIR_NC_Morphologic_Flatness                                 | 4.000000     | 3.741657 | 690     | 0.001003 |
| 22         | T2_ED_Histogram_Bins-16_Bins-16_Bin-2_Probability             | 4.200000     | 2.588436 | 29      | 0.065732 |
| 23         | FLAIR_ET_Histogram_Bins-16_Bins-16_Bin-12_Frequency           | 4.200000     | 3.834058 | 204     | 0.031761 |
| 24         | T2_NC_Intensity_InterQuartileRange                            | 4.200000     | 3.962323 | 544     | 0.007909 |
| 25         | T2_ED_GLSZM_Bins-16_Radius-1_ZoneSizeEntropy                  | 4.200000     | 2.280351 | 666     | 0.001965 |
| 26         | T2_ED_Histogram_Bins-16_Bins-16_Bin-2_Frequency               | 4.400000     | 3.049590 | 69      | 0.051986 |
| 27         | T2_ET_Intensity_MedianAbsoluteDeviation                       | 4.400000     | 2.966479 | 636     | 0.003796 |
| 28         | T2_NC_Histogram_Bins-16_Bins-16_Bin-14_Frequency              | 4.600000     | 3.130495 | 105     | 0.044018 |
| 29         | T2_ET_Histogram_Bins-16_Bins-16_Bin-8_Probability             | 4.600000     | 2.880972 | 316     | 0.022676 |
| 30         | T2_ET_GLSZM_Bins-16_Radius-1_ZoneSizeNoneUniformityNormalized | 4.800000     | 4.494441 | 19      | 0.071713 |
| 31         | T1_ED_Histogram_Bins-16_Bins-16_Uniformity                    | 4.800000     | 5.495453 | 20      | 0.071392 |
| 32         | T1_ED_Histogram_Bins-16_Bins-16_Bin-1_Probability             | 4.800000     | 3.114482 | 680     | 0.001294 |

## Попарное пересечение выбранных признаков на UPenn

| method_a  | method_b         | intersection | union | jaccard  |
| --------- | ---------------- | ------------ | ----- | -------- |
| le_fscore | do_xgb_ga        | 1            | 144   | 0.006944 |
| le_fscore | calabrese_mi_rfe | 4            | 37    | 0.108108 |
| do_xgb_ga | calabrese_mi_rfe | 12           | 156   | 0.076923 |

## Состав выбранных признаков на UPenn

### Семейства признаков

| method           | dimension | value       | count | fraction |
| ---------------- | --------- | ----------- | ----- | -------- |
| le_fscore        | family    | HISTOGRAM   | 4     | 0.444444 |
| le_fscore        | family    | INTENSITY   | 2     | 0.222222 |
| le_fscore        | family    | GLSZM       | 2     | 0.222222 |
| le_fscore        | family    | MORPHOLOGIC | 1     | 0.111111 |
| do_xgb_ga        | family    | HISTOGRAM   | 55    | 0.404412 |
| do_xgb_ga        | family    | INTENSITY   | 30    | 0.220588 |
| do_xgb_ga        | family    | GLSZM       | 27    | 0.198529 |
| do_xgb_ga        | family    | GLCM        | 10    | 0.073529 |
| do_xgb_ga        | family    | GLRLM       | 7     | 0.051471 |
| do_xgb_ga        | family    | MORPHOLOGIC | 4     | 0.029412 |
| do_xgb_ga        | family    | NGTDM       | 3     | 0.022059 |
| calabrese_mi_rfe | family    | HISTOGRAM   | 16    | 0.500000 |
| calabrese_mi_rfe | family    | INTENSITY   | 7     | 0.218750 |
| calabrese_mi_rfe | family    | MORPHOLOGIC | 4     | 0.125000 |
| calabrese_mi_rfe | family    | GLSZM       | 3     | 0.093750 |
| calabrese_mi_rfe | family    | GLCM        | 1     | 0.031250 |
| calabrese_mi_rfe | family    | GLRLM       | 1     | 0.031250 |

### MRI-модальности

| method           | dimension | value        | count | fraction |
| ---------------- | --------- | ------------ | ----- | -------- |
| le_fscore        | modality  | T1GD         | 3     | 0.333333 |
| le_fscore        | modality  | T2           | 3     | 0.333333 |
| le_fscore        | modality  | T1           | 2     | 0.222222 |
| le_fscore        | modality  | SHAPE_SHARED | 1     | 0.111111 |
| do_xgb_ga        | modality  | T2           | 39    | 0.286765 |
| do_xgb_ga        | modality  | T1GD         | 36    | 0.264706 |
| do_xgb_ga        | modality  | T1           | 30    | 0.220588 |
| do_xgb_ga        | modality  | FLAIR        | 27    | 0.198529 |
| do_xgb_ga        | modality  | SHAPE_SHARED | 4     | 0.029412 |
| calabrese_mi_rfe | modality  | T2           | 14    | 0.437500 |
| calabrese_mi_rfe | modality  | T1GD         | 7     | 0.218750 |
| calabrese_mi_rfe | modality  | T1           | 5     | 0.156250 |
| calabrese_mi_rfe | modality  | SHAPE_SHARED | 4     | 0.125000 |
| calabrese_mi_rfe | modality  | FLAIR        | 2     | 0.062500 |

### Области опухоли

| method           | dimension | value | count | fraction |
| ---------------- | --------- | ----- | ----- | -------- |
| le_fscore        | region    | ED    | 5     | 0.555556 |
| le_fscore        | region    | NC    | 3     | 0.333333 |
| le_fscore        | region    | ET    | 1     | 0.111111 |
| do_xgb_ga        | region    | NC    | 49    | 0.360294 |
| do_xgb_ga        | region    | ED    | 47    | 0.345588 |
| do_xgb_ga        | region    | ET    | 40    | 0.294118 |
| calabrese_mi_rfe | region    | ET    | 12    | 0.375000 |
| calabrese_mi_rfe | region    | NC    | 11    | 0.343750 |
| calabrese_mi_rfe | region    | ED    | 9     | 0.281250 |

## Дополнительное сравнение по ROC-AUC на UPenn

| method           | selected_features | roc_auc  | balanced_accuracy | accuracy |
| ---------------- | ----------------- | -------- | ----------------- | -------- |
| do_xgb_ga        | 136               | 0.606169 | 0.565065          | 0.578125 |
| le_fscore        | 9                 | 0.596847 | 0.551301          | 0.578125 |
| calabrese_mi_rfe | 32                | 0.583458 | 0.563939          | 0.589844 |

ROC-AUC в этой таблице используется для внутреннего сравнения готовых feature subsets. Это не независимая оценка generalization, поскольку feature selection был выполнен на полном UPenn dataset до этой CV.
