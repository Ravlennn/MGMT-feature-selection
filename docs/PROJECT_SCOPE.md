# Project scope

Original follow-up plan:

1. Refresh knowledge from review articles and the selected papers.
2. Search GitHub for implementations of radiomic feature-processing approaches from those papers.
3. Try the approaches first on the original/reference data and then on UPenn-GBM.
4. Compare at least 2–3 approaches and prepare a comparative report.

This repository implements exactly that scope with three approaches:

- **Le et al.** — F-score ranking, top-9 feature subset.
- **Do et al.** — XGBoost gain preselection followed by GA-RF.
- **Calabrese-inspired** — Mutual Information preselection followed by RF-RFE, final 32 features.

## What is in scope

- Reproduce/check the feature-selection logic on the small reference table where possible.
- Build one common structural-radiomics UPenn dataset.
- Apply all three selectors to the same UPenn feature matrix.
- Compare feature-count reduction, feature overlap, feature families, MRI modalities, and tumor regions.
- Generate report-ready tables automatically.

## What is deliberately out of scope for the main pipeline

- Nested cross-validation benchmarking.
- Extensive classifier/hyperparameter tuning.
- Treating predictive AUC as the main project target.
- Full DTI/DSC ablation studies.
- VASARI integration unless there is extra time.

Those can be supplementary experiments, but they are not required to complete the follow-up.
