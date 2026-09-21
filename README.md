# MGMT Radiomics Feature Selection

A compact project for reproducing and comparing three radiomics feature-processing approaches for MGMT promoter methylation:

- **Le et al.** — F-score ranking.
- **Do et al.** — XGBoost gain → Genetic Algorithm with Random-Forest fitness.
- **Calabrese-inspired** — Mutual Information → Random-Forest RFE.

The project follows one simple flow:

```text
reference/original data
        ↓
3 feature-selection methods
        ↓
method sanity check / reproduction

official UPenn-GBM structural radiomics
        ↓
common cleaning
        ↓
3 feature-selection methods
        ↓
feature-set comparison
        ↓
report-ready tables + markdown summary
```

The main goal is **comparison of feature-processing approaches**, not construction of an optimized MGMT classifier.

## Repository structure

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
│   ├── build_report.py
│   └── run_all.py
├── results/
│   ├── reference/
│   └── upenn/
└── docs/
    └── PROJECT_SCOPE.md
```

Everything that was only useful while exploring the data (`audit_*`, nested CV, outer-fold generation, temporary benchmark scripts, `__pycache__`, `egg-info`, etc.) has been removed.

## Setup

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

Python 3.11+ is supported.

## Fastest path to the final report

Validated reference results are already included in `results/reference/`, so under a tight deadline you only need:

```powershell
python scripts/run_all.py --skip-reference
```

This command will:

1. rebuild `data/processed/upenn/upenn_clean.csv` from the official raw UPenn files;
2. run Le, Do and Calabrese-inspired feature selection on the same UPenn dataset;
3. compare the selected feature sets;
4. generate `docs/results_summary.md` with report-ready tables.

The Do GA-RF and Calabrese RF-RFE stages are the expensive parts.

## Full reproduction from scratch

To rerun both the reference stage and UPenn stage:

```powershell
python scripts/run_all.py
```

For a quick smoke check of the code with reduced GA/RF-RFE settings:

```powershell
python scripts/run_all.py --quick
```

Do **not** use `--quick` results in the final report.

## Individual pipelines

Build the canonical UPenn dataset:

```powershell
python scripts/build_upenn.py
```

Expected result:

```text
291 clinical cases with known MGMT
→ 256 cases with all 12 structural radiomics blocks
1728 raw structural features
→ remove 84 constants
→ remove 164 exact duplicates
→ median-impute 4 residual values
= 1480 clean radiomics features
```

Rerun the reference-data experiments:

```powershell
python scripts/run_reference.py
```

Apply all three approaches to UPenn:

```powershell
python scripts/run_upenn.py
```

Generate report-ready summary:

```powershell
python scripts/build_report.py
```

## Outputs used in the report

### Reference

`results/reference/summary.csv` contains the compact reference comparison.

Current validated reference result:

```text
Le:          704 → 9
Do:          704 → 38 XGB candidates → 21 GA features
Calabrese:   704 → 32
```

The Do GA internal 5-fold RF accuracy is approximately `0.9255` in the stored validated run.

For Le, the published top-9 ordering was reproduced exactly on the 53-patient table.

The Calabrese pipeline is a **method sanity check on the shared TCGA table**, not a reproduction of the original UCSF cohort because that original feature matrix is not included here.

### UPenn

After `run_upenn.py`:

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
    └── feature_composition.csv
```

`feature_composition.csv` automatically summarizes selected features by:

- feature family (`GLCM`, `GLRLM`, `GLSZM`, histogram, intensity, morphology, ...);
- MRI modality (`FLAIR`, `T1`, `T1GD`, `T2`);
- tumor region (`ED`, `ET`, `NC`).

Morphologic/volumetric features are marked as `SHAPE_SHARED`, because the same segmentation geometry is shared across structural MRI modalities and exact duplicate copies are removed during preprocessing.

## UPenn data choice

The main experiment uses only structural MRI radiomics:

```text
FLAIR × ED/ET/NC
T1GD  × ED/ET/NC
T1    × ED/ET/NC
T2    × ED/ET/NC
```

DTI and DSC blocks are intentionally excluded from the main pipeline because they substantially reduce the number of patients with complete data and make the comparison less aligned with the structural MRI approaches used in the reviewed papers.

## About the Kaggle nested-CV run

The long nested-CV Do experiment started during development can finish if desired, but it is **supplementary**. It is not required by the original follow-up and is not part of the core cleaned pipeline. If its results are useful, cite them separately as an additional validation experiment rather than mixing them with the main feature-selection comparison.

See `docs/PROJECT_SCOPE.md` whenever the project scope starts drifting again.
