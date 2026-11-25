# Data Cleaning and Preprocessing Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white) ![seaborn](https://img.shields.io/badge/seaborn-4C72B0) ![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green)

**IncodeVision Data Science Internship | Task 01 (Track Code: DS)**

A reusable Python pipeline that takes a raw CSV with missing values, duplicates and inconsistent data and turns it into a clean, analysis-ready and model-ready dataset.

**Dataset:** Titanic passengers (891 rows, 15 columns), open-source from the [seaborn-data](https://github.com/mwaskom/seaborn-data) repository.

## Results

| Metric | Before | After |
|---|---|---|
| Rows | 891 | 784 |
| Duplicate rows | 107 | 0 |
| Missing cells | 869 | 0 |

## Pipeline steps

1. **Inspect** the raw data: shape, missing-value heatmap, duplicate count.
2. **Remove duplicates:** 107 exact duplicate rows dropped.
3. **Fix inconsistent text:** whitespace stripped, consistent casing (`sex`, `who` lower-case, `embarked` upper-case).
4. **Drop redundant columns:** `alive` (same as `survived`), `class` (same as `pclass`), `embark_town` (same as `embarked`).
5. **Handle missing values**
   - `deck` (688 missing, about 77%): too sparse to impute, so filled with `"Unknown"`.
   - `embarked` (2 missing): filled with the mode.
   - `age` (177 missing): filled with the median of the passenger's `pclass` + `sex` group (more accurate than one global median).
6. **Fix data types:** ints, booleans and categories set correctly.
7. **Handle outliers** with the IQR rule: 102 `fare` and 15 `age` outliers capped (winsorized) instead of deleted, so no rows are lost.
8. **Feature engineering, encoding and scaling:** `family_size = sibsp + parch + 1`, one-hot encoding for categorical columns, z-score scaling for numeric columns.
9. **Validate and save** clean data and a cleaning log.

## Project structure

```
DS-Task01-Data-Cleaning/
├── data_cleaning.ipynb   # notebook with outputs (start here)
├── data_cleaning.py          # full pipeline (run this)
├── requirements.txt
├── data/
│   ├── titanic_raw.csv       # original data
│   ├── titanic_clean.csv     # cleaned data
│   └── titanic_ml_ready.csv  # encoded + scaled, ready for modeling
└── outputs/
    ├── 01_missing_before.png
    ├── 02_outliers_fare.png
    ├── 03_missing_after.png
    ├── cleaning_summary.csv
    └── cleaning_log.txt
```

## How to run

Open the notebook for the full walkthrough with outputs: [`data_cleaning.ipynb`](data_cleaning.ipynb). Or run the script:

```bash
pip install -r requirements.txt
python data_cleaning.py
```

## Visuals

![Missing before](outputs/01_missing_before.png)
![Outliers](outputs/02_outliers_fare.png)
![Missing after](outputs/03_missing_after.png)

## Note

After age imputation, 7 rows look identical to others. The dataset has no passenger ID, so these are different people and were kept on purpose.

## Tech stack

Python, pandas, NumPy, matplotlib, seaborn.
