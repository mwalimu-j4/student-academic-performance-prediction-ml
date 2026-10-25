# Student Academic Performance Prediction Using Supervised Machine Learning

Reproducible coursework project using the official [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

## Approved project design

- **Primary data:** Mathematics course (`student-mat.csv`) from the official UCI archive.
- **Target:** `G3`, the final grade documented by UCI as an integer from 0 to 20.
- **Classification label:** `pass` when `G3 >= 10`; otherwise `fail`. The threshold is an explicit project choice.
- **Primary predictors:** demographic, school, family, social, study, support, and attendance variables.
- **Excluded from the primary model:** `G1` and `G2`. They are earlier-period grades, and UCI warns that they are strongly correlated with `G3`. Excluding them makes the primary experiment a more realistic early prediction task.

The project does not commit the dataset. The download script retrieves the archive from the official UCI URL.

## Project structure

```text
student-academic-performance-prediction-ml/
├── data/
│   ├── raw/                 # downloaded CSV, ignored by Git
│   └── processed/           # optional processed data, ignored by Git
├── results/                 # metrics, plots, and fitted model outputs
├── RESULTS_AND_DISCUSSION.md
├── scripts/
│   ├── download_data.py
│   └── train_model.py
├── src/student_performance/
│   ├── data.py
│   └── modeling.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduce the primary experiment

```powershell
python scripts/download_data.py
python scripts/train_model.py
```

The training script:

1. Loads the semicolon-delimited UCI CSV.
2. Verifies the documented schema and target.
3. Creates the binary target from `G3`.
4. Excludes `G1`, `G2`, and `G3` from the primary predictors.
5. Uses a stratified holdout test set and cross-validation on the training set.
6. Compares a class-balanced Decision Tree baseline with a class-balanced Random Forest main model.
7. Fits one-hot encoding and all model parameters inside each scikit-learn `Pipeline`, using training data only.
8. Writes cross-validation metrics, untouched holdout-test metrics, confusion-matrix plots, a comparison plot/table, and fitted pipelines under `results/`.

The test set is created once with a fixed stratified split and is not used for model selection or preprocessing fitting. Test precision, recall, and F1 use binary averaging with `pass` (label 1) as the positive class. Model comparison emphasizes F1, recall, precision, and ROC-AUC alongside accuracy because the pass/fail classes are not perfectly balanced. The saved files are `decision_tree_pipeline.joblib` and `random_forest_pipeline.joblib`.

Evaluation outputs include `metrics.json`, `model_comparison.csv`, `model_comparison.png`, and one confusion-matrix image per model. The training command also reloads each saved pipeline and predicts two held-out examples as a serialization check.

For the practical question of identifying students at risk of failing, the evaluation also reports the `fail` class (label 0) recall, precision, and F1. The selection criterion is higher held-out fail recall, with held-out fail F1 as a tie-breaker. This is a comparison criterion, not a claim that an externally specified performance objective was achieved.

Metrics and scores are generated only by running the experiment; no results are claimed in this repository beforehand.

## Results and discussion

The complete academic Results and Discussion section, including the Summary and Conclusion, is in [RESULTS_AND_DISCUSSION.md](./RESULTS_AND_DISCUSSION.md). The values below are the results from the clean execution recorded in `results/metrics.json` and `results/model_comparison.csv`; they are not target claims or estimates from an unexecuted experiment.

The held-out test set contained 79 examples: 26 fail and 53 pass. Precision, recall, and F1 use binary averaging with `pass` (label 1) as the positive class. For the practical risk question, the comparison criterion was higher held-out fail recall, with fail F1 as the tie-breaker. No external performance threshold was specified.

| Model | Accuracy | Pass precision | Pass recall | Pass F1 | ROC-AUC | Fail precision | Fail recall | Fail F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Decision Tree | 0.5949 | 0.7234 | 0.6415 | 0.6800 | 0.5287 | 0.4063 | 0.5000 | 0.4483 |
| Random Forest | 0.7089 | 0.7679 | 0.8113 | 0.7890 | 0.6295 | 0.5652 | 0.5000 | 0.5306 |

The Random Forest is preferred because fail recall tied at 0.5000, while fail precision and fail F1 were higher. It also outperformed the Decision Tree on accuracy, pass F1, and ROC-AUC. This is a relative comparison, not evidence that an external performance objective was achieved. Both models missed 13 of 26 failing students in the test set, so neither should be used as a standalone high-stakes decision tool.

The five-fold training cross-validation means were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.6014 | 0.6998 | 0.7032 | 0.6971 | 0.5717 |
| Random Forest | 0.6773 | 0.7378 | 0.8065 | 0.7703 | 0.6795 |

Limitations include the small dataset (395 examples), only 26 fail examples in the test set, majority-pass class imbalance, and collection from two Portuguese secondary schools. Results may not generalize to other institutions or cohorts. Future improvements should include larger and more diverse data, external institution- or time-based validation, training-only threshold selection based on intervention costs, probability calibration, and subgroup performance analysis. The full discussion avoids adding unmeasured scores or unsupported conclusions.

## Data source

UCI Machine Learning Repository, dataset 320, *Student Performance*:

- Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance
- Download archive: https://archive.ics.uci.edu/static/public/320/student+performance.zip
- DOI: https://doi.org/10.24432/C5TG7T
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Original files: `student-mat.csv`, `student-por.csv`, `student.txt`, and `student-merge.R`
- Citation: Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
- Related paper: P. Cortez and A. Silva, “Using Data Mining to Predict Secondary School Student Performance,” FUBUTEC 2008, pp. 5–12.

The download script writes `data/raw/source_manifest.json`, including the source metadata and SHA-256 hashes of retrieved files. It also writes `results/data_verification.json`, containing the observed schema and data-quality audit.

## Acquisition and verification

Run:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
```

The script verifies the schema before creating transformations and reports row count, column count, pandas data types, exact duplicates, missing values, the observed `G3` distribution, binary pass/fail distribution, and unique values for every categorical column. Exact duplicate rows are removed only in the explicitly saved processed table and the number removed is recorded. Missing values cause processing to stop with an error rather than being silently deleted or imputed.

The UCI documentation defines `G3` on a 0–20 scale. This project uses `G3 >= 10` as pass and `G3 < 10` as fail. Because the intended prediction is made before the period grades are available, the processed early-prediction table excludes `G1`, `G2`, and `G3`; the binary label is derived from `G3` before it is removed.

## Testing

```powershell
python -m pytest
```
