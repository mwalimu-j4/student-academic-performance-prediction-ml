# Predicting Student Dropout and Academic Success Using Machine Learning

Production-oriented research MVP: a FastAPI inference service and Next.js/Tailwind dashboard backed by a reproducible scikit-learn pipeline.

## What is implemented

- Official UCI Machine Learning Repository dataset 697: 4,424 unique records, 37 columns, no missing values, labels `Dropout`, `Enrolled`, `Graduate`.
- Leakage-controlled admission-time feature set (24 fields). All curricular-unit progress and grades are excluded because they are only observed after enrolment.
- Stratified 75/25 split: 3,318 training and 1,106 untouched test records. No records are duplicated.
- Training-only five-fold cross-validation comparing balanced and unweighted Logistic Regression, balanced and unweighted Random Forest, and a Random Forest with a 3x Enrolled class weight. The selected model is chosen by cross-validated macro-F1; the holdout is not used for selection.
- Persisted model, metadata, complete per-class/macro/weighted metrics, confusion matrix, and one-vs-rest ROC-AUC in `models/` and `results/`.
- FastAPI endpoints for health, prediction, metrics, dataset summary, and prediction history. PostgreSQL history is enabled with `DATABASE_URL`; local development uses in-memory history.
- Responsive Next.js/Tailwind prediction dashboard. It uses `NEXT_PUBLIC_API_URL` and never embeds credentials.

## Verified evaluation

The current run is recorded in [`results/uci697_metrics.json`](./results/uci697_metrics.json). Cross-validation selected the Random Forest with a 3x Enrolled class weight: CV macro-F1 **0.5821 ± 0.0164**, compared with **0.5754 ± 0.0142** for balanced Random Forest, **0.5232 ± 0.0140** for balanced Logistic Regression, and **0.5224 ± 0.0209** for unweighted Random Forest. Only after that selection, the 1,106-row held-out set was evaluated once: **0.6040 accuracy**, **0.5474 macro-F1**, **0.6056 weighted-F1**, and **0.7422 macro one-vs-rest ROC-AUC**.

The selected model's held-out per-class F1 was **0.6113 Dropout**, **0.3294 Enrolled**, and **0.7015 Graduate**. Its confusion matrix, in `[Dropout, Enrolled, Graduate]` order, was `[[195, 61, 99], [34, 70, 95], [54, 95, 403]]`.

The Enrolled class is difficult because it is the minority class (794 of 4,424 records), has substantial overlap with the other outcomes at admission time, and is confused with Graduate (95 of 199 Enrolled test records) and Dropout (34). The earlier balanced Random Forest had Enrolled F1 **0.3000**; the 3x training-only weight increased it to **0.3294**, while overall accuracy moved from **0.5913** to **0.6040**. This is measured improvement on the single holdout, not a guarantee. Further evidence-based work should use repeated or nested training-only CV to tune the Enrolled weight, probability calibration or a cost-sensitive decision rule, and external/time-based validation. Adding semester progress features could improve discrimination but would change the prediction time and create a different, later model.

These are retrospective results on one random split, not a guarantee of individual performance. The Enrolled class remains substantially harder to distinguish. The data comes from two Portuguese institutions and may not transfer to other populations, time periods, or policies. The dataset does not establish causal relationships, fairness, or intervention effectiveness.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\download_uci697.py
python scripts\train_uci697.py
pytest
```

Run the API:

```powershell
$env:CORS_ORIGINS = "http://localhost:3000"
uvicorn backend.app.main:app --reload --port 8000
```

Run the frontend:

```powershell
cd frontend
npm install
$env:NEXT_PUBLIC_API_URL = "http://localhost:8000"
npm run build
npm run start
```

## Deployment

1. **Neon/PostgreSQL:** create a database and set `DATABASE_URL` to its SSL connection string. The API creates `prediction_history` on first persisted prediction. Do not commit the value.
2. **Render:** create a Python web service rooted at `backend`, build with `pip install -r requirements.txt`, start with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and set `DATABASE_URL` and `CORS_ORIGINS` to the final Vercel origin. `backend/render.yaml` contains the same configuration. Render must have access to the repository-root `models/` and `results/` artifacts; if Render's root directory checkout does not include them, deploy from the repository root with `rootDir` omitted and use `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`.
3. **Vercel:** import the repository, set the root directory to `frontend`, build with `npm run build`, and set `NEXT_PUBLIC_API_URL` to the actual Render API URL. Update Render `CORS_ORIGINS` if the Vercel domain changes. Current frontend dependency audit is clean after upgrading Next.js to 16.4.3.
4. Verify `<render-url>/health`, `<render-url>/api/v1/metrics`, and a browser prediction before calling the deployment live. This repository does not claim that public services have been deployed.

## Dataset and license

UCI dataset page: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success  
DOI: https://doi.org/10.24432/C5MC89  
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

The downloaded archive is ignored by Git. Re-run the download and training scripts to reproduce artifacts. This application is decision support for research and should not be the sole basis for admission, grading, discipline, funding, or access to support.
