# ML/DL Model Evaluation & Comparison Platform (for Imbalanced Datasets)

A research-oriented, Power BI–style platform for training and comparing
Machine Learning and Deep Learning classifiers on imbalanced datasets —
with proper methodology (no data leakage), imbalance-aware metrics, and
automatically generated, data-driven insights and conclusions.

```
Frontend (React + Vite + Tailwind + Recharts)  →  Backend (FastAPI + Python)
                                                        ├── scikit-learn (10 ML models)
                                                        ├── XGBoost, LightGBM
                                                        ├── TensorFlow/Keras (10 DL architectures)
                                                        ├── imbalanced-learn (SMOTE, ADASYN, ...)
                                                        └── SQLite (experiment storage)
```

---

## 1. What's included

- **10 ML models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN,
  Naive Bayes, Gradient Boosting, AdaBoost, XGBoost, LightGBM.
- **10 DL architectures**: ANN, MLP, CNN, 1D-CNN, LSTM, GRU, BiLSTM, BiGRU,
  CNN-LSTM, Attention Network — all built for tabular data.
- **8 imbalance techniques**: None, Random Oversampling, Random Undersampling,
  SMOTE, ADASYN, SMOTEENN, SMOTETomek, Class Weighting.
- **10 metrics**: Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC,
  MCC, Balanced Accuracy, G-Mean.
- Automatic dataset profiling (imbalance ratio, class distribution, missing
  values, duplicates — nothing hard-coded).
- Quick Analysis (few clicks, sensible defaults) and Advanced Analysis
  (test size, random seed, epochs, scaling, primary metric) modes.
- Model comparison, individual model drill-down, ML vs DL comparison,
  before/after imbalance-impact comparison.
- Automatically generated Insights and Conclusion sections — every sentence
  is computed from the actual experiment results, never templated.
- Experiment history: view, re-run, delete, export (JSON, CSV, Excel, PDF).
- Friendly error handling — no raw Python tracebacks reach the UI.

## 2. Project structure

```
ml-dashboard/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py, database.py, db_models.py, schemas.py
│   │   ├── ml/
│   │   │   ├── registry.py      # ← add new models/imbalance techniques HERE
│   │   │   ├── dataset_analysis.py
│   │   │   ├── preprocessing.py
│   │   │   ├── imbalance.py
│   │   │   ├── metrics.py
│   │   │   ├── train_ml.py
│   │   │   ├── train_dl.py
│   │   │   ├── insights.py
│   │   │   └── orchestrator.py  # ties everything together per experiment
│   │   ├── routers/             # dataset.py, models.py, experiments.py, reports.py
│   │   └── utils/errors.py
│   ├── sample_data/              # small synthetic imbalanced CSV to try immediately
│   ├── requirements.txt
│   └── run.py
└── frontend/
    ├── src/
    │   ├── pages/                # Home, Dataset, Models, NewExperiment, Dashboard,
    │   │                         # Compare, ModelDetail, Insights, History, Reports
    │   ├── components/           # ModelSelector, ImbalanceSelector, DatasetUploader,
    │   │                         # MetricSelector, charts/*, Layout/*
    │   ├── context/AppContext.jsx
    │   └── api/client.js
    └── package.json
```

## 3. Setup

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py                   # serves on http://localhost:8000
```

The first run creates `backend/storage/app.db` (SQLite) and
`backend/storage/datasets/` automatically. API docs are available at
`http://localhost:8000/docs` (FastAPI's built-in Swagger UI).

> **Note on TensorFlow / XGBoost / LightGBM**: these are the heaviest
> dependencies. If your machine can't install TensorFlow, the platform
> still works for all 10 ML models — DL models will simply return a
> friendly "could not be trained" error instead of crashing the app.

### Frontend

```bash
cd frontend
npm install
npm run dev                     # serves on http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://localhost:8000` (see
`vite.config.js`), so just open `http://localhost:5173`.

### Try it immediately

A small synthetic imbalanced dataset ships in `backend/sample_data/`. On the
**Dataset** page, click the sample dataset chip instead of uploading your
own file to try the full workflow in under a minute.

## 4. Core workflow (few clicks, as designed)

1. **Dataset** — upload a CSV or pick a sample. Row/feature counts, target
   column, class distribution, and imbalance ratio are detected automatically.
2. **New Experiment** — select models (search, filter by ML/DL, "Select All",
   or the Quick Analysis preset), pick an imbalance technique, optionally
   expand Advanced Configuration, then **Run Analysis**.
3. **Dashboard** — KPI cards, model comparison charts (switch metric on the
   fly), confusion matrices, ROC/PR curves, ML vs DL, before/after imbalance
   impact.
4. **Compare** — pick any subset of evaluated models for a focused
   side-by-side table + radar chart.
5. **Model detail** — click any model name anywhere to drill into its full
   metric set, confusion matrix, curves, feature importance, and (for DL)
   training loss curves.
6. **Insights / Conclusion** — every statement is generated from the actual
   numbers in this run.
7. **History** — revisit, re-run, delete, or export any past experiment.
8. **Reports** — export as PDF, Excel, CSV, or JSON.

## 5. Research-methodology guarantees

- The train/test split happens **once**, before any resampling.
- SMOTE/ADASYN/etc. are re-applied fresh to each model's **training fold
  only** — the test fold never sees a synthetic sample.
- A fixed `random_state` is used throughout for reproducibility.
- Preprocessing (imputation, scaling, encoding) is *fit* on the training
  fold and only *transformed* on the test fold.
- Insights and conclusions are template-free: they read numbers out of the
  actual `results` list at generation time (see `app/ml/insights.py`).

## 6. Extending the platform

- **Add a model**: add one entry to `ML_MODELS`/`DL_MODELS` in
  `backend/app/ml/registry.py`. It appears in the frontend's model picker
  automatically — no frontend changes needed.
- **Add an imbalance technique**: add an entry to `IMBALANCE_METHODS` in the
  same file, and a branch in `backend/app/ml/imbalance.py`.
- **Add a metric**: extend `compute_metrics()` in `backend/app/ml/metrics.py`
  and add it to `/api/metrics` in `backend/app/routers/models.py`.
- **Add a dataset**: drop a CSV into `backend/sample_data/` — it appears in
  the sample-dataset picker automatically.

## 7. Known limitations (also surfaced in the generated Conclusion)

- Each experiment run uses a single train/test split by default (the
  `cv_folds` config field is reserved for a future cross-validation mode).
- DL models train synchronously within the HTTP request for simplicity;
  for large datasets or many DL models, consider moving `run_experiment()`
  to a background task/queue (e.g. Celery, RQ, or FastAPI `BackgroundTasks`
  with a job-status endpoint).
- Hyperparameters use sensible library defaults rather than per-dataset
  tuning — Advanced Analysis exposes the parameters most worth tuning
  first (epochs, batch size, test size, random state).
