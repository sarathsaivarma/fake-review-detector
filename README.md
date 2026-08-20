# Fake Review Detection System

An end-to-end MLOps pipeline for detecting fraudulent/fake product reviews using
a fine-tuned BERT classifier, with full lifecycle support: data versioning,
experiment tracking, containerized serving, drift monitoring, and automated
retraining.

## Architecture

```
                       ┌─────────────────┐
                       │  Raw Review Data │
                       └────────┬─────────┘
                                │  DVC-tracked
                                ▼
                       ┌─────────────────┐
                       │  Data Prep /     │
                       │  Feature Pipeline│
                       └────────┬─────────┘
                                │
                                ▼
                  ┌─────────────────────────┐
                  │  BERT Fine-tuning        │
                  │  (HuggingFace Transformers)│
                  │  tracked via MLflow       │
                  └────────┬─────────────────┘
                            │  best model → MLflow Model Registry
                            ▼
                  ┌─────────────────────────┐
                  │  FastAPI Prediction API  │
                  │  (Dockerized)            │
                  └────────┬─────────────────┘
                            │  live traffic + predictions logged
                            ▼
                  ┌─────────────────────────┐
                  │  Evidently AI Monitoring  │
                  │  (drift + performance)    │
                  └────────┬─────────────────┘
                            │  drift/degradation alert
                            ▼
                  ┌─────────────────────────┐
                  │  GitHub Actions           │
                  │  retrain → validate →     │
                  │  register → deploy        │
                  └─────────────────────────┘
```

## Repo layout

```
fake-review-detector/
├── data/
│   ├── raw/                # DVC-tracked raw review dumps
│   └── processed/          # DVC-tracked cleaned/split datasets
├── src/
│   ├── data/
│   │   └── prepare_data.py     # cleaning, labeling, train/val/test split
│   ├── train/
│   │   ├── train.py            # BERT fine-tuning + MLflow logging
│   │   └── evaluate.py         # held-out eval + registry gate
│   ├── serve/
│   │   └── app.py               # FastAPI inference service
│   ├── monitoring/
│   │   └── monitor.py           # Evidently drift/performance job
│   └── utils/
│       └── config.py
├── tests/
│   └── test_api.py
├── .github/workflows/
│   ├── ci-cd.yml            # lint/test/build/deploy on push
│   └── retrain.yml          # scheduled + drift-triggered retraining
├── dvc.yaml                 # DVC pipeline (prepare → train → evaluate)
├── params.yaml               # single source of truth for hyperparameters
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Data versioning
dvc init
dvc remote add -d storage s3://your-bucket/fake-review-dvc   # or gcs/azure/local
dvc add data/raw/reviews.csv
git add data/raw/reviews.csv.dvc .gitignore
git commit -m "Track raw review dataset with DVC"
dvc push

# 3. Run the full DVC pipeline (prepare -> train -> evaluate)
dvc repro

# 4. Inspect experiments
mlflow ui   # http://localhost:5000

# 5. Serve locally
docker compose up --build
# API docs at http://localhost:8000/docs

# 6. Run a monitoring pass against fresh production data
python src/monitoring/monitor.py --reference data/processed/test.csv --current data/live/recent_predictions.csv
```

## Design notes

- **Model**: `bert-base-uncased` fine-tuned as a binary sequence classifier
  (genuine=0 / fake=1). Swappable for `distilbert` for lower-latency serving.
- **Dataset versioning**: DVC tracks raw + processed data and pipeline stages
  (`dvc.yaml`), so every training run is reproducible from a specific data
  version + commit.
- **Experiment tracking**: every run logs params, metrics (F1, precision,
  recall, ROC-AUC), and the model artifact to MLflow. Promotion to
  `Production` in the MLflow Model Registry is a manual or gated step
  (`evaluate.py` only registers if metrics clear a threshold).
- **Serving**: FastAPI loads the current `Production`-stage model from the
  MLflow registry at startup; `/predict` returns a fraud probability + label.
- **Monitoring**: Evidently AI computes data drift (text/statistical feature
  drift) and, once ground-truth labels trickle in (e.g. from user reports or
  audits), classification performance drift. Reports are generated on a
  schedule and on-demand.
- **Retraining loop**: `retrain.yml` runs on a schedule and can also be
  triggered by a monitoring job that detects drift beyond a threshold. It
  reruns `dvc repro`, and only promotes the new model if it beats the current
  `Production` model on the held-out set.
