# Results — What This Project Demonstrates

This document summarizes what was built and actually verified running, for anyone
evaluating this as a portfolio piece.

## What it is

An MLOps pipeline for detecting fraudulent product reviews:
- **Model**: BERT (`bert-base-uncased`) fine-tuned as a binary classifier
- **Data versioning**: DVC
- **Experiment tracking**: MLflow (params, metrics, model registry with staged promotion)
- **Serving**: FastAPI
- **Monitoring**: Evidently AI + statistical drift testing (KS-test)
- **CI/CD**: GitHub Actions (lint/test/build/deploy + scheduled/drift-triggered retraining)

## What was verified end-to-end (Google Colab, CPU runtime)

Every stage below was actually executed, not just written — see `fake_review_detector_demo.ipynb`
for the exact runnable sequence.

1. **Data preparation** — cleaned and stratified-split a labeled review dataset
   (15-row demo set included; schema documented below for swapping in real data).
2. **Model training** — fine-tuned BERT for 3 epochs, with training loss decreasing
   run over run; the run was logged to MLflow (params, per-epoch metrics, model artifact).
3. **Evaluation & promotion gate** — scored the trained model against a held-out
   test set and only promoted it to the MLflow Model Registry's `production` alias
   because it cleared the configured minimum F1 threshold. The gate logic (in
   `src/train/evaluate.py`) also checks a candidate model beats the *current*
   production model by a configurable margin before replacing it — this is the
   mechanism that prevents a bad retrain from silently degrading a live system.
4. **Serving** — the FastAPI app loaded the production-aliased model from the
   MLflow registry and correctly classified new review text: reviews using
   generic superlative language ("BEST PRODUCT EVER BUY NOW!!!") scored as
   likely-fake, while specific, nuanced reviews scored as likely-genuine.
5. **Monitoring** — the drift-detection job compared live prediction traffic
   against the training distribution using per-feature KS-tests, and generated
   an Evidently AI HTML report, both of which need no server or paid service —
   fully self-contained.

## Known limitations, stated plainly

- **The demo dataset is 15 rows.** Every metric shown (F1, precision, recall, AUC)
  is a perfect 1.0 — that's an artifact of a trivially separable toy dataset, not a
  claim about real-world performance. Swapping in a real labeled dataset with the
  same column schema (`review_text`, `is_fake`, etc. — see `params.yaml`) is a
  drop-in change; the pipeline logic doesn't need to be rewritten.
- **DVC remote storage, a persistent MLflow server, and the Docker container**
  are implemented and documented but not exercised in the notebook — those need
  real infrastructure (cloud storage, a running server, a container host) rather
  than a single Colab session.
- **GitHub Actions workflows** (`.github/workflows/`) are written and would work
  once pushed to an actual GitHub repo with the relevant secrets configured, but
  weren't run here since that requires a live repo.
- **Library version sensitivity**: this pipeline was built against specific
  versions of `mlflow`, `transformers`, and `evidently`; each has changed its API
  in newer releases (MLflow deprecated plain file-store tracking in favor of a
  database backend; `transformers.TrainingArguments` renamed some fields;
  Evidently rewrote its Report/Dataset API in 0.7.x). The code now handles this
  defensively — `train.py` filters `TrainingArguments` kwargs to whatever the
  installed version accepts, and `monitor.py` falls back gracefully if
  Evidently's HTML report generation fails — but it's worth knowing going in that
  ML tooling moves fast and pinning versions (`requirements.txt` already does
  this) is what keeps a given environment reproducible.

## How to reproduce this

Open `fake_review_detector_demo.ipynb` in Google Colab, upload
`fake-review-detector.zip` when prompted, and run every cell top to bottom.
No manual fixes should be needed — every compatibility issue encountered during
development is already patched into the shipped code.
