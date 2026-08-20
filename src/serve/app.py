"""
FastAPI service that loads the current Production model from the MLflow
Model Registry and exposes a /predict endpoint. Also logs every request +
prediction to a local log (data/live/predictions.csv) so the monitoring job
has current production data to compare against the training reference set.
"""
import csv
import os
import time
from datetime import datetime, timezone
from typing import List

import mlflow
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

with open(os.getenv("PARAMS_PATH", "params.yaml")) as f:
    PARAMS = yaml.safe_load(f)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", PARAMS["mlflow"]["tracking_uri"])
MODEL_NAME = PARAMS["registry"]["model_name"]
MAX_LENGTH = PARAMS["model"]["max_length"]
LOG_PATH = "data/live/predictions.csv"

app = FastAPI(
    title="Fake Review Detection API",
    description="Classifies product reviews as genuine or fraudulent.",
    version="1.0.0",
)

_model = None
_tokenizer = None
_model_version = None


class ReviewRequest(BaseModel):
    review_text: str = Field(..., min_length=1, description="The review text to classify")
    review_id: str | None = Field(None, description="Optional client-side ID for tracing")


class BatchReviewRequest(BaseModel):
    reviews: List[ReviewRequest]


class PredictionResponse(BaseModel):
    review_id: str | None
    is_fake: bool
    fraud_probability: float
    model_version: str


@app.on_event("startup")
def load_model():
    global _model, _tokenizer, _model_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    uri = f"models:/{MODEL_NAME}@production"
    loaded = mlflow.transformers.load_model(uri)
    _model, _tokenizer = loaded.model, loaded.tokenizer
    _model.eval()

    client = mlflow.tracking.MlflowClient()
    version_info = client.get_model_version_by_alias(MODEL_NAME, "production")
    _model_version = version_info.version

    os.makedirs("data/live", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "review_id", "review_text", "fraud_probability",
                 "is_fake", "model_version", "latency_ms"]
            )


def _predict_one(text: str):
    start = time.time()
    with torch.no_grad():
        enc = _tokenizer(
            text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        logits = _model(**enc).logits
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    latency_ms = (time.time() - start) * 1000
    return prob, latency_ms


def _log_prediction(review_id, text, prob, is_fake, latency_ms):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(), review_id, text,
            prob, is_fake, _model_version, round(latency_ms, 2),
        ])


@app.get("/health")
def health():
    return {"status": "ok", "model_version": _model_version}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: ReviewRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prob, latency_ms = _predict_one(req.review_text)
    is_fake = prob >= 0.5
    _log_prediction(req.review_id, req.review_text, prob, is_fake, latency_ms)
    return PredictionResponse(
        review_id=req.review_id, is_fake=is_fake,
        fraud_probability=round(prob, 4), model_version=str(_model_version),
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(req: BatchReviewRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    results = []
    for item in req.reviews:
        prob, latency_ms = _predict_one(item.review_text)
        is_fake = prob >= 0.5
        _log_prediction(item.review_id, item.review_text, prob, is_fake, latency_ms)
        results.append(PredictionResponse(
            review_id=item.review_id, is_fake=is_fake,
            fraud_probability=round(prob, 4), model_version=str(_model_version),
        ))
    return results
