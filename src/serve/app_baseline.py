"""
Offline-runnable serving app: loads whatever model is currently pointed to
by runs/production_pointer.json and exposes /health, /predict, /predict/batch.
Stands in for src/serve/app.py (FastAPI + MLflow registry) when there's no
MLflow server available. Logs every prediction to data/live/predictions.csv,
same as the real app, so monitoring has something to compare against.

Usage:
    python src/serve/app_baseline.py --params params.yaml --port 8000
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import joblib
from flask import Flask, jsonify, request

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import load_params
from src.train import local_tracker

app = Flask(__name__)
STATE = {"pipeline": None, "run_id": None}
LOG_PATH = "data/live/predictions.csv"


def load_production_model():
    prod = local_tracker.get_production_run()
    if prod is None:
        raise RuntimeError(
            "No production model found. Run train_baseline.py then "
            "evaluate_baseline.py first."
        )
    STATE["pipeline"] = joblib.load(prod["artifact_path"])
    STATE["run_id"] = prod["run_id"]

    os.makedirs("data/live", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "review_id", "review_text", "fraud_probability",
                 "is_fake", "model_version", "latency_ms"]
            )


def _log_prediction(review_id, text, prob, is_fake, latency_ms):
    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(), review_id, text,
            prob, is_fake, STATE["run_id"], round(latency_ms, 2),
        ])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_version": STATE["run_id"]})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    text = body.get("review_text", "")
    review_id = body.get("review_id")
    if not text:
        return jsonify({"error": "review_text is required"}), 422

    start = time.time()
    prob = float(STATE["pipeline"].predict_proba([text])[0, 1])
    latency_ms = (time.time() - start) * 1000
    is_fake = prob >= 0.5

    _log_prediction(review_id, text, prob, is_fake, latency_ms)
    return jsonify({
        "review_id": review_id,
        "is_fake": is_fake,
        "fraud_probability": round(prob, 4),
        "model_version": STATE["run_id"],
    })


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    body = request.get_json(force=True)
    reviews = body.get("reviews", [])
    results = []
    for item in reviews:
        text = item.get("review_text", "")
        review_id = item.get("review_id")
        start = time.time()
        prob = float(STATE["pipeline"].predict_proba([text])[0, 1])
        latency_ms = (time.time() - start) * 1000
        is_fake = prob >= 0.5
        _log_prediction(review_id, text, prob, is_fake, latency_ms)
        results.append({
            "review_id": review_id, "is_fake": is_fake,
            "fraud_probability": round(prob, 4), "model_version": STATE["run_id"],
        })
    return jsonify(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    load_params(args.params)  # validated for consistency; not otherwise needed here
    load_production_model()
    app.run(host="0.0.0.0", port=args.port)
