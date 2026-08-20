"""
Offline-runnable baseline model: TF-IDF + Logistic Regression.

Stands in for src/train/train.py (BERT fine-tuning + MLflow) in
environments without internet access / a GPU / MLflow server. Same pipeline
shape: reads processed data, trains, logs params+metrics+artifact via a
local run tracker so the rest of the MLOps flow (evaluate/gate/promote,
serve, monitor) works unmodified.

Usage:
    python src/train/train_baseline.py --params params.yaml
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import load_params, base_arg_parser
from src.train import local_tracker


def main():
    parser = base_arg_parser("Train TF-IDF + LogisticRegression baseline")
    args = parser.parse_args()
    all_params = load_params(args.params)
    data_p = all_params["data"]

    text_col, label_col = data_p["text_column"], data_p["label_column"]
    train_df = pd.read_csv(f"{data_p['processed_dir']}/train.csv")
    val_df = pd.read_csv(f"{data_p['processed_dir']}/val.csv")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=20000, sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=42
        )),
    ])

    pipeline.fit(train_df[text_col], train_df[label_col])

    val_preds = pipeline.predict(val_df[text_col])
    val_probs = pipeline.predict_proba(val_df[text_col])[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        val_df[label_col], val_preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(val_df[label_col], val_probs)
    except ValueError:
        auc = float("nan")

    metrics = {"val_precision": precision, "val_recall": recall,
               "val_f1": f1, "val_roc_auc": auc}
    params = {"model_type": "tfidf_logreg", "ngram_range": "(1,2)",
              "max_features": 20000, "C": 1.0}

    run_id = local_tracker.start_run()
    artifact_path = os.path.join("runs", run_id, "model.joblib")
    joblib.dump(pipeline, artifact_path)
    local_tracker.log_run(run_id, params, metrics, artifact_path)

    os.makedirs("models", exist_ok=True)
    with open("models/latest_run_id.txt", "w") as f:
        f.write(run_id)

    print(f"Run {run_id} complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
