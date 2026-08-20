"""
Evaluates the just-trained model (recorded in models/latest_run_id.txt) on
the held-out test set, and only promotes it to the MLflow Model Registry's
"Production" stage if it clears a minimum threshold AND beats the current
Production model by the configured margin. This is the gate that keeps
retraining from silently degrading the live system.
"""
import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.tracking import MlflowClient
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import load_params, base_arg_parser  # noqa: E402


def score_test_set(model, tokenizer, df, text_col, label_col, max_length):
    model.eval()
    all_probs, all_preds = [], []
    with torch.no_grad():
        for i in range(0, len(df), 32):
            batch_texts = df[text_col].iloc[i:i + 32].tolist()
            enc = tokenizer(batch_texts, truncation=True, padding=True,
                             max_length=max_length, return_tensors="pt")
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.extend(probs)
            all_preds.extend(np.argmax(logits.numpy(), axis=1))

    labels = df[label_col].values
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, all_preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(labels, all_probs)
    return {"test_precision": precision, "test_recall": recall,
            "test_f1": f1, "test_roc_auc": auc}


def main():
    parser = base_arg_parser("Evaluate + conditionally promote model")
    args = parser.parse_args()
    all_params = load_params(args.params)
    data_p, model_p = all_params["data"], all_params["model"]
    reg_p, mlflow_p = all_params["registry"], all_params["mlflow"]

    mlflow.set_tracking_uri(mlflow_p["tracking_uri"])
    client = MlflowClient()

    with open("models/latest_run_id.txt") as f:
        run_id = f.read().strip()

    loaded = mlflow.transformers.load_model(f"runs:/{run_id}/model")
    model, tokenizer = loaded.model, loaded.tokenizer

    test_df = pd.read_csv(f"{data_p['processed_dir']}/test.csv")
    metrics = score_test_set(
        model, tokenizer, test_df, data_p["text_column"],
        data_p["label_column"], model_p["max_length"],
    )

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Test metrics:", metrics)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    candidate_f1 = metrics["test_f1"]
    model_name = reg_p["model_name"]

    if candidate_f1 < reg_p["min_f1_threshold"]:
        print(f"Candidate F1 {candidate_f1:.4f} below minimum threshold "
              f"{reg_p['min_f1_threshold']}. Not registering.")
        return

    current_prod_f1 = None
    try:
        prod_version = client.get_model_version_by_alias(model_name, "production")
        prod_run = client.get_run(prod_version.run_id)
        current_prod_f1 = prod_run.data.metrics.get("test_f1")
    except Exception:
        print("No existing Production model found; treating this as first deploy.")

    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)

    if current_prod_f1 is None or candidate_f1 >= current_prod_f1 + reg_p["promotion_margin"]:
        client.set_registered_model_alias(model_name, "production", result.version)
        print(f"Promoted version {result.version} to Production "
              f"(F1 {candidate_f1:.4f} vs prior {current_prod_f1}).")
    else:
        client.set_registered_model_alias(model_name, "candidate", result.version)
        print(f"Registered version {result.version} as candidate only "
              f"(F1 {candidate_f1:.4f} did not beat Production {current_prod_f1:.4f} "
              f"by margin {reg_p['promotion_margin']}).")


if __name__ == "__main__":
    main()
