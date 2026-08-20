"""
Evaluates the latest baseline run on the held-out test set and promotes it
to "production" (a pointer file the serving app reads) only if it clears
the minimum F1 threshold and beats the current production model by the
configured margin. Mirrors src/train/evaluate.py's promotion gate logic.

Usage:
    python src/train/evaluate_baseline.py --params params.yaml
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import load_params, base_arg_parser
from src.train import local_tracker


def main():
    parser = base_arg_parser("Evaluate + conditionally promote baseline model")
    args = parser.parse_args()
    all_params = load_params(args.params)
    data_p, reg_p = all_params["data"], all_params["registry"]

    with open("models/latest_run_id.txt") as f:
        run_id = f.read().strip()

    model_path = os.path.join("runs", run_id, "model.joblib")
    pipeline = joblib.load(model_path)

    test_df = pd.read_csv(f"{data_p['processed_dir']}/test.csv")
    text_col, label_col = data_p["text_column"], data_p["label_column"]

    preds = pipeline.predict(test_df[text_col])
    probs = pipeline.predict_proba(test_df[text_col])[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df[label_col], preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(test_df[label_col], probs)
    except ValueError:
        auc = float("nan")

    metrics = {"test_precision": precision, "test_recall": recall,
               "test_f1": f1, "test_roc_auc": auc}

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Test metrics:", json.dumps(metrics, indent=2))

    candidate_f1 = metrics["test_f1"]

    if candidate_f1 < reg_p["min_f1_threshold"]:
        print(f"Candidate F1 {candidate_f1:.4f} below minimum threshold "
              f"{reg_p['min_f1_threshold']}. Not promoting.")
        return

    prod = local_tracker.get_production_run()
    current_prod_f1 = prod["metrics"]["test_f1"] if prod else None

    if current_prod_f1 is None or candidate_f1 >= current_prod_f1 + reg_p["promotion_margin"]:
        local_tracker.set_production_run(run_id, metrics, model_path)
        prior = f"{current_prod_f1:.4f}" if current_prod_f1 is not None else "none"
        print(f"Promoted run {run_id} to production (F1 {candidate_f1:.4f} vs prior {prior}).")
    else:
        print(f"Run {run_id} registered but NOT promoted "
              f"(F1 {candidate_f1:.4f} did not beat production {current_prod_f1:.4f} "
              f"by margin {reg_p['promotion_margin']}).")


if __name__ == "__main__":
    main()
