"""
Fine-tunes a BERT sequence classifier on labeled reviews and logs the run
(params, metrics, model artifact) to MLflow.

Usage:
    python src/train/train.py --params params.yaml
"""
import inspect
import os
import sys

import mlflow
import mlflow.transformers
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import load_params, base_arg_parser


def load_split(path, text_col, label_col, tokenizer, max_length):
    df = pd.read_csv(path)
    ds = Dataset.from_pandas(df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    ))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return ds.map(tokenize, batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": auc}


def build_training_args(train_p):
    """Build TrainingArguments defensively so this works across transformers
    versions that have renamed/removed fields over time (e.g. warmup_ratio,
    eval_strategy vs evaluation_strategy)."""
    candidate_kwargs = {
        "output_dir": "outputs/checkpoints",
        "num_train_epochs": train_p["epochs"],
        "per_device_train_batch_size": train_p["batch_size"],
        "per_device_eval_batch_size": train_p["batch_size"],
        "learning_rate": float(train_p["learning_rate"]),
        "weight_decay": train_p["weight_decay"],
        "warmup_ratio": train_p["warmup_ratio"],
        "eval_strategy": train_p["eval_strategy"],
        "evaluation_strategy": train_p["eval_strategy"],
        "save_strategy": train_p["eval_strategy"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "logging_steps": 50,
        "report_to": [],
        "seed": train_p["seed"],
    }
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    filtered = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    dropped = set(candidate_kwargs) - set(filtered)
    if dropped:
        print(f"NOTE: dropping unsupported TrainingArguments kwargs: {dropped}")
    return TrainingArguments(**filtered)


def main():
    parser = base_arg_parser("Fine-tune BERT fake review classifier")
    args = parser.parse_args()
    all_params = load_params(args.params)
    data_p, model_p, train_p, mlflow_p = (
        all_params["data"], all_params["model"], all_params["train"], all_params["mlflow"]
    )

    torch.manual_seed(train_p["seed"])

    mlflow.set_tracking_uri(mlflow_p["tracking_uri"])
    mlflow.set_experiment(mlflow_p["experiment_name"])

    tokenizer = AutoTokenizer.from_pretrained(model_p["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_p["base_model"], num_labels=model_p["num_labels"]
    )

    train_ds = load_split(
        f"{data_p['processed_dir']}/train.csv", data_p["text_column"],
        data_p["label_column"], tokenizer, model_p["max_length"],
    )
    val_ds = load_split(
        f"{data_p['processed_dir']}/val.csv", data_p["text_column"],
        data_p["label_column"], tokenizer, model_p["max_length"],
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args(train_p)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_p["early_stopping_patience"])],
    )

    with mlflow.start_run() as run:
        mlflow.log_params({
            "base_model": model_p["base_model"],
            "max_length": model_p["max_length"],
            **{f"train.{k}": v for k, v in train_p.items()},
        })

        trainer.train()
        eval_metrics = trainer.evaluate()
        mlflow.log_metrics({k: v for k, v in eval_metrics.items() if isinstance(v, (int, float))})

        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer": tokenizer},
            artifact_path="model",
            task="text-classification",
        )

        os.makedirs("models", exist_ok=True)
        with open("models/latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Run {run.info.run_id} complete. Val F1: {eval_metrics.get('eval_f1'):.4f}")


if __name__ == "__main__":
    main()
