"""
Prepares the raw review dataset for training:
- basic text cleaning
- drops empty / too-short reviews
- stratified train/val/test split
- writes processed CSVs that DVC tracks as pipeline outputs

Expected raw schema (data/raw/reviews.csv):
    review_text, rating, verified_purchase, reviewer_id, product_id, is_fake

`is_fake` is the binary label (1 = fraudulent/fake, 0 = genuine). If you're
bootstrapping without labels, see the heuristic-labeling note at the bottom.
"""
import os
import re
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import load_params, base_arg_parser


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"<.*?>", " ", text)                  # HTML tags
    text = re.sub(r"\s+", " ", text)                    # whitespace
    return text.strip()


def main():
    parser = base_arg_parser("Prepare review dataset")
    args = parser.parse_args()
    params = load_params(args.params)["data"]

    df = pd.read_csv(params["raw_path"])
    text_col, label_col = params["text_column"], params["label_column"]

    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found. This pipeline expects labeled data "
            "(from verified fraud reports, platform takedowns, or a labeled "
            "benchmark like the Yelp/Amazon fake review datasets). If you only "
            "have unlabeled data, use src/data/weak_labeling.py to bootstrap "
            "labels via heuristics (duplicate text, burst posting, "
            "reviewer/product graph anomalies) before running this stage."
        )

    df[text_col] = df[text_col].astype(str).map(clean_text)
    df = df[df[text_col].str.split().str.len() >= params["min_review_length"]]
    df = df.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col])

    train_df, temp_df = train_test_split(
        df,
        test_size=params["test_size"] + params["val_size"],
        stratify=df[label_col],
        random_state=params["random_state"],
    )
    relative_test = params["test_size"] / (params["test_size"] + params["val_size"])
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df[label_col],
        random_state=params["random_state"],
    )

    os.makedirs(params["processed_dir"], exist_ok=True)
    train_df.to_csv(f"{params['processed_dir']}/train.csv", index=False)
    val_df.to_csv(f"{params['processed_dir']}/val.csv", index=False)
    test_df.to_csv(f"{params['processed_dir']}/test.csv", index=False)

    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"class balance (train):\n{train_df[label_col].value_counts(normalize=True)}")


if __name__ == "__main__":
    main()
