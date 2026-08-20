"""
Optional bootstrapping step: if you don't have verified fraud labels, this
generates weak labels from well-known fake-review signals so you can get a
first model trained, then refine with human-in-the-loop review.

Signals used (each contributes to a suspicion score):
    - near-duplicate review text across different products/reviewers
    - reviewer posting many reviews in a very short time window ("burst")
    - review posted without a verified purchase
    - extreme rating (1 or 5) combined with very short, generic text
    - reviewer with an unusually high fraction of extreme ratings overall

This is a heuristic, not ground truth -- treat it as a starting point and
replace with real labels (audited fraud reports, platform takedown data,
or human-reviewed samples) as soon as they're available.
"""
import argparse

import pandas as pd


def compute_suspicion_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)

    dup_counts = df["review_text"].str.lower().str.strip().value_counts()
    is_dup = df["review_text"].str.lower().str.strip().map(dup_counts) > 1
    score += is_dup.astype(float) * 0.35

    df["_ts"] = pd.to_datetime(df["review_date"], errors="coerce")
    burst = (
        df.sort_values("_ts")
        .groupby("reviewer_id")["_ts"]
        .diff()
        .dt.total_seconds()
        .fillna(1e9)
    )
    score += (burst < 60).astype(float) * 0.25

    if "verified_purchase" in df.columns:
        score += (~df["verified_purchase"].astype(bool)).astype(float) * 0.15

    extreme = df["rating"].isin([1, 5])
    short_generic = df["review_text"].str.split().str.len() <= 6
    score += (extreme & short_generic).astype(float) * 0.15

    reviewer_extreme_rate = df.groupby("reviewer_id")["rating"].transform(
        lambda r: r.isin([1, 5]).mean()
    )
    score += (reviewer_extreme_rate > 0.9).astype(float) * 0.10

    return score.clip(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["fraud_score"] = compute_suspicion_score(df)
    df["is_fake"] = (df["fraud_score"] >= args.threshold).astype(int)
    df.drop(columns=["_ts"], errors="ignore").to_csv(args.output, index=False)
    print(f"Weak-labeled {len(df)} rows. Fake rate: {df['is_fake'].mean():.2%}")
    print("NOTE: sample and manually verify a subset before trusting these labels.")


if __name__ == "__main__":
    main()
