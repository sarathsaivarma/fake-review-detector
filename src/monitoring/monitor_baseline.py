"""
Offline-runnable drift check: compares simple text-derived feature
distributions (word count, char count, exclamation ratio, upper-case ratio)
between the training reference set and recent production traffic using a
two-sample Kolmogorov-Smirnov test per feature. Stands in for
src/monitoring/monitor.py (Evidently AI) when that package isn't available.

Exits with code 1 if the share of significantly-drifted features crosses
the configured threshold, same contract as the real monitor.

Usage:
    python src/monitoring/monitor_baseline.py \
        --reference data/processed/test.csv \
        --current data/live/predictions.csv \
        --params params.yaml
"""
import argparse
import os
import sys

import pandas as pd
from scipy.stats import ks_2samp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import load_params  # noqa: E402


def build_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)
    out["word_count"] = out["text"].str.split().str.len()
    out["char_count"] = out["text"].str.len()
    out["exclamation_ratio"] = out["text"].str.count("!") / out["char_count"].clip(lower=1)
    out["upper_ratio"] = out["text"].str.count(r"[A-Z]") / out["char_count"].clip(lower=1)
    return out.drop(columns=["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="p-value threshold below which a feature is 'drifted'",
    )
    args = parser.parse_args()

    params = load_params(args.params)
    text_col = params["data"]["text_column"]
    mon_p = params["monitoring"]

    ref_df = pd.read_csv(args.reference)
    cur_df = pd.read_csv(args.current)
    if text_col not in cur_df.columns and "review_text" in cur_df.columns:
        cur_df = cur_df.rename(columns={"review_text": text_col})

    ref_feats = build_features(ref_df, text_col)
    cur_feats = build_features(cur_df, text_col)

    drifted = []
    print(f"{'feature':<20} {'ks_stat':>10} {'p_value':>10} {'drifted':>10}")
    for col in ref_feats.columns:
        stat, p_value = ks_2samp(ref_feats[col], cur_feats[col])
        is_drifted = p_value < args.alpha
        drifted.append(is_drifted)
        print(f"{col:<20} {stat:>10.4f} {p_value:>10.4f} {str(is_drifted):>10}")

    drift_share = sum(drifted) / len(drifted)
    print(f"\nShare of drifted features: {drift_share:.2%} "
          f"(threshold: {mon_p['drift_share_threshold']:.2%})")

    if drift_share >= mon_p["drift_share_threshold"]:
        print("DRIFT ALERT: exceeds threshold. Flagging for retrain.")
        sys.exit(1)

    print("No significant drift detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
