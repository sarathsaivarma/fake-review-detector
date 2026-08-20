"""
Runs data drift checks comparing the training reference distribution
against recent production traffic.

Uses Evidently AI (0.7.x API: Dataset/DataDefinition/Report/DataDriftPreset)
to generate a human-readable HTML drift report, and a scipy two-sample
Kolmogorov-Smirnov test per feature to make the actual pass/fail decision.
Evidently's internal result schema has changed significantly across
versions (0.4 -> 0.5 -> 0.7), so the KS-test gate keeps the CI/CD contract
(exit code 0/1) stable regardless of which Evidently version is installed;
the HTML report is still generated for human review.

Usage:
    python src/monitoring/monitor.py \
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


def generate_evidently_report(
    ref_feats: pd.DataFrame, cur_feats: pd.DataFrame, output_html: str
) -> bool:
    """Best-effort HTML report via Evidently. Returns True if it succeeded.
    Failures here don't block the pipeline -- the KS-test gate below is
    the source of truth for the pass/fail decision."""
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset

        numerical_cols = list(ref_feats.columns)
        data_definition = DataDefinition(numerical_columns=numerical_cols)

        ref_dataset = Dataset.from_pandas(ref_feats, data_definition=data_definition)
        cur_dataset = Dataset.from_pandas(cur_feats, data_definition=data_definition)

        report = Report(metrics=[DataDriftPreset()])
        my_eval = report.run(reference_data=ref_dataset, current_data=cur_dataset)

        os.makedirs(os.path.dirname(output_html), exist_ok=True)
        my_eval.save_html(output_html)
        print(f"Evidently HTML report saved to {output_html}")
        return True
    except Exception as e:
        print(f"NOTE: Evidently report generation skipped ({type(e).__name__}: {e}). "
              f"Continuing with KS-test drift gate only.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Training/test reference data CSV")
    parser.add_argument("--current", required=True, help="Recent production predictions CSV")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--output-html", default="reports/drift_report.html")
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

    generate_evidently_report(ref_feats, cur_feats, args.output_html)

    drifted = []
    print(f"\n{'feature':<20} {'ks_stat':>10} {'p_value':>10} {'drifted':>10}")
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
