"""Shared helpers for loading params.yaml across pipeline stages."""
import argparse
import os
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        params = yaml.safe_load(f)

    # Allow environment variables to override select params without editing
    # the file -- useful for Colab (local sqlite tracking) vs. a real
    # deployment (a hosted MLflow server) without maintaining two configs.
    if os.getenv("MLFLOW_TRACKING_URI"):
        params.setdefault("mlflow", {})["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

    return params


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    return parser
