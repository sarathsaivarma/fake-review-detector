"""
Minimal local experiment tracker used as an offline stand-in for MLflow.
Appends one JSON record per run to runs/tracking.jsonl and stores model
artifacts under runs/<run_id>/. Same idea as MLflow (params + metrics +
artifact per run, queryable history) without needing a tracking server.
"""
import json
import os
import uuid
from datetime import datetime, timezone

RUNS_DIR = "runs"
LOG_PATH = os.path.join(RUNS_DIR, "tracking.jsonl")


def start_run():
    run_id = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(RUNS_DIR, run_id), exist_ok=True)
    return run_id


def log_run(run_id: str, params: dict, metrics: dict, artifact_path: str):
    os.makedirs(RUNS_DIR, exist_ok=True)
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": metrics,
        "artifact_path": artifact_path,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def all_runs():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_production_run():
    """Returns the run currently pointed to by runs/production_pointer.json, or None."""
    pointer_path = os.path.join(RUNS_DIR, "production_pointer.json")
    if not os.path.exists(pointer_path):
        return None
    with open(pointer_path) as f:
        return json.load(f)


def set_production_run(run_id: str, metrics: dict, artifact_path: str):
    os.makedirs(RUNS_DIR, exist_ok=True)
    pointer_path = os.path.join(RUNS_DIR, "production_pointer.json")
    with open(pointer_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "artifact_path": artifact_path,
        }, f, indent=2)
