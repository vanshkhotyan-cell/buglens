from __future__ import annotations
import hashlib, subprocess
from pathlib import Path
from typing import Any
import dagshub, mlflow, mlflow.data, pandas as pd, yaml


def init_tracking(experiment_name: str = "buglens") -> None:
    dagshub.init(repo_owner="vanshkhotyan-cell", repo_name="buglens", mlflow=True)
    mlflow.set_experiment(experiment_name)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_dvc_lock_hash(lock_path: str = "dvc.lock") -> str:
    path = Path(lock_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "untracked"


def log_lineage_and_params(params_path: str = "params.yaml") -> dict[str, Any]:
    mlflow.log_param("git_commit", get_git_commit())
    mlflow.log_param("dvc_lock_hash", get_dvc_lock_hash())
    path = Path(params_path)
    if not path.exists():
        return {}
    params = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section, values in params.items():
        if isinstance(values, dict):
            for key, val in values.items():
                mlflow.log_param(f"{section}.{key}", val)
    return params


def log_training_dataset(df: pd.DataFrame, source: str, name: str = "buglens-train") -> None:
    dataset = mlflow.data.from_pandas(df, source=source, name=name)
    mlflow.log_input(dataset, context="training")
