from __future__ import annotations

import re
from pathlib import Path

MODEL_ROOT = Path("data/models")
REPORT_ROOT = Path("data/reports")


class ModelPathError(ValueError):
    """Raised when a model namespace is unsafe or invalid."""


def normalize_model_name(model_name: str) -> str:
    value = (model_name or "").strip().lower()
    if not value:
        raise ModelPathError("model_name cannot be empty.")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ModelPathError(
            "model_name may contain only lowercase letters, numbers, "
            "underscores, and hyphens."
        )
    return value


def model_directory(model_name: str) -> Path:
    return MODEL_ROOT / normalize_model_name(model_name)


def report_directory(model_name: str) -> Path:
    return REPORT_ROOT / normalize_model_name(model_name)


def training_model_path(model_name: str) -> str:
    return str(model_directory(model_name) / "best_model.joblib")


def optimized_model_path(model_name: str) -> str:
    return str(model_directory(model_name) / "optimized_model.joblib")


def preferred_model_path(model_name: str) -> str:
    optimized = Path(optimized_model_path(model_name))
    if optimized.exists():
        return str(optimized)
    return training_model_path(model_name)


def training_report_path(model_name: str) -> str:
    return str(report_directory(model_name) / "model_comparison.json")


def optimization_report_path(model_name: str) -> str:
    return str(report_directory(model_name) / "model_optimization.json")
