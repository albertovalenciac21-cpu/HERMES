from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


MODEL_DIRECTORY = Path("data/models")
REPORT_DIRECTORY = Path("data/reports")


class ModelRegistryError(RuntimeError):
    """Raised when model registry information cannot be read."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(mode="r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelRegistryError(
            f"Could not read report '{path}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ModelRegistryError(
            f"Report '{path}' does not contain a JSON object."
        )

    return payload


def list_registered_models() -> dict[str, Any]:
    """List saved joblib model packages and their basic metadata."""

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    models: list[dict[str, Any]] = []

    for path in sorted(MODEL_DIRECTORY.rglob("*.joblib")):
        entry: dict[str, Any] = {
            "file_name": path.name,
            "model_namespace": (path.parent.name if path.parent != MODEL_DIRECTORY else "legacy"),
            "model_path": str(path),
            "size_bytes": path.stat().st_size,
            "load_status": "not_inspected",
        }

        try:
            package = joblib.load(path)

            if isinstance(package, dict):
                entry.update(
                    {
                        "load_status": "loaded",
                        "model_name": package.get("model_name"),
                        "model_type": package.get("model_type"),
                        "feature_count": len(
                            package.get("feature_names", [])
                        ),
                        "target_column": package.get(
                            "target_column"
                        ),
                        "source_dataset_path": package.get(
                            "source_dataset_path"
                        ),
                        "clinically_validated": package.get(
                            "clinically_validated",
                            False,
                        ),
                        "research_use_only": package.get(
                            "research_use_only",
                            True,
                        ),
                    }
                )
            else:
                entry["load_status"] = "unsupported_package"

        except Exception as exc:
            entry.update(
                {
                    "load_status": "load_failed",
                    "error": str(exc),
                }
            )

        models.append(entry)

    return {
        "registered_model_count": len(models),
        "model_directory": str(MODEL_DIRECTORY),
        "models": models,
        "research_use_only": True,
    }


def list_training_reports() -> dict[str, Any]:
    """List saved model reports with key performance information."""

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []

    for path in sorted(REPORT_DIRECTORY.rglob("*.json")):
        entry: dict[str, Any] = {
            "file_name": path.name,
            "model_namespace": (path.parent.name if path.parent != REPORT_DIRECTORY else "legacy"),
            "report_path": str(path),
            "size_bytes": path.stat().st_size,
        }

        try:
            payload = _load_json(path)
            entry.update(
                {
                    "read_status": "loaded",
                    "project": payload.get("project"),
                    "training_status": payload.get(
                        "training_status"
                    ),
                    "best_model": payload.get("best_model"),
                    "cross_validated_metrics": payload.get(
                        "cross_validated_metrics"
                    ),
                    "dataset_summary": payload.get(
                        "dataset_summary"
                    ),
                    "research_use_only": payload.get(
                        "research_use_only",
                        True,
                    ),
                }
            )
        except ModelRegistryError as exc:
            entry.update(
                {
                    "read_status": "read_failed",
                    "error": str(exc),
                }
            )

        reports.append(entry)

    return {
        "training_report_count": len(reports),
        "report_directory": str(REPORT_DIRECTORY),
        "reports": reports,
        "research_use_only": True,
    }
