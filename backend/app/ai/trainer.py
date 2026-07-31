from __future__ import annotations

from typing import Any

from backend.app.ai.model_paths import (
    normalize_model_name,
    training_model_path,
    training_report_path,
)

from backend.app.services.model_training import (
    ModelTrainingError,
    train_tcga_brca_model_comparison,
)


DEFAULT_AI_DATASET_PATH = (
    "data/processed/"
    "tcga_brca_cohort_100_patients_ml_ready.csv"
)
DEFAULT_AI_MODEL_PATH = (
    "data/models/trojan_horse_best_model_v1.joblib"
)
DEFAULT_AI_REPORT_PATH = (
    "data/reports/trojan_horse_model_comparison_v1.json"
)


class AITrainingError(RuntimeError):
    """Raised when the AI training workflow cannot complete."""


def train_ai_models(
    dataset_path: str = DEFAULT_AI_DATASET_PATH,
    best_model_path: str = DEFAULT_AI_MODEL_PATH,
    report_path: str = DEFAULT_AI_REPORT_PATH,
    random_state: int = 42,
    model_name: str = "brca",
) -> dict[str, Any]:
    """
    Train, compare, rank, and save Project HERMES models.

    The underlying training service evaluates logistic regression,
    random forest, and gradient boosting with identical stratified
    cross-validation folds. The best model is selected primarily by
    balanced accuracy.
    """

    namespace = normalize_model_name(model_name)
    if best_model_path == DEFAULT_AI_MODEL_PATH:
        best_model_path = training_model_path(namespace)
    if report_path == DEFAULT_AI_REPORT_PATH:
        report_path = training_report_path(namespace)

    try:
        result = train_tcga_brca_model_comparison(
            dataset_path=dataset_path,
            best_model_path=best_model_path,
            report_path=report_path,
            random_state=random_state,
        )
    except (ModelTrainingError, ValueError, OSError) as exc:
        raise AITrainingError(str(exc)) from exc

    return {
        "ai_training_status": "complete",
        "platform": "Project HERMES",
        "model_namespace": namespace,
        "task": "TCGA-BRCA vital-status classification",
        **result,
        "next_milestone": (
            "Build and test the AI prediction endpoint using the "
            "saved best-model package."
        ),
        "research_use_only": True,
    }
