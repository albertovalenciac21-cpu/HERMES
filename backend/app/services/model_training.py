from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


PROCESSED_DIRECTORY = Path("data/processed")
MODEL_DIRECTORY = Path("data/models")
REPORT_DIRECTORY = Path("data/reports")

DEFAULT_DATASET_PATH = (
    PROCESSED_DIRECTORY
    / "tcga_brca_pilot_25_patients_ml_ready.csv"
)

DEFAULT_MODEL_PATH = (
    MODEL_DIRECTORY
    / "tcga_brca_vital_status_logistic_regression.joblib"
)

DEFAULT_REPORT_PATH = (
    REPORT_DIRECTORY
    / "tcga_brca_vital_status_baseline_report.json"
)

DEFAULT_COMPARISON_MODEL_PATH = (
    MODEL_DIRECTORY
    / "trojan_horse_best_model_v1.joblib"
)

DEFAULT_COMPARISON_REPORT_PATH = (
    REPORT_DIRECTORY
    / "trojan_horse_model_comparison_v1.json"
)

IDENTIFIER_COLUMN = "patient_id"
TARGET_COLUMN = "vital_status"

NON_FEATURE_COLUMNS = {
    IDENTIFIER_COLUMN,
    TARGET_COLUMN,
    "days_to_death",
    "days_to_last_follow_up",
}

TARGET_MAPPING = {
    "alive": 0,
    "dead": 1,
}


class ModelTrainingError(RuntimeError):
    """Raised when a machine-learning model cannot be trained."""


def _is_missing(value: str | None) -> bool:
    if value is None:
        return True

    return value.strip().lower() in {
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
    }


def _read_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise ModelTrainingError(
            f"The processed dataset was not found at '{path}'."
        )

    if path.stat().st_size == 0:
        raise ModelTrainingError(
            f"The processed dataset at '{path}' is empty."
        )

    with path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if not reader.fieldnames:
            raise ModelTrainingError(
                "The processed dataset has no column headers."
            )

        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    if not rows:
        raise ModelTrainingError(
            "The processed dataset has no patient rows."
        )

    return fieldnames, rows


def _parse_target(value: str | None) -> int | None:
    if _is_missing(value):
        return None

    return TARGET_MAPPING.get(str(value).strip().lower())


def _parse_feature(
    value: str | None,
    column: str,
    patient_id: str,
) -> float:
    if _is_missing(value):
        raise ModelTrainingError(
            f"Feature '{column}' is missing for patient "
            f"'{patient_id}'."
        )

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelTrainingError(
            f"Feature '{column}' contains a nonnumeric value "
            f"for patient '{patient_id}': '{value}'."
        ) from exc

    if math.isnan(parsed) or math.isinf(parsed):
        raise ModelTrainingError(
            f"Feature '{column}' contains a non-finite value "
            f"for patient '{patient_id}'."
        )

    return parsed


def _calculate_specificity(
    y_true: list[int],
    y_predicted: list[int],
) -> float:
    matrix = confusion_matrix(
        y_true,
        y_predicted,
        labels=[0, 1],
    )

    true_negative = int(matrix[0][0])
    false_positive = int(matrix[0][1])
    denominator = true_negative + false_positive

    if denominator == 0:
        return 0.0

    return true_negative / denominator


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            payload,
            output_file,
            indent=2,
            ensure_ascii=False,
        )


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _prepare_training_data(
    source_path: Path,
) -> dict[str, Any]:
    fieldnames, rows = _read_csv(source_path)

    required_columns = {
        IDENTIFIER_COLUMN,
        TARGET_COLUMN,
    }

    missing_required_columns = sorted(
        required_columns - set(fieldnames)
    )

    if missing_required_columns:
        raise ModelTrainingError(
            "The processed dataset is missing required columns: "
            f"{missing_required_columns}"
        )

    feature_names = [
        column
        for column in fieldnames
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_names:
        raise ModelTrainingError(
            "No model feature columns were found."
        )

    patient_ids: list[str] = []
    feature_matrix: list[list[float]] = []
    targets: list[int] = []
    excluded_patients: list[dict[str, str]] = []

    for row in rows:
        patient_id = (
            row.get(IDENTIFIER_COLUMN)
            or ""
        ).strip()

        target = _parse_target(row.get(TARGET_COLUMN))

        if target is None:
            excluded_patients.append(
                {
                    "patient_id": patient_id,
                    "reason": "Missing or unsupported vital_status.",
                    "vital_status": str(
                        row.get(TARGET_COLUMN) or ""
                    ),
                }
            )
            continue

        feature_values = [
            _parse_feature(
                value=row.get(column),
                column=column,
                patient_id=patient_id,
            )
            for column in feature_names
        ]

        patient_ids.append(patient_id)
        feature_matrix.append(feature_values)
        targets.append(target)

    if len(feature_matrix) < 4:
        raise ModelTrainingError(
            "At least four patients with valid outcomes are "
            "required for model training."
        )

    class_counts = Counter(targets)
    alive_count = int(class_counts.get(0, 0))
    dead_count = int(class_counts.get(1, 0))

    if alive_count == 0 or dead_count == 0:
        raise ModelTrainingError(
            "Both Alive and Dead patients are required to train "
            "a binary classification model."
        )

    minority_class_count = min(alive_count, dead_count)

    if minority_class_count < 2:
        raise ModelTrainingError(
            "At least two patients are required in each outcome "
            "class for stratified cross-validation."
        )

    number_of_folds = min(5, minority_class_count)

    return {
        "rows": rows,
        "patient_ids": patient_ids,
        "feature_names": feature_names,
        "feature_matrix": feature_matrix,
        "targets": targets,
        "excluded_patients": excluded_patients,
        "alive_count": alive_count,
        "dead_count": dead_count,
        "number_of_folds": number_of_folds,
    }


def _evaluate_model(
    model: Any,
    feature_matrix: list[list[float]],
    targets: list[int],
    cross_validator: StratifiedKFold,
) -> dict[str, Any]:
    predicted_probabilities = cross_val_predict(
        estimator=model,
        X=feature_matrix,
        y=targets,
        cv=cross_validator,
        method="predict_proba",
    )[:, 1]

    predicted_classes = [
        1 if probability >= 0.5 else 0
        for probability in predicted_probabilities
    ]

    confusion = confusion_matrix(
        targets,
        predicted_classes,
        labels=[0, 1],
    )

    metrics = {
        "accuracy": _round_metric(
            accuracy_score(targets, predicted_classes)
        ),
        "balanced_accuracy": _round_metric(
            balanced_accuracy_score(
                targets,
                predicted_classes,
            )
        ),
        "sensitivity_recall": _round_metric(
            recall_score(
                targets,
                predicted_classes,
                zero_division=0,
            )
        ),
        "specificity": _round_metric(
            _calculate_specificity(
                targets,
                predicted_classes,
            )
        ),
        "precision": _round_metric(
            precision_score(
                targets,
                predicted_classes,
                zero_division=0,
            )
        ),
        "f1_score": _round_metric(
            f1_score(
                targets,
                predicted_classes,
                zero_division=0,
            )
        ),
        "roc_auc": _round_metric(
            roc_auc_score(
                targets,
                predicted_probabilities,
            )
        ),
    }

    return {
        "metrics": metrics,
        "confusion_matrix": confusion.tolist(),
    }


def _extract_feature_importance(
    model: Any,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    values = None
    value_name = "importance"

    if hasattr(model, "coef_"):
        values = model.coef_[0]
        value_name = "coefficient"
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_

    if values is None:
        return []

    importance = [
        {
            "feature": feature_name,
            value_name: round(float(value), 6),
            "absolute_importance": round(
                abs(float(value)),
                6,
            ),
        }
        for feature_name, value in zip(
            feature_names,
            values,
        )
    ]

    importance.sort(
        key=lambda item: item["absolute_importance"],
        reverse=True,
    )

    return importance


def train_tcga_brca_baseline_model(
    dataset_path: str | None = None,
    model_path: str | None = None,
    report_path: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    source_path = (
        Path(dataset_path)
        if dataset_path
        else DEFAULT_DATASET_PATH
    )

    resolved_model_path = (
        Path(model_path)
        if model_path
        else DEFAULT_MODEL_PATH
    )

    resolved_report_path = (
        Path(report_path)
        if report_path
        else DEFAULT_REPORT_PATH
    )

    data = _prepare_training_data(source_path)

    cross_validator = StratifiedKFold(
        n_splits=data["number_of_folds"],
        shuffle=True,
        random_state=random_state,
    )

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )

    evaluation = _evaluate_model(
        model=model,
        feature_matrix=data["feature_matrix"],
        targets=data["targets"],
        cross_validator=cross_validator,
    )

    model.fit(
        data["feature_matrix"],
        data["targets"],
    )

    feature_importance = _extract_feature_importance(
        model,
        data["feature_names"],
    )

    resolved_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": model,
        "model_name": "logistic_regression",
        "feature_names": data["feature_names"],
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "positive_class": "Dead",
        "negative_class": "Alive",
        "decision_threshold": 0.5,
        "training_patient_ids": data["patient_ids"],
        "source_dataset_path": str(source_path),
        "research_use_only": True,
    }

    joblib.dump(model_package, resolved_model_path)

    report = {
        "project": "TCGA-BRCA",
        "model_name": "Baseline vital-status logistic regression",
        "model_type": "LogisticRegression",
        "training_status": "complete",
        "source_dataset_path": str(source_path),
        "model_path": str(resolved_model_path),
        "dataset_summary": {
            "source_patient_count": len(data["rows"]),
            "training_patient_count": len(
                data["feature_matrix"]
            ),
            "excluded_patient_count": len(
                data["excluded_patients"]
            ),
            "alive_count": data["alive_count"],
            "dead_count": data["dead_count"],
            "feature_count": len(data["feature_names"]),
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "number_of_folds": data["number_of_folds"],
            "shuffle": True,
            "random_state": random_state,
            "decision_threshold": 0.5,
        },
        "cross_validated_metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "top_feature_importance": feature_importance[:15],
        "all_feature_importance": feature_importance,
        "excluded_patients": data["excluded_patients"],
        "research_use_only": True,
    }

    _write_json(resolved_report_path, report)

    return {
        "training_status": "complete",
        "model_type": "LogisticRegression",
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "source_patient_count": len(data["rows"]),
        "training_patient_count": len(
            data["feature_matrix"]
        ),
        "excluded_patient_count": len(
            data["excluded_patients"]
        ),
        "class_distribution": {
            "alive": data["alive_count"],
            "dead": data["dead_count"],
        },
        "feature_count": len(data["feature_names"]),
        "cross_validation_folds": data["number_of_folds"],
        "cross_validated_metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "model_path": str(resolved_model_path),
        "report_path": str(resolved_report_path),
        "ready_for_technical_prediction_testing": True,
        "clinically_validated": False,
        "research_use_only": True,
    }


def train_tcga_brca_model_comparison(
    dataset_path: str | None = None,
    best_model_path: str | None = None,
    report_path: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and compare multiple classifiers using identical folds.

    The best model is selected primarily by balanced accuracy,
    followed by ROC-AUC and F1 score. This prioritizes performance
    across both classes in the imbalanced vital-status dataset.
    """

    source_path = (
        Path(dataset_path)
        if dataset_path
        else DEFAULT_DATASET_PATH
    )

    resolved_model_path = (
        Path(best_model_path)
        if best_model_path
        else DEFAULT_COMPARISON_MODEL_PATH
    )

    resolved_report_path = (
        Path(report_path)
        if report_path
        else DEFAULT_COMPARISON_REPORT_PATH
    )

    data = _prepare_training_data(source_path)

    models = {
        "logistic_regression": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=3,
            random_state=random_state,
        ),
    }

    comparison_results: list[dict[str, Any]] = []

    for model_name, model in models.items():
        cross_validator = StratifiedKFold(
            n_splits=data["number_of_folds"],
            shuffle=True,
            random_state=random_state,
        )

        evaluation = _evaluate_model(
            model=model,
            feature_matrix=data["feature_matrix"],
            targets=data["targets"],
            cross_validator=cross_validator,
        )

        comparison_results.append(
            {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "metrics": evaluation["metrics"],
                "confusion_matrix": evaluation[
                    "confusion_matrix"
                ],
            }
        )

    comparison_results.sort(
        key=lambda result: (
            result["metrics"]["balanced_accuracy"],
            result["metrics"]["roc_auc"],
            result["metrics"]["f1_score"],
        ),
        reverse=True,
    )

    best_result = comparison_results[0]
    best_model_name = best_result["model_name"]
    best_model = models[best_model_name]

    best_model.fit(
        data["feature_matrix"],
        data["targets"],
    )

    feature_importance = _extract_feature_importance(
        best_model,
        data["feature_names"],
    )

    resolved_model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": best_model,
        "model_name": best_model_name,
        "model_type": type(best_model).__name__,
        "selection_metric": "balanced_accuracy",
        "selection_metrics": best_result["metrics"],
        "feature_names": data["feature_names"],
        "target_column": TARGET_COLUMN,
        "target_mapping": TARGET_MAPPING,
        "positive_class": "Dead",
        "negative_class": "Alive",
        "decision_threshold": 0.5,
        "training_patient_ids": data["patient_ids"],
        "source_dataset_path": str(source_path),
        "research_use_only": True,
        "clinically_validated": False,
    }

    joblib.dump(model_package, resolved_model_path)

    report = {
        "project": "TCGA-BRCA",
        "training_status": "complete",
        "framework_name": "HERMES model comparison v1",
        "source_dataset_path": str(source_path),
        "best_model_path": str(resolved_model_path),
        "selection_rule": (
            "Highest balanced accuracy; ties resolved by ROC-AUC "
            "and then F1 score."
        ),
        "dataset_summary": {
            "source_patient_count": len(data["rows"]),
            "training_patient_count": len(
                data["feature_matrix"]
            ),
            "excluded_patient_count": len(
                data["excluded_patients"]
            ),
            "alive_count": data["alive_count"],
            "dead_count": data["dead_count"],
            "feature_count": len(data["feature_names"]),
        },
        "cross_validation": {
            "method": "StratifiedKFold",
            "number_of_folds": data["number_of_folds"],
            "shuffle": True,
            "random_state": random_state,
            "decision_threshold": 0.5,
        },
        "models_compared": comparison_results,
        "best_model": best_result,
        "best_model_top_feature_importance": (
            feature_importance[:20]
        ),
        "excluded_patients": data["excluded_patients"],
        "limitations": [
            (
                "Vital status is a preliminary binary endpoint "
                "that does not account for follow-up time."
            ),
            (
                "The cohort is imbalanced, with fewer deceased "
                "than living patients."
            ),
            (
                "Model selection and evaluation use the same "
                "cohort and therefore require later validation "
                "on an independent holdout or external cohort."
            ),
            (
                "The saved model is for research and technical "
                "development only, not clinical care."
            ),
        ],
        "recommended_next_step": (
            "Add nested tuning or a locked holdout set, then build "
            "the prediction endpoint around the saved model package."
        ),
        "research_use_only": True,
        "clinically_validated": False,
    }

    _write_json(resolved_report_path, report)

    return {
        "training_status": "complete",
        "framework": "multi_model_comparison",
        "models_compared": [
            result["model_name"]
            for result in comparison_results
        ],
        "best_model": best_result,
        "model_rankings": comparison_results,
        "selection_metric": "balanced_accuracy",
        "source_patient_count": len(data["rows"]),
        "training_patient_count": len(
            data["feature_matrix"]
        ),
        "class_distribution": {
            "alive": data["alive_count"],
            "dead": data["dead_count"],
        },
        "feature_count": len(data["feature_names"]),
        "cross_validation_folds": data["number_of_folds"],
        "best_model_path": str(resolved_model_path),
        "report_path": str(resolved_report_path),
        "ready_for_prediction_endpoint_development": True,
        "clinically_validated": False,
        "research_use_only": True,
    }
