from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.ai.model_paths import (
    normalize_model_name,
    optimized_model_path as namespace_optimized_model_path,
    optimization_report_path as namespace_optimization_report_path,
)

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
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
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)

from backend.app.services.model_training import (
    ModelTrainingError,
    TARGET_COLUMN,
    TARGET_MAPPING,
    _extract_feature_importance,
    _prepare_training_data,
    _write_json,
)


DEFAULT_OPTIMIZATION_DATASET_PATH = (
    "data/processed/"
    "tcga_brca_cohort_250_patients_ml_ready.csv"
)
DEFAULT_OPTIMIZED_MODEL_PATH = (
    "data/models/"
    "trojan_horse_optimized_model_v3.joblib"
)
DEFAULT_OPTIMIZATION_REPORT_PATH = (
    "data/reports/"
    "trojan_horse_model_optimization_v3.json"
)


class AIOptimizationError(RuntimeError):
    """Raised when the AI optimization workflow cannot complete."""


def _round_metric(value: float) -> float:
    return round(float(value), 4)


def _specificity(
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

    return (
        true_negative / denominator
        if denominator
        else 0.0
    )


def _calculate_metrics(
    targets: list[int],
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )

    metrics = {
        "accuracy": _round_metric(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "balanced_accuracy": _round_metric(
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),
        "sensitivity_recall": _round_metric(
            recall_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": _round_metric(
            _specificity(
                targets,
                predictions,
            )
        ),
        "precision": _round_metric(
            precision_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": _round_metric(
            f1_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": _round_metric(
            roc_auc_score(
                targets,
                probabilities,
            )
        ),
    }

    return {
        "threshold": round(
            float(threshold),
            3,
        ),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }


def _select_threshold(
    targets: list[int],
    probabilities: np.ndarray,
) -> dict[str, Any]:
    candidates = np.arange(
        0.15,
        0.76,
        0.01,
    )

    evaluations = [
        _calculate_metrics(
            targets=targets,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in candidates
    ]

    evaluations.sort(
        key=lambda result: (
            result["metrics"][
                "balanced_accuracy"
            ],
            result["metrics"][
                "sensitivity_recall"
            ],
            result["metrics"]["f1_score"],
            -abs(result["threshold"] - 0.5),
        ),
        reverse=True,
    )

    return evaluations[0]


def _model_search_spaces(
    random_state: int,
) -> dict[str, dict[str, Any]]:
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                solver="liblinear",
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            ),
            "parameters": {
                "penalty": ["l1", "l2"],
                "C": [
                    0.01,
                    0.05,
                    0.1,
                    0.5,
                    1.0,
                    5.0,
                    10.0,
                ],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight=(
                    "balanced_subsample"
                ),
                random_state=random_state,
                n_jobs=-1,
            ),
            "parameters": {
                "n_estimators": [
                    250,
                    500,
                ],
                "max_depth": [
                    None,
                    5,
                    10,
                ],
                "min_samples_leaf": [
                    1,
                    2,
                    4,
                ],
                "max_features": [
                    "sqrt",
                    0.5,
                ],
            },
        },
        "gradient_boosting": {
            "estimator": (
                GradientBoostingClassifier(
                    random_state=random_state,
                )
            ),
            "parameters": {
                "n_estimators": [
                    100,
                    200,
                ],
                "learning_rate": [
                    0.03,
                    0.05,
                    0.1,
                ],
                "max_depth": [
                    1,
                    2,
                    3,
                ],
                "min_samples_leaf": [
                    2,
                    4,
                    8,
                ],
            },
        },
    }


def optimize_ai_models(
    dataset_path: str = (
        DEFAULT_OPTIMIZATION_DATASET_PATH
    ),
    optimized_model_path: str = (
        DEFAULT_OPTIMIZED_MODEL_PATH
    ),
    report_path: str = (
        DEFAULT_OPTIMIZATION_REPORT_PATH
    ),
    random_state: int = 42,
    model_name: str = "brca",
) -> dict[str, Any]:
    """
    Tune three classifiers and optimize their decision thresholds.

    Hyperparameters are selected by stratified cross-validated
    balanced accuracy. Out-of-fold probabilities are then used to
    select a development decision threshold that prioritizes
    balanced accuracy, sensitivity, and F1 score.

    This remains a development estimate. A locked holdout or
    independent external cohort is still required before making
    performance claims.
    """

    namespace = normalize_model_name(model_name)
    if optimized_model_path == DEFAULT_OPTIMIZED_MODEL_PATH:
        optimized_model_path = namespace_optimized_model_path(namespace)
    if report_path == DEFAULT_OPTIMIZATION_REPORT_PATH:
        report_path = namespace_optimization_report_path(namespace)

    try:
        source_path = Path(dataset_path)
        data = _prepare_training_data(
            source_path
        )

        inner_cv = StratifiedKFold(
            n_splits=data["number_of_folds"],
            shuffle=True,
            random_state=random_state,
        )

        search_spaces = _model_search_spaces(
            random_state=random_state,
        )

        results: list[dict[str, Any]] = []
        fitted_searches: dict[str, GridSearchCV] = {}

        for model_name, specification in (
            search_spaces.items()
        ):
            search = GridSearchCV(
                estimator=specification[
                    "estimator"
                ],
                param_grid=specification[
                    "parameters"
                ],
                scoring="balanced_accuracy",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )

            search.fit(
                data["feature_matrix"],
                data["targets"],
            )

            best_estimator = clone(
                search.best_estimator_
            )

            probability_cv = (
                StratifiedKFold(
                    n_splits=(
                        data[
                            "number_of_folds"
                        ]
                    ),
                    shuffle=True,
                    random_state=(
                        random_state + 101
                    ),
                )
            )

            probabilities = cross_val_predict(
                estimator=best_estimator,
                X=data["feature_matrix"],
                y=data["targets"],
                cv=probability_cv,
                method="predict_proba",
                n_jobs=-1,
            )[:, 1]

            default_evaluation = (
                _calculate_metrics(
                    targets=data["targets"],
                    probabilities=probabilities,
                    threshold=0.5,
                )
            )

            optimized_evaluation = (
                _select_threshold(
                    targets=data["targets"],
                    probabilities=probabilities,
                )
            )

            results.append(
                {
                    "model_name": model_name,
                    "model_type": type(
                        search.best_estimator_
                    ).__name__,
                    "best_parameters": (
                        search.best_params_
                    ),
                    "grid_search_best_balanced_accuracy": (
                        _round_metric(
                            search.best_score_
                        )
                    ),
                    "default_threshold_evaluation": (
                        default_evaluation
                    ),
                    "optimized_threshold_evaluation": (
                        optimized_evaluation
                    ),
                }
            )

            fitted_searches[
                model_name
            ] = search

        results.sort(
            key=lambda result: (
                result[
                    "optimized_threshold_evaluation"
                ]["metrics"][
                    "balanced_accuracy"
                ],
                result[
                    "optimized_threshold_evaluation"
                ]["metrics"]["roc_auc"],
                result[
                    "optimized_threshold_evaluation"
                ]["metrics"][
                    "sensitivity_recall"
                ],
                result[
                    "optimized_threshold_evaluation"
                ]["metrics"]["f1_score"],
            ),
            reverse=True,
        )

        best_result = results[0]
        best_model_name = best_result[
            "model_name"
        ]
        best_search = fitted_searches[
            best_model_name
        ]
        best_model = (
            best_search.best_estimator_
        )

        best_model.fit(
            data["feature_matrix"],
            data["targets"],
        )

        feature_importance = (
            _extract_feature_importance(
                best_model,
                data["feature_names"],
            )
        )

        resolved_model_path = Path(
            optimized_model_path
        )
        resolved_report_path = Path(
            report_path
        )

        resolved_model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        optimized_threshold = (
            best_result[
                "optimized_threshold_evaluation"
            ]["threshold"]
        )

        model_package = {
            "model": best_model,
            "model_namespace": namespace,
            "model_name": best_model_name,
            "model_type": type(
                best_model
            ).__name__,
            "best_parameters": (
                best_search.best_params_
            ),
            "selection_metric": (
                "optimized_threshold_"
                "balanced_accuracy"
            ),
            "selection_metrics": (
                best_result[
                    "optimized_threshold_evaluation"
                ]["metrics"]
            ),
            "feature_names": data[
                "feature_names"
            ],
            "target_column": TARGET_COLUMN,
            "target_mapping": TARGET_MAPPING,
            "positive_class": "Dead",
            "negative_class": "Alive",
            "decision_threshold": (
                optimized_threshold
            ),
            "training_patient_ids": data[
                "patient_ids"
            ],
            "source_dataset_path": str(
                source_path
            ),
            "optimization_version": "v3",
            "research_use_only": True,
            "clinically_validated": False,
        }

        joblib.dump(
            model_package,
            resolved_model_path,
        )

        report = {
            "project": "Project HERMES",
            "model_namespace": namespace,
            "task": (
                "TCGA-BRCA vital-status "
                "classification"
            ),
            "optimization_status": "complete",
            "source_dataset_path": str(
                source_path
            ),
            "optimized_model_path": str(
                resolved_model_path
            ),
            "dataset_summary": {
                "source_patient_count": len(
                    data["rows"]
                ),
                "training_patient_count": len(
                    data["feature_matrix"]
                ),
                "excluded_patient_count": len(
                    data["excluded_patients"]
                ),
                "alive_count": data[
                    "alive_count"
                ],
                "dead_count": data[
                    "dead_count"
                ],
                "feature_count": len(
                    data["feature_names"]
                ),
            },
            "search_strategy": {
                "method": "GridSearchCV",
                "scoring": (
                    "balanced_accuracy"
                ),
                "cross_validation": (
                    "StratifiedKFold"
                ),
                "number_of_folds": data[
                    "number_of_folds"
                ],
                "random_state": random_state,
                "threshold_range": (
                    "0.15 to 0.75 in "
                    "0.01 increments"
                ),
            },
            "model_rankings": results,
            "best_model": best_result,
            "best_model_top_feature_importance": (
                feature_importance[:20]
            ),
            "all_best_model_feature_importance": (
                feature_importance
            ),
            "limitations": [
                (
                    "Hyperparameter selection and "
                    "threshold development use the "
                    "same cohort."
                ),
                (
                    "Performance must be confirmed "
                    "on a locked holdout set or an "
                    "independent external cohort."
                ),
                (
                    "Vital status does not account "
                    "for time-to-event or censoring."
                ),
                (
                    "The present cohort is TCGA-BRCA "
                    "and is not yet restricted to "
                    "confirmed TNBC cases."
                ),
                (
                    "The model is for research and "
                    "technical development only."
                ),
            ],
            "recommended_next_step": (
                "Build a validated inference endpoint "
                "around the optimized model package, "
                "then add local feature explanations."
            ),
            "research_use_only": True,
            "clinically_validated": False,
        }

        _write_json(
            resolved_report_path,
            report,
        )

        return {
            "ai_optimization_status": (
                "complete"
            ),
            "platform": (
                "Project HERMES"
            ),
            "models_optimized": [
                result["model_name"]
                for result in results
            ],
            "best_model": best_result,
            "source_patient_count": len(
                data["rows"]
            ),
            "training_patient_count": len(
                data["feature_matrix"]
            ),
            "class_distribution": {
                "alive": data["alive_count"],
                "dead": data["dead_count"],
            },
            "feature_count": len(
                data["feature_names"]
            ),
            "optimized_model_path": str(
                resolved_model_path
            ),
            "report_path": str(
                resolved_report_path
            ),
            "ready_for_prediction_endpoint": (
                True
            ),
            "clinically_validated": False,
            "research_use_only": True,
        }

    except (
        ModelTrainingError,
        ValueError,
        OSError,
    ) as exc:
        raise AIOptimizationError(
            str(exc)
        ) from exc
