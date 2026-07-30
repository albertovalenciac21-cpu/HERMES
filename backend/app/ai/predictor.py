from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.app.ai.model_paths import preferred_model_path

import joblib
import numpy as np

from backend.app.ai.explainability import (
    ExplainabilityError,
    generate_shap_explanation,
)


DEFAULT_PREDICTION_MODEL_PATH = (
    "data/models/"
    "trojan_horse_optimized_model_v3.joblib"
)

DEFAULT_PREDICTION_DATASET_PATH = (
    "data/processed/"
    "tcga_brca_cohort_250_patients_ml_ready.csv"
)


class AIPredictionError(RuntimeError):
    """Raised when an AI prediction cannot be completed."""


def _load_model_package(
    model_path: str | None = None,
    model_name: str = "brca",
) -> dict[str, Any]:
    resolved_path = Path(model_path or preferred_model_path(model_name))

    if not resolved_path.exists():
        raise AIPredictionError(
            f"The model was not found at '{resolved_path}'."
        )

    try:
        package = joblib.load(resolved_path)
    except Exception as exc:
        raise AIPredictionError(
            f"The model package could not be loaded: {exc}"
        ) from exc

    if not isinstance(package, dict):
        raise AIPredictionError(
            "The saved artifact is not a valid model package."
        )

    required_keys = {
        "model",
        "feature_names",
        "target_mapping",
    }

    missing_keys = sorted(
        required_keys.difference(package)
    )

    if missing_keys:
        raise AIPredictionError(
            "The model package is missing required keys: "
            + ", ".join(missing_keys)
        )

    return package


def _normalize_feature_payload(
    supplied_features: dict[str, Any],
    expected_features: list[str],
) -> tuple[np.ndarray, dict[str, float], list[str]]:
    if not supplied_features:
        raise AIPredictionError(
            "No model features were supplied."
        )

    unexpected_features = sorted(
        set(supplied_features).difference(
            expected_features
        )
    )

    if unexpected_features:
        raise AIPredictionError(
            "Unexpected features were supplied: "
            + ", ".join(unexpected_features)
        )

    missing_features = [
        feature
        for feature in expected_features
        if feature not in supplied_features
    ]

    normalized: dict[str, float] = {}

    for feature in expected_features:
        raw_value = supplied_features.get(
            feature,
            0.0,
        )

        if raw_value is None or raw_value == "":
            raw_value = 0.0

        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise AIPredictionError(
                f"Feature '{feature}' must be numeric."
            ) from exc

        if not np.isfinite(numeric_value):
            raise AIPredictionError(
                f"Feature '{feature}' must be finite."
            )

        normalized[feature] = numeric_value

    matrix = np.asarray(
        [[
            normalized[feature]
            for feature in expected_features
        ]],
        dtype=float,
    )

    return matrix, normalized, missing_features


def _linear_feature_contributions(
    model: Any,
    feature_names: list[str],
    feature_values: dict[str, float],
) -> list[dict[str, Any]]:
    coefficients = getattr(model, "coef_", None)

    if coefficients is None:
        return []

    coefficient_array = np.asarray(coefficients)

    if (
        coefficient_array.ndim != 2
        or coefficient_array.shape[0] != 1
    ):
        return []

    values = np.asarray(
        [
            feature_values[feature]
            for feature in feature_names
        ],
        dtype=float,
    )

    contributions = coefficient_array[0] * values
    records: list[dict[str, Any]] = []

    for feature, value, coefficient, impact in zip(
        feature_names,
        values,
        coefficient_array[0],
        contributions,
    ):
        records.append(
            {
                "feature": feature,
                "feature_value": round(float(value), 6),
                "coefficient": round(
                    float(coefficient),
                    6,
                ),
                "linear_contribution": round(
                    float(impact),
                    6,
                ),
                "direction": (
                    "toward_dead"
                    if impact > 0
                    else (
                        "toward_alive"
                        if impact < 0
                        else "neutral"
                    )
                ),
            }
        )

    records.sort(
        key=lambda item: abs(
            item["linear_contribution"]
        ),
        reverse=True,
    )

    return records


def predict_from_features(
    features: dict[str, Any],
    model_path: str | None = None,
    model_name: str = "brca",
    top_feature_count: int = 10,
    include_shap: bool = True,
) -> dict[str, Any]:
    resolved_model_path = model_path or preferred_model_path(model_name)
    package = _load_model_package(
        model_path=resolved_model_path,
        model_name=model_name,
    )

    model = package["model"]
    expected_features = list(
        package["feature_names"]
    )

    (
        feature_matrix,
        normalized_features,
        missing_features,
    ) = _normalize_feature_payload(
        supplied_features=features,
        expected_features=expected_features,
    )

    if not hasattr(model, "predict_proba"):
        raise AIPredictionError(
            "The saved model does not support "
            "probability predictions."
        )

    probabilities = model.predict_proba(
        feature_matrix
    )[0]

    classes = list(
        getattr(model, "classes_", [0, 1])
    )

    try:
        positive_index = classes.index(1)
    except ValueError as exc:
        raise AIPredictionError(
            "The model does not contain the expected "
            "positive class label 1."
        ) from exc

    dead_probability = float(
        probabilities[positive_index]
    )
    alive_probability = float(
        1.0 - dead_probability
    )

    decision_threshold = float(
        package.get("decision_threshold", 0.5)
    )

    predicted_numeric = int(
        dead_probability >= decision_threshold
    )

    inverse_mapping = {
        int(value): key
        for key, value in package[
            "target_mapping"
        ].items()
    }

    predicted_label = inverse_mapping.get(
        predicted_numeric,
        (
            "Dead"
            if predicted_numeric == 1
            else "Alive"
        ),
    )

    contributions = _linear_feature_contributions(
        model=model,
        feature_names=expected_features,
        feature_values=normalized_features,
    )

    source_dataset_path = package.get(
        "source_dataset_path",
        DEFAULT_PREDICTION_DATASET_PATH,
    )

    shap_result: dict[str, Any]

    if include_shap:
        try:
            shap_result = generate_shap_explanation(
                model=model,
                feature_matrix=feature_matrix,
                feature_names=expected_features,
                feature_values=normalized_features,
                background_dataset_path=(
                    source_dataset_path
                ),
                top_feature_count=(
                    top_feature_count
                ),
            )
        except ExplainabilityError as exc:
            shap_result = {
                "available": False,
                "error": str(exc),
                "fallback_used": (
                    "local_linear_explanation"
                    if contributions
                    else None
                ),
            }
    else:
        shap_result = {
            "available": False,
            "disabled_by_request": True,
        }

    return {
        "prediction_status": "complete",
        "predicted_class": predicted_label,
        "predicted_numeric_class": (
            predicted_numeric
        ),
        "probabilities": {
            "alive": round(
                alive_probability,
                6,
            ),
            "dead": round(
                dead_probability,
                6,
            ),
        },
        "decision_threshold": round(
            decision_threshold,
            6,
        ),
        "threshold_margin": round(
            dead_probability
            - decision_threshold,
            6,
        ),
        "model": {
            "model_name": package.get(
                "model_name"
            ),
            "model_type": package.get(
                "model_type",
                type(model).__name__,
            ),
            "optimization_version": (
                package.get(
                    "optimization_version"
                )
            ),
            "source_dataset_path": (
                source_dataset_path
            ),
            "model_path": str(
                Path(resolved_model_path)
            ),
        },
        "input_summary": {
            "expected_feature_count": len(
                expected_features
            ),
            "supplied_feature_count": len(
                features
            ),
            "features_defaulted_to_zero": (
                missing_features
            ),
        },
        "shap_explanation": shap_result,
        "local_linear_explanation": {
            "available": bool(
                contributions
            ),
            "method": (
                "coefficient_times_feature_value"
                if contributions
                else None
            ),
            "top_contributions": (
                contributions[
                    :top_feature_count
                ]
            ),
            "note": (
                "This is an exact decomposition of "
                "the fitted linear decision function. "
                "SHAP is reported separately."
                if contributions
                else (
                    "Local linear contributions are "
                    "not available for this model type."
                )
            ),
        },
        "research_use_only": True,
        "clinically_validated": False,
        "clinical_warning": (
            "This prediction is a research prototype "
            "and must not be used for diagnosis, "
            "prognosis, or treatment decisions."
        ),
    }


def _read_patient_features(
    patient_id: str,
    dataset_path: str,
) -> dict[str, Any]:
    resolved_path = Path(dataset_path)

    if not resolved_path.exists():
        raise AIPredictionError(
            f"The dataset was not found at "
            f"'{resolved_path}'."
        )

    with resolved_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise AIPredictionError(
                "The dataset does not have a header."
            )

        patient_column = (
            "patient_id"
            if "patient_id" in reader.fieldnames
            else (
                "patient"
                if "patient" in reader.fieldnames
                else None
            )
        )

        if patient_column is None:
            raise AIPredictionError(
                "The dataset does not contain "
                "'patient_id' or 'patient'."
            )

        for row in reader:
            if (
                str(
                    row.get(patient_column, "")
                ).strip()
                == patient_id.strip()
            ):
                return row

    raise AIPredictionError(
        f"Patient '{patient_id}' was not found "
        f"in '{resolved_path}'."
    )


def predict_existing_patient(
    patient_id: str,
    dataset_path: str | None = None,
    model_path: str | None = None,
    model_name: str = "brca",
    top_feature_count: int = 10,
    include_shap: bool = True,
) -> dict[str, Any]:
    resolved_model_path = model_path or preferred_model_path(model_name)
    package = _load_model_package(
        model_path=resolved_model_path,
        model_name=model_name,
    )
    resolved_dataset_path = dataset_path or package.get(
        "source_dataset_path", DEFAULT_PREDICTION_DATASET_PATH
    )

    row = _read_patient_features(
        patient_id=patient_id,
        dataset_path=resolved_dataset_path,
    )

    features = {
        feature: row.get(feature)
        for feature in package[
            "feature_names"
        ]
    }

    result = predict_from_features(
        features=features,
        model_path=resolved_model_path,
        model_name=model_name,
        top_feature_count=top_feature_count,
        include_shap=include_shap,
    )

    result["patient_id"] = patient_id
    result["prediction_source"] = {
        "type": (
            "existing_ml_ready_dataset_row"
        ),
        "dataset_path": str(
            Path(resolved_dataset_path)
        ),
    }

    if "vital_status" in row:
        result["observed_label"] = (
            row["vital_status"]
        )
        result["evaluation_warning"] = (
            "This patient may have been used during "
            "model development. The result is a "
            "technical pipeline test, not an "
            "independent validation."
        )

    return result
