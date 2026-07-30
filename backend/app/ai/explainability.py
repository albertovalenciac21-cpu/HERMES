from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


class ExplainabilityError(RuntimeError):
    """Raised when a local SHAP explanation cannot be generated."""


def _load_background_matrix(
    dataset_path: str,
    feature_names: list[str],
    maximum_rows: int = 100,
) -> np.ndarray:
    resolved_path = Path(dataset_path)

    if not resolved_path.exists():
        raise ExplainabilityError(
            "The SHAP background dataset was not found at "
            f"'{resolved_path}'."
        )

    rows: list[list[float]] = []

    with resolved_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if not reader.fieldnames:
            raise ExplainabilityError(
                "The SHAP background dataset does not have a header."
            )

        missing_columns = [
            feature
            for feature in feature_names
            if feature not in reader.fieldnames
        ]

        if missing_columns:
            raise ExplainabilityError(
                "The SHAP background dataset is missing features: "
                + ", ".join(missing_columns)
            )

        for row in reader:
            values: list[float] = []

            for feature in feature_names:
                raw_value = row.get(feature)

                if raw_value is None or raw_value == "":
                    raw_value = 0.0

                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise ExplainabilityError(
                        f"Background feature '{feature}' is not numeric."
                    ) from exc

                if not np.isfinite(numeric_value):
                    numeric_value = 0.0

                values.append(numeric_value)

            rows.append(values)

            if len(rows) >= maximum_rows:
                break

    if not rows:
        raise ExplainabilityError(
            "The SHAP background dataset does not contain any rows."
        )

    return np.asarray(rows, dtype=float)


def _extract_positive_class_values(
    explanation: Any,
) -> tuple[np.ndarray, float | None]:
    """
    Normalize SHAP output across older and newer SHAP releases.
    """

    values = getattr(explanation, "values", explanation)
    base_values = getattr(explanation, "base_values", None)

    if isinstance(values, list):
        values_array = np.asarray(values[-1])
    else:
        values_array = np.asarray(values)

    if values_array.ndim == 3:
        values_array = values_array[:, :, -1]

    if values_array.ndim == 2:
        patient_values = values_array[0]
    elif values_array.ndim == 1:
        patient_values = values_array
    else:
        raise ExplainabilityError(
            "SHAP returned an unexpected explanation shape."
        )

    base_value: float | None = None

    if base_values is not None:
        base_array = np.asarray(base_values)

        if base_array.ndim == 0:
            base_value = float(base_array)
        elif base_array.ndim == 1:
            base_value = float(base_array[-1])
        elif base_array.ndim >= 2:
            base_value = float(base_array.reshape(-1)[-1])

    return patient_values, base_value


def generate_shap_explanation(
    model: Any,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    feature_values: dict[str, float],
    background_dataset_path: str,
    top_feature_count: int = 10,
) -> dict[str, Any]:
    """
    Generate a local SHAP explanation for one prediction.

    For the current logistic-regression model, SHAP values are
    reported in the model's linear output space (log-odds).
    """

    try:
        import shap
    except ImportError as exc:
        raise ExplainabilityError(
            "The 'shap' package is not installed. Run "
            "'python -m pip install shap' inside the virtual environment."
        ) from exc

    background = _load_background_matrix(
        dataset_path=background_dataset_path,
        feature_names=feature_names,
    )

    try:
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(
                model,
                background,
                feature_perturbation="interventional",
            )
        else:
            masker = shap.maskers.Independent(background)
            explainer = shap.Explainer(
                model,
                masker=masker,
                feature_names=feature_names,
            )

        explanation = explainer(feature_matrix)

    except Exception as exc:
        raise ExplainabilityError(
            f"SHAP could not explain this prediction: {exc}"
        ) from exc

    shap_values, base_value = _extract_positive_class_values(
        explanation
    )

    if len(shap_values) != len(feature_names):
        raise ExplainabilityError(
            "The number of SHAP values does not match the model features."
        )

    records: list[dict[str, Any]] = []

    for feature, shap_value in zip(
        feature_names,
        shap_values,
    ):
        impact = float(shap_value)

        records.append(
            {
                "feature": feature,
                "feature_value": round(
                    float(feature_values[feature]),
                    6,
                ),
                "shap_value": round(impact, 6),
                "absolute_shap_value": round(
                    abs(impact),
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
        key=lambda item: item["absolute_shap_value"],
        reverse=True,
    )

    return {
        "available": True,
        "method": "SHAP",
        "explainer_type": type(explainer).__name__,
        "output_space": (
            "log_odds"
            if hasattr(model, "coef_")
            else "model_output"
        ),
        "base_value": (
            round(base_value, 6)
            if base_value is not None
            else None
        ),
        "top_contributions": records[:top_feature_count],
        "all_contribution_count": len(records),
        "background_dataset_path": str(
            Path(background_dataset_path)
        ),
        "background_patient_count": int(
            background.shape[0]
        ),
        "interpretation": (
            "Positive SHAP values move the prediction toward the "
            "positive class (Dead); negative values move it toward "
            "the negative class (Alive)."
        ),
    }
