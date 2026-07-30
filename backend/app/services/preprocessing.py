from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from backend.app.services.dataset_validator import (
    DatasetValidationError,
    validate_tcga_brca_dataset,
)


DATASET_DIRECTORY = Path("data/datasets")
PROCESSED_DIRECTORY = Path("data/processed")

DEFAULT_DATASET_PATH = (
    DATASET_DIRECTORY / "tcga_brca_pilot_25_patients.csv"
)

DEFAULT_GENES = [
    "TP53",
    "BRCA1",
    "BRCA2",
    "ERBB2",
    "PIK3CA",
    "PTEN",
    "RB1",
    "ESR1",
    "PGR",
    "EGFR",
    "CD274",
    "MYC",
]

IDENTIFIER_COLUMN = "patient_id"

CATEGORICAL_FEATURE_COLUMNS = [
    "tumor_stage",
    "tumor_grade",
    "primary_diagnosis",
]

OUTCOME_REFERENCE_COLUMNS = [
    "vital_status",
    "days_to_death",
    "days_to_last_follow_up",
]

MISSING_CATEGORY = "__MISSING__"


class DatasetPreprocessingError(RuntimeError):
    """Raised when a cohort dataset cannot be preprocessed."""


def _normalize_genes(
    genes: list[str] | None,
) -> list[str]:
    """
    Normalize gene symbols and remove duplicates while
    preserving their original order.
    """

    selected = genes or DEFAULT_GENES
    normalized: list[str] = []
    seen: set[str] = set()

    for gene in selected:
        symbol = gene.strip().upper()

        if not symbol or symbol in seen:
            continue

        normalized.append(symbol)
        seen.add(symbol)

    if not normalized:
        raise ValueError(
            "At least one gene must be provided."
        )

    return normalized


def _is_missing(
    value: str | None,
) -> bool:
    """
    Determine whether a CSV value should be considered missing.
    """

    if value is None:
        return True

    cleaned = value.strip().lower()

    return cleaned in {
        "",
        "na",
        "n/a",
        "nan",
        "null",
        "none",
    }


def _parse_float(
    value: str | None,
) -> float | None:
    """
    Convert a CSV value to a finite float.

    Missing or non-finite values return None.
    """

    if _is_missing(value):
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetPreprocessingError(
            f"Value '{value}' is not numeric."
        ) from exc

    if math.isnan(parsed) or math.isinf(parsed):
        return None

    return parsed


def _read_dataset(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Read a cohort CSV file.
    """

    if not path.exists():
        raise DatasetPreprocessingError(
            f"The dataset was not found at '{path}'."
        )

    if path.stat().st_size == 0:
        raise DatasetPreprocessingError(
            f"The dataset at '{path}' is empty."
        )

    with path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if not reader.fieldnames:
            raise DatasetPreprocessingError(
                "The dataset does not contain column headers."
            )

        rows = list(reader)

    if not rows:
        raise DatasetPreprocessingError(
            "The dataset does not contain patient rows."
        )

    return list(reader.fieldnames), rows


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """
    Write the preprocessed dataset to CSV.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """
    Write preprocessing metadata as formatted JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def _safe_feature_name(
    value: str,
) -> str:
    """
    Convert a categorical value into a safe feature-name suffix.
    """

    cleaned = value.strip().lower()

    characters: list[str] = []

    for character in cleaned:
        if character.isalnum():
            characters.append(character)
        else:
            characters.append("_")

    safe_name = "".join(characters)

    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")

    safe_name = safe_name.strip("_")

    return safe_name or "missing"


def _median(
    values: list[float],
) -> float:
    """
    Calculate a numeric median with a safe all-missing fallback.
    """

    if not values:
        return 0.0

    return float(statistics.median(values))


def _mean(
    values: list[float],
) -> float:
    """
    Calculate an arithmetic mean.
    """

    if not values:
        return 0.0

    return float(statistics.fmean(values))


def _population_standard_deviation(
    values: list[float],
) -> float:
    """
    Calculate population standard deviation.

    A fallback of 1.0 prevents division by zero for constant
    columns.
    """

    if len(values) < 2:
        return 1.0

    standard_deviation = float(
        statistics.pstdev(values)
    )

    if standard_deviation == 0:
        return 1.0

    return standard_deviation


def _apply_log1p(
    value: float,
) -> float:
    """
    Apply a log1p transformation to a non-negative value.
    """

    if value < 0:
        raise DatasetPreprocessingError(
            "Log-transformed features cannot contain "
            "negative values."
        )

    return math.log1p(value)


def _build_numeric_feature_columns(
    genes: list[str],
) -> list[str]:
    """
    Return continuous columns used as model features.
    """

    return [
        "age_at_diagnosis_years",
        "total_mutation_count",
        *[
            f"{gene}_tpm"
            for gene in genes
        ],
    ]


def _build_mutation_feature_columns(
    genes: list[str],
) -> list[str]:
    """
    Return binary mutation columns used as model features.
    """

    return [
        f"{gene}_mutated"
        for gene in genes
    ]


def _prepare_numeric_features(
    rows: list[dict[str, str]],
    numeric_columns: list[str],
    log_transform_columns: set[str],
) -> tuple[
    dict[str, list[float]],
    dict[str, dict[str, float | bool]],
]:
    """
    Impute, optionally log-transform, and standardize numeric
    features.

    Returns standardized values and reusable parameters.
    """

    standardized_values: dict[str, list[float]] = {}
    parameters: dict[
        str,
        dict[str, float | bool],
    ] = {}

    for column in numeric_columns:
        parsed_values = [
            _parse_float(row.get(column))
            for row in rows
        ]

        observed_values = [
            value
            for value in parsed_values
            if value is not None
        ]

        median_value = _median(observed_values)

        imputed_values = [
            (
                value
                if value is not None
                else median_value
            )
            for value in parsed_values
        ]

        use_log_transform = (
            column in log_transform_columns
        )

        if use_log_transform:
            transformed_values = [
                _apply_log1p(value)
                for value in imputed_values
            ]
        else:
            transformed_values = imputed_values

        mean_value = _mean(transformed_values)

        standard_deviation = (
            _population_standard_deviation(
                transformed_values
            )
        )

        standardized_values[column] = [
            (
                value - mean_value
            ) / standard_deviation
            for value in transformed_values
        ]

        parameters[column] = {
            "median_imputation_value": median_value,
            "log1p_transformed": use_log_transform,
            "mean_after_transformation": mean_value,
            "standard_deviation_after_transformation": (
                standard_deviation
            ),
            "missing_value_count": sum(
                value is None
                for value in parsed_values
            ),
        }

    return standardized_values, parameters


def _prepare_categorical_features(
    rows: list[dict[str, str]],
    categorical_columns: list[str],
) -> tuple[
    dict[str, list[float]],
    dict[str, list[str]],
]:
    """
    One-hot encode categorical clinical variables.
    """

    encoded_features: dict[str, list[float]] = {}
    categories_by_column: dict[str, list[str]] = {}

    for column in categorical_columns:
        cleaned_values = [
            (
                MISSING_CATEGORY
                if _is_missing(row.get(column))
                else str(row.get(column)).strip()
            )
            for row in rows
        ]

        categories = sorted(
            set(cleaned_values),
            key=lambda value: value.lower(),
        )

        categories_by_column[column] = categories

        used_feature_names: set[str] = set()

        for category in categories:
            base_feature_name = (
                f"{column}__"
                f"{_safe_feature_name(category)}"
            )

            feature_name = base_feature_name
            suffix = 2

            while feature_name in used_feature_names:
                feature_name = (
                    f"{base_feature_name}_{suffix}"
                )
                suffix += 1

            used_feature_names.add(feature_name)

            encoded_features[feature_name] = [
                1.0 if value == category else 0.0
                for value in cleaned_values
            ]

    return encoded_features, categories_by_column


def _prepare_mutation_features(
    rows: list[dict[str, str]],
    mutation_columns: list[str],
) -> dict[str, list[int]]:
    """
    Preserve validated mutation indicators as binary values.
    """

    mutation_features: dict[str, list[int]] = {}

    for column in mutation_columns:
        values: list[int] = []

        for row in rows:
            parsed = _parse_float(
                row.get(column)
            )

            if parsed not in {0.0, 1.0}:
                raise DatasetPreprocessingError(
                    f"Column '{column}' contains a value "
                    "other than 0 or 1."
                )

            values.append(int(parsed))

        mutation_features[column] = values

    return mutation_features


def preprocess_cohort_dataset(
    dataset_path: str | None = None,
    output_path: str | None = None,
    metadata_path: str | None = None,
    genes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Convert any compatible validated cohort into an ML-ready
    feature matrix.

    Numeric clinical and molecular features are median-imputed
    and standardized. TPM values and total mutation count are
    log1p-transformed before standardization. Categorical
    clinical features are one-hot encoded. Binary mutation
    indicators remain 0 or 1.
    """

    selected_genes = _normalize_genes(genes)

    source_path = (
        Path(dataset_path)
        if dataset_path
        else DEFAULT_DATASET_PATH
    )

    validation = validate_tcga_brca_dataset(
        dataset_path=str(source_path),
        genes=selected_genes,
    )

    if not validation.get("valid_for_scaling"):
        raise DatasetPreprocessingError(
            "Dataset validation failed. Correct the reported "
            "errors before preprocessing."
        )

    fieldnames, rows = _read_dataset(
        source_path
    )

    required_identifier_columns = [
        IDENTIFIER_COLUMN,
    ]

    missing_identifiers = [
        column
        for column in required_identifier_columns
        if column not in fieldnames
    ]

    if missing_identifiers:
        raise DatasetPreprocessingError(
            "The dataset is missing required identifier "
            f"columns: {missing_identifiers}"
        )

    numeric_columns = (
        _build_numeric_feature_columns(
            selected_genes
        )
    )

    mutation_columns = (
        _build_mutation_feature_columns(
            selected_genes
        )
    )

    log_transform_columns = {
        "total_mutation_count",
        *[
            f"{gene}_tpm"
            for gene in selected_genes
        ],
    }

    numeric_features, numeric_parameters = (
        _prepare_numeric_features(
            rows=rows,
            numeric_columns=numeric_columns,
            log_transform_columns=(
                log_transform_columns
            ),
        )
    )

    categorical_features, categories = (
        _prepare_categorical_features(
            rows=rows,
            categorical_columns=(
                CATEGORICAL_FEATURE_COLUMNS
            ),
        )
    )

    mutation_features = _prepare_mutation_features(
        rows=rows,
        mutation_columns=mutation_columns,
    )

    numeric_feature_names = [
        f"{column}__scaled"
        for column in numeric_columns
    ]

    categorical_feature_names = list(
        categorical_features.keys()
    )

    mutation_feature_names = list(
        mutation_features.keys()
    )

    model_feature_names = [
        *numeric_feature_names,
        *categorical_feature_names,
        *mutation_feature_names,
    ]

    processed_rows: list[dict[str, Any]] = []

    for row_index, source_row in enumerate(rows):
        processed_row: dict[str, Any] = {
            IDENTIFIER_COLUMN: source_row.get(
                IDENTIFIER_COLUMN,
                "",
            ),
        }

        for column in OUTCOME_REFERENCE_COLUMNS:
            processed_row[column] = source_row.get(
                column,
                "",
            )

        for column in numeric_columns:
            processed_row[
                f"{column}__scaled"
            ] = round(
                numeric_features[column][row_index],
                8,
            )

        for feature_name, values in (
            categorical_features.items()
        ):
            processed_row[feature_name] = int(
                values[row_index]
            )

        for feature_name, values in (
            mutation_features.items()
        ):
            processed_row[feature_name] = (
                values[row_index]
            )

        processed_rows.append(processed_row)

    default_output_path = (
        PROCESSED_DIRECTORY
        / f"{source_path.stem}_ml_ready.csv"
    )

    resolved_output_path = (
        Path(output_path)
        if output_path
        else default_output_path
    )

    default_metadata_path = (
        PROCESSED_DIRECTORY
        / f"{source_path.stem}_preprocessing.json"
    )

    resolved_metadata_path = (
        Path(metadata_path)
        if metadata_path
        else default_metadata_path
    )

    output_fieldnames = [
        IDENTIFIER_COLUMN,
        *OUTCOME_REFERENCE_COLUMNS,
        *model_feature_names,
    ]

    _write_csv(
        path=resolved_output_path,
        rows=processed_rows,
        fieldnames=output_fieldnames,
    )

    metadata = {
        "project": "Project Trojan Horse",
        "cohort_name": source_path.stem,
        "preprocessing_version": "1.0",
        "source_dataset_path": str(source_path),
        "processed_dataset_path": str(
            resolved_output_path
        ),
        "row_count": len(processed_rows),
        "selected_genes": selected_genes,
        "identifier_column": IDENTIFIER_COLUMN,
        "outcome_reference_columns": (
            OUTCOME_REFERENCE_COLUMNS
        ),
        "model_feature_count": len(
            model_feature_names
        ),
        "model_feature_names": model_feature_names,
        "feature_groups": {
            "numeric_scaled": numeric_feature_names,
            "categorical_one_hot": (
                categorical_feature_names
            ),
            "binary_mutations": (
                mutation_feature_names
            ),
        },
        "numeric_parameters": numeric_parameters,
        "categorical_parameters": {
            "missing_category_value": (
                MISSING_CATEGORY
            ),
            "categories": categories,
        },
        "transformations": {
            "numeric_missing_values": (
                "median imputation"
            ),
            "age_at_diagnosis_years": (
                "median imputation followed by "
                "standardization"
            ),
            "total_mutation_count": (
                "median imputation, log1p transformation, "
                "and standardization"
            ),
            "rna_tpm": (
                "median imputation, log1p transformation, "
                "and standardization"
            ),
            "categorical_clinical_features": (
                "missing-category replacement and one-hot "
                "encoding"
            ),
            "mutation_features": (
                "preserved as binary 0/1 indicators"
            ),
        },
        "data_leakage_protection": {
            "excluded_from_model_features": (
                OUTCOME_REFERENCE_COLUMNS
            ),
            "reason": (
                "Survival and follow-up variables are retained "
                "for later outcome construction but are not "
                "included as predictors."
            ),
        },
        "research_use_only": True,
    }

    _write_json(
        path=resolved_metadata_path,
        payload=metadata,
    )

    return {
        "preprocessing_status": "complete",
        "source_dataset_path": str(source_path),
        "processed_dataset_path": str(
            resolved_output_path
        ),
        "metadata_path": str(
            resolved_metadata_path
        ),
        "patient_count": len(processed_rows),
        "original_column_count": len(fieldnames),
        "model_feature_count": len(
            model_feature_names
        ),
        "feature_groups": {
            "numeric_scaled_count": len(
                numeric_feature_names
            ),
            "categorical_one_hot_count": len(
                categorical_feature_names
            ),
            "binary_mutation_count": len(
                mutation_feature_names
            ),
        },
        "numeric_missing_values_imputed": {
            column: int(
                parameters["missing_value_count"]
            )
            for column, parameters
            in numeric_parameters.items()
        },
        "excluded_outcome_reference_columns": (
            OUTCOME_REFERENCE_COLUMNS
        ),
        "validation_status": validation.get(
            "validation_status"
        ),
        "ready_for_model_development": True,
        "important_note": (
            "This dataset is suitable for research pipeline development. "
            "Model performance must be interpreted according to cohort size, "
            "class balance, validation design, and external validation."
        ),
    }

def preprocess_tcga_brca_dataset(
    dataset_path: str | None = None,
    output_path: str | None = None,
    metadata_path: str | None = None,
    genes: list[str] | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper for older API routes."""

    return preprocess_cohort_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        metadata_path=metadata_path,
        genes=genes,
    )
