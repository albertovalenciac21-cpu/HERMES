from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


DATASET_DIRECTORY = Path("data/datasets")

DEFAULT_DATASET_PATH = (
    DATASET_DIRECTORY / "tcga_brca_pilot_5_patients.csv"
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


class DatasetValidationError(RuntimeError):
    """Raised when a cohort dataset cannot be validated."""


def _is_missing(value: str | None) -> bool:
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
    Convert a CSV value to float.

    Missing values return None.
    Invalid non-missing values raise ValueError.
    """

    if _is_missing(value):
        return None

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Value '{value}' is not numeric."
        ) from exc

    if math.isnan(parsed) or math.isinf(parsed):
        return None

    return parsed


def _expected_columns(
    genes: list[str],
) -> list[str]:
    """
    Return the expected dataset columns.
    """

    columns = [
        "patient_id",
        "age_at_diagnosis_years",
        "tumor_stage",
        "tumor_grade",
        "primary_diagnosis",
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
        "total_mutation_count",
    ]

    for gene in genes:
        columns.append(f"{gene}_tpm")
        columns.append(f"{gene}_mutated")

    return columns


def _read_csv(
    dataset_path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    """
    Read a CSV file and return headers and rows.
    """

    if not dataset_path.exists():
        raise DatasetValidationError(
            f"The dataset was not found at '{dataset_path}'."
        )

    if dataset_path.stat().st_size == 0:
        raise DatasetValidationError(
            f"The dataset at '{dataset_path}' is empty."
        )

    with dataset_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as input_file:
        reader = csv.DictReader(input_file)

        if not reader.fieldnames:
            raise DatasetValidationError(
                "The dataset does not contain column headers."
            )

        rows = list(reader)

    return list(reader.fieldnames), rows


def validate_tcga_brca_dataset(
    dataset_path: str | None = None,
    genes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Validate a TCGA-BRCA machine-learning dataset.
    """

    selected_genes = [
        gene.strip().upper()
        for gene in (genes or DEFAULT_GENES)
        if gene.strip()
    ]

    path = (
        Path(dataset_path)
        if dataset_path
        else DEFAULT_DATASET_PATH
    )

    fieldnames, rows = _read_csv(path)

    expected_columns = _expected_columns(selected_genes)

    missing_columns = [
        column
        for column in expected_columns
        if column not in fieldnames
    ]

    unexpected_columns = [
        column
        for column in fieldnames
        if column not in expected_columns
    ]

    patient_ids: list[str] = []
    duplicate_patient_ids: list[str] = []
    seen_patient_ids: set[str] = set()

    invalid_numeric_values: list[dict[str, Any]] = []
    invalid_mutation_values: list[dict[str, Any]] = []

    missing_counts = {
        column: 0
        for column in fieldnames
    }

    numeric_columns = [
        "age_at_diagnosis_years",
        "days_to_death",
        "days_to_last_follow_up",
        "total_mutation_count",
    ]

    rna_columns = [
        f"{gene}_tpm"
        for gene in selected_genes
    ]

    mutation_columns = [
        f"{gene}_mutated"
        for gene in selected_genes
    ]

    numeric_columns.extend(rna_columns)

    complete_rna_patients = 0
    complete_mutation_patients = 0

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        patient_id = (
            row.get("patient_id")
            or ""
        ).strip()

        if patient_id:
            patient_ids.append(patient_id)

            if patient_id in seen_patient_ids:
                duplicate_patient_ids.append(patient_id)

            seen_patient_ids.add(patient_id)

        for column in fieldnames:
            if _is_missing(row.get(column)):
                missing_counts[column] += 1

        for column in numeric_columns:
            if column not in fieldnames:
                continue

            value = row.get(column)

            try:
                _parse_float(value)
            except ValueError:
                invalid_numeric_values.append(
                    {
                        "row": row_number,
                        "patient_id": patient_id,
                        "column": column,
                        "value": value,
                    }
                )

        patient_rna_complete = True

        for column in rna_columns:
            if column not in fieldnames:
                patient_rna_complete = False
                continue

            value = row.get(column)

            if _is_missing(value):
                patient_rna_complete = False
                continue

            try:
                _parse_float(value)
            except ValueError:
                patient_rna_complete = False

        if patient_rna_complete:
            complete_rna_patients += 1

        patient_mutations_complete = True

        for column in mutation_columns:
            if column not in fieldnames:
                patient_mutations_complete = False
                continue

            value = (
                row.get(column)
                or ""
            ).strip()

            if value not in {"0", "1", "0.0", "1.0"}:
                patient_mutations_complete = False

                invalid_mutation_values.append(
                    {
                        "row": row_number,
                        "patient_id": patient_id,
                        "column": column,
                        "value": value,
                    }
                )

        if patient_mutations_complete:
            complete_mutation_patients += 1

    row_count = len(rows)

    missing_percentages: dict[str, float] = {}

    for column, missing_count in missing_counts.items():
        if row_count == 0:
            percentage = 0.0
        else:
            percentage = round(
                missing_count / row_count * 100,
                2,
            )

        missing_percentages[column] = percentage

    warnings: list[str] = []
    errors: list[str] = []

    if row_count == 0:
        errors.append(
            "The dataset contains no patient rows."
        )

    if missing_columns:
        errors.append(
            "Required columns are missing."
        )

    if duplicate_patient_ids:
        errors.append(
            "Duplicate patient IDs were detected."
        )

    if invalid_numeric_values:
        errors.append(
            "Invalid values were detected in numeric columns."
        )

    if invalid_mutation_values:
        errors.append(
            "Mutation columns contain values other than 0 or 1."
        )

    if missing_counts.get("patient_id", 0) > 0:
        errors.append(
            "One or more rows are missing a patient ID."
        )

    clinical_columns = [
        "age_at_diagnosis_years",
        "tumor_stage",
        "tumor_grade",
        "primary_diagnosis",
        "vital_status",
        "days_to_death",
        "days_to_last_follow_up",
    ]

    high_missing_clinical_columns = [
        column
        for column in clinical_columns
        if missing_percentages.get(column, 0) >= 40
    ]

    if high_missing_clinical_columns:
        warnings.append(
            "Some clinical columns have at least 40% "
            "missing values."
        )

    if complete_rna_patients < row_count:
        warnings.append(
            "At least one patient has an incomplete RNA "
            "feature set."
        )

    if complete_mutation_patients < row_count:
        warnings.append(
            "At least one patient has an incomplete mutation "
            "feature set."
        )

    valid_for_scaling = not errors

    return {
        "dataset_path": str(path),
        "validation_status": (
            "passed"
            if valid_for_scaling
            else "failed"
        ),
        "valid_for_scaling": valid_for_scaling,
        "dataset_summary": {
            "row_count": row_count,
            "column_count": len(fieldnames),
            "unique_patient_count": len(
                set(patient_ids)
            ),
            "complete_rna_patient_count": (
                complete_rna_patients
            ),
            "complete_mutation_patient_count": (
                complete_mutation_patients
            ),
        },
        "column_validation": {
            "expected_column_count": len(
                expected_columns
            ),
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
        },
        "patient_validation": {
            "duplicate_patient_ids": sorted(
                set(duplicate_patient_ids)
            ),
            "missing_patient_id_count": (
                missing_counts.get("patient_id", 0)
            ),
        },
        "value_validation": {
            "invalid_numeric_values": (
                invalid_numeric_values
            ),
            "invalid_mutation_values": (
                invalid_mutation_values
            ),
        },
        "missing_data": {
            "counts": missing_counts,
            "percentages": missing_percentages,
            "high_missing_clinical_columns": (
                high_missing_clinical_columns
            ),
        },
        "errors": errors,
        "warnings": warnings,
        "recommended_next_step": (
            "Scale the pilot cohort to 25 patients."
            if valid_for_scaling
            else (
                "Correct the reported dataset errors before "
                "scaling the cohort."
            )
        ),
    }