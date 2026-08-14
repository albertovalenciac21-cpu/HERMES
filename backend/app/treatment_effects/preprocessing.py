"""
HERMES 2.0 Transcriptomic Preprocessing and QC

This module performs quality control and reproducible preprocessing
for baseline transcriptomic data used in treatment-effect modeling.

Important:
NeoTRIP expression values are already:
    - TPM-derived
    - log2 transformed
    - ComBat corrected

Therefore this module does NOT re-normalize the data.

Instead, it performs:
    - structural integrity checks
    - missing-value checks
    - low-variance filtering
    - low-information filtering
    - patient-level alignment checks
    - feature summary generation

The goal is to construct a stable molecular input matrix X suitable
for downstream treatment-effect modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from backend.app.treatment_effects.cohort_loader import (
    NeoTRIPCohort,
    load_neotrip_baseline,
)


@dataclass
class TranscriptomicQCReport:
    n_patients: int
    n_genes_input: int
    n_genes_after_filtering: int
    n_missing_values: int
    n_constant_genes: int
    n_low_variance_genes: int
    n_low_information_genes: int
    variance_threshold: float
    expression_prevalence_threshold: float
    minimum_expression_value: float

    @property
    def genes_removed(self) -> int:
        return self.n_genes_input - self.n_genes_after_filtering

    @property
    def retention_fraction(self) -> float:
        if self.n_genes_input == 0:
            return 0.0

        return (
            self.n_genes_after_filtering
            / self.n_genes_input
        )


@dataclass
class ProcessedTranscriptome:
    expression: pd.DataFrame
    report: TranscriptomicQCReport
    retained_genes: List[str]
    removed_genes: List[str]


def validate_expression_matrix(
    expression: pd.DataFrame,
) -> None:
    """
    Validate structural integrity of the transcriptomic matrix.
    """

    if expression.empty:
        raise ValueError(
            "Expression matrix is empty."
        )

    if expression.index.duplicated().any():
        duplicates = (
            expression.index[
                expression.index.duplicated()
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate patient identifiers detected: "
            f"{duplicates[:10]}"
        )

    if expression.columns.duplicated().any():
        duplicates = (
            expression.columns[
                expression.columns.duplicated()
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            "Duplicate gene identifiers detected: "
            f"{duplicates[:10]}"
        )

    if expression.isna().any().any():
        n_missing = int(
            expression.isna().sum().sum()
        )

        raise ValueError(
            f"Expression matrix contains "
            f"{n_missing} missing values."
        )

    if not all(
        np.issubdtype(dtype, np.number)
        for dtype in expression.dtypes
    ):
        raise TypeError(
            "All expression columns must be numeric."
        )

    values = expression.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "Expression matrix contains "
            "non-finite values."
        )


def calculate_gene_qc(
    expression: pd.DataFrame,
    variance_threshold: float = 0.01,
    expression_prevalence_threshold: float = 0.10,
    minimum_expression_value: float = 0.0,
) -> pd.DataFrame:
    """
    Calculate gene-level quality-control metrics.

    Parameters
    ----------
    variance_threshold:
        Genes with variance below this threshold are considered
        low-variance.

    expression_prevalence_threshold:
        Minimum fraction of patients in which a gene must have
        expression above minimum_expression_value.

    minimum_expression_value:
        Threshold used when calculating expression prevalence.
    """

    validate_expression_matrix(expression)

    variance = expression.var(
        axis=0,
        ddof=1,
    )

    mean_expression = expression.mean(
        axis=0,
    )

    median_expression = expression.median(
        axis=0,
    )

    prevalence = (
        expression.gt(
            minimum_expression_value
        )
        .mean(axis=0)
    )

    constant = variance.eq(0.0)

    low_variance = variance.lt(
        variance_threshold
    )

    low_information = prevalence.lt(
        expression_prevalence_threshold
    )

    qc = pd.DataFrame(
        {
            "mean_expression": mean_expression,
            "median_expression": median_expression,
            "variance": variance,
            "expression_prevalence": prevalence,
            "constant": constant,
            "low_variance": low_variance,
            "low_information": low_information,
        }
    )

    qc.index.name = "gene"

    return qc


def filter_transcriptome(
    expression: pd.DataFrame,
    variance_threshold: float = 0.01,
    expression_prevalence_threshold: float = 0.10,
    minimum_expression_value: float = 0.0,
) -> ProcessedTranscriptome:
    """
    Apply unsupervised transcriptomic QC filtering.

    Critically, no treatment or outcome labels are used here.
    This avoids treatment/outcome-driven feature leakage.
    """

    qc = calculate_gene_qc(
        expression=expression,
        variance_threshold=variance_threshold,
        expression_prevalence_threshold=(
            expression_prevalence_threshold
        ),
        minimum_expression_value=(
            minimum_expression_value
        ),
    )

    remove_mask = (
        qc["constant"]
        | qc["low_variance"]
        | qc["low_information"]
    )

    retained_genes = (
        qc.index[~remove_mask]
        .astype(str)
        .tolist()
    )

    removed_genes = (
        qc.index[remove_mask]
        .astype(str)
        .tolist()
    )

    filtered = expression.loc[
        :,
        retained_genes,
    ].copy()

    report = TranscriptomicQCReport(
        n_patients=expression.shape[0],
        n_genes_input=expression.shape[1],
        n_genes_after_filtering=(
            filtered.shape[1]
        ),
        n_missing_values=int(
            expression.isna().sum().sum()
        ),
        n_constant_genes=int(
            qc["constant"].sum()
        ),
        n_low_variance_genes=int(
            qc["low_variance"].sum()
        ),
        n_low_information_genes=int(
            qc["low_information"].sum()
        ),
        variance_threshold=variance_threshold,
        expression_prevalence_threshold=(
            expression_prevalence_threshold
        ),
        minimum_expression_value=(
            minimum_expression_value
        ),
    )

    return ProcessedTranscriptome(
        expression=filtered,
        report=report,
        retained_genes=retained_genes,
        removed_genes=removed_genes,
    )


def validate_cohort_alignment(
    cohort: NeoTRIPCohort,
    processed: ProcessedTranscriptome,
) -> None:
    """
    Ensure processed expression remains aligned to the
    canonical HERMES patient ordering.
    """

    cohort_ids = [
        record.patient_id
        for record in cohort.records
    ]

    expression_ids = (
        processed.expression.index
        .astype(str)
        .tolist()
    )

    if cohort_ids != expression_ids:
        raise ValueError(
            "Processed expression matrix is not aligned "
            "with HERMES patient records."
        )


def summarize_processed_transcriptome(
    processed: ProcessedTranscriptome,
) -> Dict[str, float]:
    """
    Return compact QC summary statistics.
    """

    report = processed.report

    return {
        "patients": report.n_patients,
        "genes_input": report.n_genes_input,
        "genes_retained": (
            report.n_genes_after_filtering
        ),
        "genes_removed": report.genes_removed,
        "retention_fraction": (
            report.retention_fraction
        ),
        "constant_genes": (
            report.n_constant_genes
        ),
        "low_variance_genes": (
            report.n_low_variance_genes
        ),
        "low_information_genes": (
            report.n_low_information_genes
        ),
    }


def preprocess_neotrip_baseline(
    variance_threshold: float = 0.01,
    expression_prevalence_threshold: float = 0.10,
    minimum_expression_value: float = 0.0,
) -> ProcessedTranscriptome:
    """
    Load and preprocess the NeoTRIP baseline transcriptome.
    """

    cohort = load_neotrip_baseline()

    processed = filter_transcriptome(
        expression=cohort.expression,
        variance_threshold=variance_threshold,
        expression_prevalence_threshold=(
            expression_prevalence_threshold
        ),
        minimum_expression_value=(
            minimum_expression_value
        ),
    )

    validate_cohort_alignment(
        cohort,
        processed,
    )

    return processed


if __name__ == "__main__":
    processed = preprocess_neotrip_baseline()

    summary = summarize_processed_transcriptome(
        processed
    )

    print(
        "=== HERMES 2.0 Transcriptomic QC ==="
    )

    print(
        f"Patients: "
        f"{summary['patients']}"
    )

    print(
        f"Genes input: "
        f"{summary['genes_input']}"
    )

    print(
        f"Genes retained: "
        f"{summary['genes_retained']}"
    )

    print(
        f"Genes removed: "
        f"{summary['genes_removed']}"
    )

    print(
        f"Retention fraction: "
        f"{summary['retention_fraction']:.4f}"
    )

    print(
        f"Constant genes: "
        f"{summary['constant_genes']}"
    )

    print(
        f"Low-variance genes: "
        f"{summary['low_variance_genes']}"
    )

    print(
        f"Low-information genes: "
        f"{summary['low_information_genes']}"
    )