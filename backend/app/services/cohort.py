from __future__ import annotations

from collections import defaultdict
from typing import Any

import requests


GDC_BASE_URL = "https://api.gdc.cancer.gov"
REQUEST_TIMEOUT = 60

TCGA_BRCA_PROJECT = "TCGA-BRCA"

RNA_DATA_CATEGORY = "Transcriptome Profiling"
RNA_DATA_TYPE = "Gene Expression Quantification"

MUTATION_DATA_CATEGORY = "Simple Nucleotide Variation"
MUTATION_DATA_TYPE = "Masked Somatic Mutation"


class CohortDiscoveryError(RuntimeError):
    """Raised when the TCGA-BRCA cohort cannot be discovered."""


def _request_gdc_files(
    filters: dict[str, Any],
    fields: list[str],
    size: int = 10000,
) -> list[dict[str, Any]]:
    """
    Query the GDC files endpoint and return matching file records.
    """

    response = requests.get(
        f"{GDC_BASE_URL}/files",
        params={
            "filters": __import__("json").dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": size,
        },
        timeout=REQUEST_TIMEOUT,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise CohortDiscoveryError(
            "The GDC files request failed with status "
            f"{response.status_code}: {response.text[:500]}"
        ) from exc

    payload = response.json()

    return payload.get("data", {}).get("hits", [])


def _build_file_filters(
    data_category: str,
    data_type: str,
) -> dict[str, Any]:
    """
    Build filters for open-access TCGA-BRCA files.
    """

    return {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [TCGA_BRCA_PROJECT],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": [data_category],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": [data_type],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "access",
                    "value": ["open"],
                },
            },
        ],
    }


def _extract_case_records(
    files: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group GDC file records by patient submitter ID.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for file_record in files:
        cases = file_record.get("cases") or []

        for case in cases:
            submitter_id = case.get("submitter_id")

            if not submitter_id:
                continue

            grouped[submitter_id].append(
                {
                    "file_id": file_record.get("file_id"),
                    "file_name": file_record.get("file_name"),
                    "data_category": file_record.get(
                        "data_category"
                    ),
                    "data_type": file_record.get("data_type"),
                    "experimental_strategy": file_record.get(
                        "experimental_strategy"
                    ),
                    "workflow_type": (
                        file_record.get("analysis") or {}
                    ).get("workflow_type"),
                    "access": file_record.get("access"),
                    "file_size": file_record.get("file_size"),
                    "md5sum": file_record.get("md5sum"),
                    "case_id": case.get("case_id"),
                    "submitter_id": submitter_id,
                }
            )

    return dict(grouped)


def get_tcga_brca_rna_files() -> dict[str, list[dict[str, Any]]]:
    """
    Retrieve open-access TCGA-BRCA RNA-expression files grouped
    by patient.
    """

    filters = _build_file_filters(
        data_category=RNA_DATA_CATEGORY,
        data_type=RNA_DATA_TYPE,
    )

    fields = [
        "file_id",
        "file_name",
        "data_category",
        "data_type",
        "experimental_strategy",
        "analysis.workflow_type",
        "access",
        "file_size",
        "md5sum",
        "cases.case_id",
        "cases.submitter_id",
    ]

    files = _request_gdc_files(
        filters=filters,
        fields=fields,
    )

    return _extract_case_records(files)


def get_tcga_brca_mutation_files(
) -> dict[str, list[dict[str, Any]]]:
    """
    Retrieve open-access TCGA-BRCA masked somatic-mutation files
    grouped by patient.
    """

    filters = _build_file_filters(
        data_category=MUTATION_DATA_CATEGORY,
        data_type=MUTATION_DATA_TYPE,
    )

    fields = [
        "file_id",
        "file_name",
        "data_category",
        "data_type",
        "experimental_strategy",
        "analysis.workflow_type",
        "access",
        "file_size",
        "md5sum",
        "cases.case_id",
        "cases.submitter_id",
    ]

    files = _request_gdc_files(
        filters=filters,
        fields=fields,
    )

    return _extract_case_records(files)


def _choose_preferred_rna_file(
    files: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Choose one preferred RNA-expression file for a patient.

    STAR - Counts is preferred when available.
    """

    if not files:
        return None

    workflow_priority = {
        "STAR - Counts": 0,
        "HTSeq - Counts": 1,
        "HTSeq - FPKM-UQ": 2,
        "HTSeq - FPKM": 3,
    }

    ordered_files = sorted(
        files,
        key=lambda item: workflow_priority.get(
            item.get("workflow_type"),
            99,
        ),
    )

    return ordered_files[0]


def _choose_preferred_mutation_file(
    files: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Choose one preferred masked somatic-mutation file.

    The GDC aliquot ensemble file is preferred when available.
    """

    if not files:
        return None

    ensemble_files = [
        file_record
        for file_record in files
        if "aliquot_ensemble_masked.maf" in (
            file_record.get("file_name") or ""
        )
    ]

    if ensemble_files:
        return ensemble_files[0]

    return files[0]


def discover_tcga_brca_cohort(
    limit: int = 25,
    require_rna: bool = True,
    require_mutations: bool = True,
) -> dict[str, Any]:
    """
    Discover TCGA-BRCA patients with matching molecular files.

    This function only discovers metadata. It does not download the
    RNA-expression or mutation files.
    """

    if limit < 1:
        raise ValueError("The cohort limit must be at least 1.")

    rna_by_patient = get_tcga_brca_rna_files()
    mutation_by_patient = get_tcga_brca_mutation_files()

    all_patient_ids = sorted(
        set(rna_by_patient) | set(mutation_by_patient)
    )

    eligible_patients: list[dict[str, Any]] = []

    for submitter_id in all_patient_ids:
        rna_files = rna_by_patient.get(submitter_id, [])
        mutation_files = mutation_by_patient.get(
            submitter_id,
            [],
        )

        has_rna = bool(rna_files)
        has_mutations = bool(mutation_files)

        if require_rna and not has_rna:
            continue

        if require_mutations and not has_mutations:
            continue

        preferred_rna = _choose_preferred_rna_file(rna_files)
        preferred_mutation = _choose_preferred_mutation_file(
            mutation_files
        )

        eligible_patients.append(
            {
                "submitter_id": submitter_id,
                "available_modalities": {
                    "rna_expression": has_rna,
                    "somatic_mutations": has_mutations,
                },
                "selected_files": {
                    "rna_expression": preferred_rna,
                    "somatic_mutations": preferred_mutation,
                },
                "file_counts": {
                    "rna_expression": len(rna_files),
                    "somatic_mutations": len(
                        mutation_files
                    ),
                },
            }
        )

    selected_patients = eligible_patients[:limit]

    return {
        "project": TCGA_BRCA_PROJECT,
        "discovery_status": "complete",
        "downloads_performed": False,
        "requirements": {
            "rna_expression": require_rna,
            "somatic_mutations": require_mutations,
        },
        "counts": {
            "patients_with_rna": len(rna_by_patient),
            "patients_with_mutations": len(
                mutation_by_patient
            ),
            "eligible_patients": len(eligible_patients),
            "returned_patients": len(selected_patients),
        },
        "patients": selected_patients,
        "next_step": (
            "Download the selected files and transform each patient "
            "into a standardized machine-learning feature record."
        ),
    }