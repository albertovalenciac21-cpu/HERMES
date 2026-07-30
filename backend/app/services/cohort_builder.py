from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

from backend.app.services.cohort import (
    discover_tcga_brca_cohort,
)
from backend.app.services.download_manager import (
    DownloadManagerError,
    download_gdc_file,
)
from backend.app.services.gdc import (
    get_tcga_brca_case,
)


COHORT_DIRECTORY = Path("data/cohort/tcga-brca")
DATASET_DIRECTORY = Path("data/datasets")

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


class CohortBuildError(RuntimeError):
    """Raised when the pilot cohort cannot be built."""


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


def _open_text_file(
    path: Path,
):
    """
    Open plain-text or gzip-compressed files.
    """

    if path.name.lower().endswith(".gz"):
        return gzip.open(
            path,
            mode="rt",
            encoding="utf-8",
            errors="replace",
        )

    return path.open(
        mode="r",
        encoding="utf-8",
        errors="replace",
    )


def _safe_number(
    value: str | None,
) -> float | None:
    """
    Convert text to a floating-point value when possible.
    """

    if value is None:
        return None

    cleaned = value.strip()

    if cleaned in {
        "",
        "NA",
        "N/A",
        "null",
        "None",
    }:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def _find_column(
    fieldnames: list[str],
    candidates: list[str],
) -> str | None:
    """
    Find the first matching column, ignoring capitalization.
    """

    lookup = {
        name.strip().lower(): name
        for name in fieldnames
        if name
    }

    for candidate in candidates:
        match = lookup.get(candidate.lower())

        if match:
            return match

    return None


def _extract_rna_expression(
    rna_path: Path,
    genes: list[str],
) -> dict[str, Any]:
    """
    Extract selected gene-expression values from a GDC
    STAR-count file.
    """

    requested_genes = set(genes)

    expression: dict[str, float | None] = {
        gene: None
        for gene in genes
    }

    with _open_text_file(rna_path) as input_file:
        data_lines = (
            line
            for line in input_file
            if not line.startswith("#")
        )

        reader = csv.DictReader(
            data_lines,
            delimiter="\t",
        )

        if not reader.fieldnames:
            raise CohortBuildError(
                f"No RNA headers were found in '{rna_path}'."
            )

        gene_column = _find_column(
            reader.fieldnames,
            [
                "gene_name",
                "gene_symbol",
                "symbol",
                "external_gene_name",
            ],
        )

        expression_column = _find_column(
            reader.fieldnames,
            [
                "tpm_unstranded",
                "tpm",
                "fpkm_unstranded",
                "fpkm",
                "unstranded",
            ],
        )

        if gene_column is None:
            raise CohortBuildError(
                "The RNA file does not contain a recognizable "
                "gene-symbol column."
            )

        if expression_column is None:
            raise CohortBuildError(
                "The RNA file does not contain a recognizable "
                "TPM or expression column."
            )

        for row in reader:
            gene_symbol = (
                row.get(gene_column)
                or ""
            ).strip().upper()

            if gene_symbol not in requested_genes:
                continue

            expression[gene_symbol] = _safe_number(
                row.get(expression_column)
            )

    found_genes = [
        gene
        for gene, value in expression.items()
        if value is not None
    ]

    return {
        "measurement_column": expression_column,
        "values": expression,
        "found_gene_count": len(found_genes),
        "missing_genes": [
            gene
            for gene in genes
            if expression[gene] is None
        ],
    }


def _extract_mutations(
    mutation_path: Path,
    genes: list[str],
) -> dict[str, Any]:
    """
    Parse a MAF file and create binary mutation indicators
    for the selected genes.
    """

    requested_genes = set(genes)

    mutation_status = {
        gene: 0
        for gene in genes
    }

    selected_mutations: list[dict[str, Any]] = []
    all_mutated_genes: set[str] = set()
    total_mutations = 0

    with _open_text_file(mutation_path) as input_file:
        data_lines = (
            line
            for line in input_file
            if not line.startswith("#")
        )

        reader = csv.DictReader(
            data_lines,
            delimiter="\t",
        )

        if not reader.fieldnames:
            raise CohortBuildError(
                f"No mutation headers were found in "
                f"'{mutation_path}'."
            )

        gene_column = _find_column(
            reader.fieldnames,
            [
                "Hugo_Symbol",
                "gene",
                "gene_symbol",
            ],
        )

        if gene_column is None:
            raise CohortBuildError(
                "The mutation file does not contain a "
                "recognizable gene-symbol column."
            )

        for row in reader:
            gene_symbol = (
                row.get(gene_column)
                or ""
            ).strip().upper()

            if not gene_symbol:
                continue

            total_mutations += 1
            all_mutated_genes.add(gene_symbol)

            if gene_symbol not in requested_genes:
                continue

            mutation_status[gene_symbol] = 1

            selected_mutations.append(
                {
                    "gene": gene_symbol,
                    "chromosome": row.get("Chromosome"),
                    "start_position": row.get(
                        "Start_Position"
                    ),
                    "end_position": row.get(
                        "End_Position"
                    ),
                    "reference_allele": row.get(
                        "Reference_Allele"
                    ),
                    "tumor_allele": (
                        row.get("Tumor_Seq_Allele2")
                        or row.get("Tumor_Allele")
                    ),
                    "variant_classification": row.get(
                        "Variant_Classification"
                    ),
                    "variant_type": row.get(
                        "Variant_Type"
                    ),
                    "protein_change": (
                        row.get("HGVSp_Short")
                        or row.get("HGVSp")
                    ),
                }
            )

    return {
        "total_mutation_count": total_mutations,
        "total_mutated_gene_count": len(
            all_mutated_genes
        ),
        "selected_gene_status": mutation_status,
        "selected_mutations": selected_mutations,
    }


def _first_item(
    value: Any,
) -> dict[str, Any]:
    """
    Return the first dictionary from a list.
    """

    if isinstance(value, list) and value:
        first = value[0]

        if isinstance(first, dict):
            return first

    return {}


def _get_clinical_record(
    submitter_id: str,
) -> dict[str, Any]:
    """
    Retrieve and simplify clinical information for one patient.
    """

    response = get_tcga_brca_case(
        submitter_id=submitter_id,
    )

    hits = (
        response
        .get("data", {})
        .get("hits", [])
    )

    if not hits:
        return {
            "available": False,
            "age_at_diagnosis_days": None,
            "age_at_diagnosis_years": None,
            "tumor_stage": None,
            "tumor_grade": None,
            "primary_diagnosis": None,
            "vital_status": None,
            "days_to_death": None,
            "days_to_last_follow_up": None,
        }

    case = hits[0]
    diagnosis = _first_item(case.get("diagnoses"))
    demographic = case.get("demographic") or {}

    age_at_diagnosis_days = diagnosis.get(
        "age_at_diagnosis"
    )

    age_at_diagnosis_years = None

    if isinstance(
        age_at_diagnosis_days,
        (int, float),
    ):
        age_at_diagnosis_years = round(
            age_at_diagnosis_days / 365.25,
            2,
        )

    return {
        "available": True,
        "age_at_diagnosis_days": (
            age_at_diagnosis_days
        ),
        "age_at_diagnosis_years": (
            age_at_diagnosis_years
        ),
        "tumor_stage": diagnosis.get("tumor_stage"),
        "tumor_grade": diagnosis.get("tumor_grade"),
        "primary_diagnosis": diagnosis.get(
            "primary_diagnosis"
        ),
        "vital_status": demographic.get(
            "vital_status"
        ),
        "days_to_death": demographic.get(
            "days_to_death"
        ),
        "days_to_last_follow_up": diagnosis.get(
            "days_to_last_follow_up"
        ),
    }


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """
    Write a formatted JSON file.
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


def _build_dataset_row(
    profile: dict[str, Any],
    genes: list[str],
) -> dict[str, Any]:
    """
    Flatten one patient profile into one ML-ready row.
    """

    clinical = profile["clinical"]
    rna_values = profile["rna_expression"]["values"]

    mutation_status = profile[
        "somatic_mutations"
    ]["selected_gene_status"]

    row: dict[str, Any] = {
        "patient_id": profile["patient"],
        "age_at_diagnosis_years": clinical.get(
            "age_at_diagnosis_years"
        ),
        "tumor_stage": clinical.get("tumor_stage"),
        "tumor_grade": clinical.get("tumor_grade"),
        "primary_diagnosis": clinical.get(
            "primary_diagnosis"
        ),
        "vital_status": clinical.get(
            "vital_status"
        ),
        "days_to_death": clinical.get(
            "days_to_death"
        ),
        "days_to_last_follow_up": clinical.get(
            "days_to_last_follow_up"
        ),
        "total_mutation_count": profile[
            "somatic_mutations"
        ].get("total_mutation_count"),
    }

    for gene in genes:
        row[f"{gene}_tpm"] = rna_values.get(gene)

        row[f"{gene}_mutated"] = (
            mutation_status.get(gene, 0)
        )

    return row


def _write_dataset_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Write flattened patient records to CSV.
    """

    if not rows:
        raise CohortBuildError(
            "No completed patient rows were available."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with output_path.open(
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


def _summarize_outcomes(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize vital-status labels in the completed cohort.
    """

    counts = {
        "alive": 0,
        "dead": 0,
        "unknown": 0,
    }

    for row in rows:
        status = str(
            row.get("vital_status") or ""
        ).strip().lower()

        if status == "alive":
            counts["alive"] += 1
        elif status == "dead":
            counts["dead"] += 1
        else:
            counts["unknown"] += 1

    labeled_count = counts["alive"] + counts["dead"]

    return {
        "counts": counts,
        "labeled_patient_count": labeled_count,
        "dead_fraction": (
            round(counts["dead"] / labeled_count, 4)
            if labeled_count
            else None
        ),
        "class_imbalance_warning": (
            labeled_count > 0
            and (
                counts["alive"] == 0
                or counts["dead"] == 0
                or min(
                    counts["alive"],
                    counts["dead"],
                ) < 5
            )
        ),
    }


def build_tcga_brca_cohort(
    target_count: int = 100,
    genes: list[str] | None = None,
    candidate_pool_size: int | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Build a scalable TCGA-BRCA machine-learning cohort.

    The builder continues through additional eligible candidates
    after individual failures until it reaches ``target_count`` or
    exhausts the discovered candidate pool.
    """

    if target_count < 1 or target_count > 500:
        raise ValueError(
            "The target cohort size must be between 1 and 500."
        )

    selected_genes = _normalize_genes(genes)

    if candidate_pool_size is None:
        candidate_pool_size = min(
            500,
            max(
                target_count + 50,
                target_count * 2,
            ),
        )

    if candidate_pool_size < target_count:
        raise ValueError(
            "candidate_pool_size must be greater than or equal "
            "to target_count."
        )

    if candidate_pool_size > 500:
        raise ValueError(
            "candidate_pool_size cannot exceed 500."
        )

    discovery = discover_tcga_brca_cohort(
        limit=candidate_pool_size,
        require_rna=True,
        require_mutations=True,
    )

    patients = discovery.get("patients", [])

    if not patients:
        raise CohortBuildError(
            "No eligible patients were discovered."
        )

    completed_profiles: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    attempted_patient_count = 0

    for patient in patients:
        if len(dataset_rows) >= target_count:
            break

        attempted_patient_count += 1
        submitter_id = patient["submitter_id"]

        selected_files = patient.get(
            "selected_files",
            {},
        )

        rna_file = selected_files.get(
            "rna_expression"
        )

        mutation_file = selected_files.get(
            "somatic_mutations"
        )

        try:
            if not rna_file or not mutation_file:
                raise CohortBuildError(
                    "The patient does not have both selected "
                    "molecular files."
                )

            patient_directory = (
                COHORT_DIRECTORY / submitter_id
            )

            rna_filename = (
                rna_file.get("file_name")
                or f"{rna_file['file_id']}.rna.tsv"
            )

            mutation_filename = (
                mutation_file.get("file_name")
                or f"{mutation_file['file_id']}.maf.gz"
            )

            rna_path = (
                patient_directory
                / "raw"
                / "rna"
                / rna_filename
            )

            mutation_path = (
                patient_directory
                / "raw"
                / "mutations"
                / mutation_filename
            )

            rna_download = download_gdc_file(
                file_id=rna_file["file_id"],
                destination=rna_path,
                expected_size=rna_file.get(
                    "file_size"
                ),
                expected_md5=rna_file.get("md5sum"),
            )

            mutation_download = download_gdc_file(
                file_id=mutation_file["file_id"],
                destination=mutation_path,
                expected_size=mutation_file.get(
                    "file_size"
                ),
                expected_md5=mutation_file.get(
                    "md5sum"
                ),
            )

            clinical = _get_clinical_record(
                submitter_id=submitter_id,
            )

            rna_expression = _extract_rna_expression(
                rna_path=rna_path,
                genes=selected_genes,
            )

            somatic_mutations = _extract_mutations(
                mutation_path=mutation_path,
                genes=selected_genes,
            )

            completed_number = len(dataset_rows) + 1

            profile = {
                "patient": submitter_id,
                "project": "TCGA-BRCA",
                "cohort_patient_number": completed_number,
                "profile_status": "complete",
                "clinical": clinical,
                "rna_expression": rna_expression,
                "somatic_mutations": somatic_mutations,
                "source_files": {
                    "rna_expression": {
                        **rna_file,
                        **rna_download,
                    },
                    "somatic_mutations": {
                        **mutation_file,
                        **mutation_download,
                    },
                },
                "selected_genes": selected_genes,
                "research_use_only": True,
            }

            profile_path = (
                patient_directory / "profile.json"
            )

            _write_json(
                path=profile_path,
                payload=profile,
            )

            completed_profiles.append(
                {
                    "patient": submitter_id,
                    "profile_path": str(profile_path),
                    "rna_downloaded_now": (
                        rna_download["downloaded_now"]
                    ),
                    "rna_resumed": rna_download[
                        "resumed_download"
                    ],
                    "rna_md5_verified": rna_download[
                        "md5_verified"
                    ],
                    "mutation_downloaded_now": (
                        mutation_download[
                            "downloaded_now"
                        ]
                    ),
                    "mutation_resumed": (
                        mutation_download[
                            "resumed_download"
                        ]
                    ),
                    "mutation_md5_verified": (
                        mutation_download[
                            "md5_verified"
                        ]
                    ),
                    "found_rna_genes": rna_expression[
                        "found_gene_count"
                    ],
                    "selected_mutation_count": len(
                        somatic_mutations[
                            "selected_mutations"
                        ]
                    ),
                    "vital_status": clinical.get(
                        "vital_status"
                    ),
                }
            )

            dataset_rows.append(
                _build_dataset_row(
                    profile=profile,
                    genes=selected_genes,
                )
            )

        except (
            DownloadManagerError,
            CohortBuildError,
            OSError,
            ValueError,
        ) as exc:
            failures.append(
                {
                    "patient": submitter_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "patient": submitter_id,
                    "error_type": type(exc).__name__,
                    "error": (
                        "Unexpected patient-processing error: "
                        f"{exc}"
                    ),
                }
            )

    completed_count = len(dataset_rows)

    safe_output_name = (
        output_name.strip()
        if output_name and output_name.strip()
        else (
            f"tcga_brca_cohort_"
            f"{completed_count}_patients"
        )
    )

    safe_output_name = safe_output_name.replace(
        ".csv",
        "",
    )

    dataset_path = (
        DATASET_DIRECTORY
        / f"{safe_output_name}.csv"
    )

    manifest_path = (
        DATASET_DIRECTORY
        / f"{safe_output_name}_manifest.json"
    )

    if dataset_rows:
        _write_dataset_csv(
            rows=dataset_rows,
            output_path=dataset_path,
        )

    outcome_summary = _summarize_outcomes(
        dataset_rows
    )

    reached_target = (
        completed_count >= target_count
    )

    manifest = {
        "project": "TCGA-BRCA",
        "build_status": (
            "complete"
            if reached_target and not failures
            else (
                "complete_with_failures"
                if reached_target
                else "partial"
            )
        ),
        "target_reached": reached_target,
        "download_engine": "curl_resumable",
        "requested_patient_count": target_count,
        "candidate_pool_size": candidate_pool_size,
        "discovered_candidate_count": len(patients),
        "attempted_patient_count": attempted_patient_count,
        "completed_patient_count": completed_count,
        "failed_patient_count": len(failures),
        "remaining_to_target": max(
            0,
            target_count - completed_count,
        ),
        "selected_genes": selected_genes,
        "outcome_summary": outcome_summary,
        "cohort_directory": str(
            COHORT_DIRECTORY
        ),
        "dataset_path": (
            str(dataset_path)
            if dataset_rows
            else None
        ),
        "patients": completed_profiles,
        "failures": failures,
        "research_use_only": True,
    }

    _write_json(
        path=manifest_path,
        payload=manifest,
    )

    manifest["manifest_path"] = str(
        manifest_path
    )

    return manifest


def build_tcga_brca_pilot_cohort(
    limit: int = 5,
    genes: list[str] | None = None,
) -> dict[str, Any]:
    """
    Backward-compatible wrapper for the original pilot endpoint.
    """

    if limit < 1 or limit > 25:
        raise ValueError(
            "The pilot cohort limit must be between 1 and 25."
        )

    return build_tcga_brca_cohort(
        target_count=limit,
        genes=genes,
        candidate_pool_size=min(
            100,
            max(limit * 2, limit + 10),
        ),
        output_name=(
            f"tcga_brca_pilot_{limit}_patients"
        ),
    )
