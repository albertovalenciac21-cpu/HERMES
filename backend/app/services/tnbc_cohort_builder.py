from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from backend.app.services.cohort import (
    CohortDiscoveryError,
    _choose_preferred_mutation_file,
    _choose_preferred_rna_file,
    get_tcga_brca_mutation_files,
    get_tcga_brca_rna_files,
)
from backend.app.services.cohort_builder import (
    DEFAULT_GENES,
    CohortBuildError,
    _build_dataset_row,
    _extract_rna_expression,
    _find_column,
    _get_clinical_record,
    _normalize_genes,
    _open_text_file,
    _summarize_outcomes,
    _write_dataset_csv,
    _write_json,
)
from backend.app.services.download_manager import (
    DownloadManagerError,
    download_gdc_file,
)
from backend.app.services.tnbc_discovery import (
    TNBCDiscoveryError,
    scan_tcga_brca_receptor_supplements,
)


TNBC_COHORT_DIRECTORY = Path("data/cohort/tcga-tnbc")
DATASET_DIRECTORY = Path("data/datasets")
SHARED_MUTATION_DIRECTORY = TNBC_COHORT_DIRECTORY / "_shared" / "mutations"


class TNBCCohortBuildError(RuntimeError):
    """Raised when a TNBC-specific molecular cohort cannot be built."""


def _extract_patient_mutations(
    mutation_path: Path,
    patient_id: str,
    genes: list[str],
) -> dict[str, Any]:
    """
    Extract mutation features for one patient from a MAF file.

    Some GDC masked-MAF files contain records for many TCGA cases. The
    Tumor_Sample_Barcode column must therefore be filtered by patient ID;
    otherwise every patient could incorrectly receive cohort-wide mutation
    features.
    """

    requested_genes = set(genes)
    normalized_patient_id = patient_id.strip().upper()

    mutation_status = {gene: 0 for gene in genes}
    selected_mutations: list[dict[str, Any]] = []
    all_mutated_genes: set[str] = set()
    total_mutations = 0

    with _open_text_file(mutation_path) as input_file:
        data_lines = (
            line
            for line in input_file
            if not line.startswith("#")
        )

        reader = csv.DictReader(data_lines, delimiter="\t")

        if not reader.fieldnames:
            raise TNBCCohortBuildError(
                f"No mutation headers were found in '{mutation_path}'."
            )

        gene_column = _find_column(
            reader.fieldnames,
            ["Hugo_Symbol", "gene", "gene_symbol"],
        )
        sample_column = _find_column(
            reader.fieldnames,
            [
                "Tumor_Sample_Barcode",
                "tumor_sample_barcode",
                "Tumor_Aliquot_Barcode",
                "case_submitter_id",
                "submitter_id",
            ],
        )

        if gene_column is None:
            raise TNBCCohortBuildError(
                "The mutation file does not contain a recognizable "
                "gene-symbol column."
            )

        if sample_column is None:
            raise TNBCCohortBuildError(
                "The mutation file does not contain a recognizable tumor "
                "sample barcode column, so patient-specific filtering is "
                "not possible."
            )

        for row in reader:
            sample_barcode = str(row.get(sample_column) or "").strip().upper()

            # TCGA sample and aliquot barcodes begin with the 12-character
            # patient barcode, for example TCGA-XX-YYYY-01A....
            if not sample_barcode.startswith(normalized_patient_id):
                continue

            gene_symbol = str(row.get(gene_column) or "").strip().upper()

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
                    "tumor_sample_barcode": sample_barcode,
                    "chromosome": row.get("Chromosome"),
                    "start_position": row.get("Start_Position"),
                    "end_position": row.get("End_Position"),
                    "reference_allele": row.get("Reference_Allele"),
                    "tumor_allele": (
                        row.get("Tumor_Seq_Allele2")
                        or row.get("Tumor_Allele")
                    ),
                    "variant_classification": row.get(
                        "Variant_Classification"
                    ),
                    "variant_type": row.get("Variant_Type"),
                    "protein_change": (
                        row.get("HGVSp_Short") or row.get("HGVSp")
                    ),
                }
            )

    return {
        "patient_filter_applied": normalized_patient_id,
        "total_mutation_count": total_mutations,
        "total_mutated_gene_count": len(all_mutated_genes),
        "selected_gene_status": mutation_status,
        "selected_mutations": selected_mutations,
    }


def _sanitize_output_name(output_name: str | None, completed_count: int) -> str:
    name = (
        output_name.strip()
        if output_name and output_name.strip()
        else f"tcga_tnbc_cohort_{completed_count}_patients"
    )

    for suffix in (".csv", ".json"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]

    safe = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in name
    ).strip("_")

    if not safe:
        raise ValueError("output_name must contain at least one valid character.")

    return safe


def build_tcga_tnbc_cohort(
    target_count: int | None = None,
    genes: list[str] | None = None,
    clinical_file_limit: int = 1000,
    output_name: str | None = None,
) -> dict[str, Any]:
    """
    Build a molecular cohort restricted to confirmed TCGA-BRCA TNBC cases.

    TNBC status is defined here as ER-negative, PR-negative, and HER2-negative
    in the scanned TCGA clinical supplements. RNA and mutation metadata are
    then intersected with those confirmed patient IDs.
    """

    if clinical_file_limit < 1 or clinical_file_limit > 1000:
        raise ValueError("clinical_file_limit must be between 1 and 1000.")

    if target_count is not None and (target_count < 1 or target_count > 500):
        raise ValueError("target_count must be between 1 and 500 when provided.")

    selected_genes = _normalize_genes(genes or DEFAULT_GENES)

    receptor_scan = scan_tcga_brca_receptor_supplements(
        limit=clinical_file_limit
    )

    confirmed_records = receptor_scan.get("confirmed_tnbc_patients", [])
    receptor_by_patient: dict[str, dict[str, Any]] = {
        str(record.get("patient_id") or "").strip().upper(): record
        for record in confirmed_records
        if str(record.get("patient_id") or "").strip()
    }

    confirmed_ids = sorted(receptor_by_patient)

    if not confirmed_ids:
        raise TNBCCohortBuildError(
            "No confirmed TNBC patients were found in the receptor scan."
        )

    rna_by_patient = get_tcga_brca_rna_files()
    mutations_by_patient = get_tcga_brca_mutation_files()

    eligible_ids = [
        patient_id
        for patient_id in confirmed_ids
        if patient_id in rna_by_patient
        and patient_id in mutations_by_patient
    ]

    missing_rna_ids = [
        patient_id
        for patient_id in confirmed_ids
        if patient_id not in rna_by_patient
    ]
    missing_mutation_ids = [
        patient_id
        for patient_id in confirmed_ids
        if patient_id not in mutations_by_patient
    ]

    requested_count = target_count or len(eligible_ids)
    candidate_ids = eligible_ids[:requested_count]

    completed_profiles: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    mutation_download_cache: dict[str, dict[str, Any]] = {}

    for patient_id in candidate_ids:
        try:
            rna_file = _choose_preferred_rna_file(rna_by_patient[patient_id])
            mutation_file = _choose_preferred_mutation_file(
                mutations_by_patient[patient_id]
            )

            if not rna_file or not mutation_file:
                raise TNBCCohortBuildError(
                    "The patient does not have both selected molecular files."
                )

            patient_directory = TNBC_COHORT_DIRECTORY / patient_id

            rna_filename = (
                rna_file.get("file_name")
                or f"{rna_file['file_id']}.rna.tsv"
            )
            mutation_filename = (
                mutation_file.get("file_name")
                or f"{mutation_file['file_id']}.maf.gz"
            )

            rna_path = patient_directory / "raw" / "rna" / rna_filename
            mutation_path = SHARED_MUTATION_DIRECTORY / mutation_filename

            rna_download = download_gdc_file(
                file_id=rna_file["file_id"],
                destination=rna_path,
                expected_size=rna_file.get("file_size"),
                expected_md5=rna_file.get("md5sum"),
            )

            mutation_file_id = str(mutation_file["file_id"])
            mutation_download = mutation_download_cache.get(mutation_file_id)

            if mutation_download is None:
                mutation_download = download_gdc_file(
                    file_id=mutation_file_id,
                    destination=mutation_path,
                    expected_size=mutation_file.get("file_size"),
                    expected_md5=mutation_file.get("md5sum"),
                )
                mutation_download_cache[mutation_file_id] = mutation_download

            clinical = _get_clinical_record(submitter_id=patient_id)
            rna_expression = _extract_rna_expression(
                rna_path=rna_path,
                genes=selected_genes,
            )
            somatic_mutations = _extract_patient_mutations(
                mutation_path=mutation_path,
                patient_id=patient_id,
                genes=selected_genes,
            )

            receptor_record = receptor_by_patient[patient_id]
            receptor_status = receptor_record.get("receptor_status") or {}

            profile = {
                "patient": patient_id,
                "project": "TCGA-BRCA",
                "cohort": "confirmed_tnbc",
                "cohort_patient_number": len(dataset_rows) + 1,
                "profile_status": "complete",
                "tnbc_definition": (
                    "ER-negative, PR-negative, and HER2-negative in "
                    "TCGA clinical supplements"
                ),
                "receptor_status": receptor_status,
                "receptor_source_file_count": receptor_record.get(
                    "source_file_count"
                ),
                "clinical": clinical,
                "rna_expression": rna_expression,
                "somatic_mutations": somatic_mutations,
                "source_files": {
                    "rna_expression": {**rna_file, **rna_download},
                    "somatic_mutations": {
                        **mutation_file,
                        **mutation_download,
                    },
                },
                "selected_genes": selected_genes,
                "research_use_only": True,
            }

            profile_path = patient_directory / "profile.json"
            _write_json(path=profile_path, payload=profile)

            row = _build_dataset_row(profile=profile, genes=selected_genes)
            row["cohort_type"] = "confirmed_tnbc"
            row["er_status"] = receptor_status.get("er")
            row["pr_status"] = receptor_status.get("pr")
            row["her2_status"] = receptor_status.get("her2")
            dataset_rows.append(row)

            completed_profiles.append(
                {
                    "patient": patient_id,
                    "profile_path": str(profile_path),
                    "rna_downloaded_now": rna_download["downloaded_now"],
                    "mutation_downloaded_now": mutation_download[
                        "downloaded_now"
                    ],
                    "found_rna_genes": rna_expression["found_gene_count"],
                    "patient_mutation_count": somatic_mutations[
                        "total_mutation_count"
                    ],
                    "selected_mutation_count": len(
                        somatic_mutations["selected_mutations"]
                    ),
                    "vital_status": clinical.get("vital_status"),
                    "receptor_status": receptor_status,
                }
            )

        except (
            DownloadManagerError,
            CohortBuildError,
            CohortDiscoveryError,
            TNBCDiscoveryError,
            TNBCCohortBuildError,
            OSError,
            ValueError,
        ) as exc:
            failures.append(
                {
                    "patient": patient_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "patient": patient_id,
                    "error_type": type(exc).__name__,
                    "error": f"Unexpected patient-processing error: {exc}",
                }
            )

    completed_count = len(dataset_rows)
    safe_output_name = _sanitize_output_name(output_name, completed_count)
    dataset_path = DATASET_DIRECTORY / f"{safe_output_name}.csv"
    manifest_path = DATASET_DIRECTORY / f"{safe_output_name}_manifest.json"

    if dataset_rows:
        _write_dataset_csv(rows=dataset_rows, output_path=dataset_path)

    target_reached = completed_count >= requested_count

    manifest: dict[str, Any] = {
        "project": "TCGA-BRCA",
        "cohort": "confirmed_tnbc",
        "build_status": (
            "complete"
            if target_reached and not failures
            else "complete_with_failures"
            if target_reached
            else "partial"
        ),
        "target_reached": target_reached,
        "tnbc_definition": (
            "ER-negative, PR-negative, and HER2-negative in TCGA clinical "
            "supplements"
        ),
        "clinical_file_limit": clinical_file_limit,
        "confirmed_tnbc_patient_count": len(confirmed_ids),
        "molecularly_eligible_tnbc_count": len(eligible_ids),
        "requested_patient_count": requested_count,
        "attempted_patient_count": len(candidate_ids),
        "completed_patient_count": completed_count,
        "failed_patient_count": len(failures),
        "missing_rna_patient_count": len(missing_rna_ids),
        "missing_mutation_patient_count": len(missing_mutation_ids),
        "missing_rna_patient_ids": missing_rna_ids,
        "missing_mutation_patient_ids": missing_mutation_ids,
        "selected_genes": selected_genes,
        "outcome_summary": _summarize_outcomes(dataset_rows),
        "patient_specific_maf_filtering": True,
        "unique_mutation_files_downloaded_or_reused": len(
            mutation_download_cache
        ),
        "cohort_directory": str(TNBC_COHORT_DIRECTORY),
        "dataset_path": str(dataset_path) if dataset_rows else None,
        "patients": completed_profiles,
        "failures": failures,
        "research_use_only": True,
    }

    _write_json(path=manifest_path, payload=manifest)
    manifest["manifest_path"] = str(manifest_path)

    return manifest
