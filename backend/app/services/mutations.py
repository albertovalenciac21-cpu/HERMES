import csv
import gzip
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests


GDC_BASE_URL = "https://api.gdc.cancer.gov"
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 180

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MUTATION_DATA_DIR = PROJECT_ROOT / "data" / "mutations"


def _send_gdc_request(
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Send a request to a GDC search endpoint.
    """

    try:
        response = requests.get(
            f"{GDC_BASE_URL}/{endpoint}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        return response.json()

    except requests.Timeout as exc:
        raise RuntimeError(
            "The request to the GDC API timed out."
        ) from exc

    except requests.HTTPError as exc:
        status_code = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )

        raise RuntimeError(
            f"The GDC API returned HTTP status {status_code}."
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            "Unable to communicate with the GDC API."
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "The GDC API returned invalid JSON."
        ) from exc


def _mutation_file_fields() -> str:
    """
    Return the metadata fields used for mutation-file searches.
    """

    return (
        "file_id,"
        "file_name,"
        "file_size,"
        "md5sum,"
        "data_category,"
        "data_type,"
        "data_format,"
        "experimental_strategy,"
        "access,"
        "state,"
        "analysis.workflow_type,"
        "analysis.workflow_version,"
        "cases.case_id,"
        "cases.submitter_id,"
        "cases.samples.sample_id,"
        "cases.samples.submitter_id,"
        "cases.samples.sample_type"
    )


def get_tcga_brca_maf_files(
    size: int = 100,
) -> dict[str, Any]:
    """
    Retrieve open-access TCGA-BRCA masked somatic mutation files.
    """

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": ["Simple Nucleotide Variation"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Masked Somatic Mutation"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_format",
                    "value": ["MAF"],
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

    params = {
        "filters": json.dumps(filters),
        "fields": _mutation_file_fields(),
        "expand": "analysis,cases,cases.samples",
        "format": "JSON",
        "size": size,
        "sort": "file_name:asc",
    }

    return _send_gdc_request(
        endpoint="files",
        params=params,
    )


def summarize_tcga_brca_maf_files() -> dict[str, Any]:
    """
    Return a concise list of available TCGA-BRCA mutation files.
    """

    response = get_tcga_brca_maf_files()
    hits = response.get("data", {}).get("hits", [])

    files = []

    for hit in hits:
        analysis = hit.get("analysis") or {}

        files.append(
            {
                "file_id": hit.get("file_id") or hit.get("id"),
                "file_name": hit.get("file_name"),
                "file_size": hit.get("file_size"),
                "data_type": hit.get("data_type"),
                "data_format": hit.get("data_format"),
                "access": hit.get("access"),
                "experimental_strategy": hit.get(
                    "experimental_strategy"
                ),
                "workflow_type": analysis.get("workflow_type"),
                "workflow_version": analysis.get(
                    "workflow_version"
                ),
            }
        )

    return {
        "project": "TCGA-BRCA",
        "data_category": "Simple Nucleotide Variation",
        "data_type": "Masked Somatic Mutation",
        "data_format": "MAF",
        "file_count": len(files),
        "files": files,
    }


def get_patient_maf_files(
    submitter_id: str,
    size: int = 25,
) -> dict[str, Any]:
    """
    Retrieve open-access mutation files associated with one patient.
    """

    normalized_patient = submitter_id.strip().upper()

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": [normalized_patient],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": ["Simple Nucleotide Variation"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Masked Somatic Mutation"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_format",
                    "value": ["MAF"],
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

    params = {
        "filters": json.dumps(filters),
        "fields": _mutation_file_fields(),
        "expand": "analysis,cases,cases.samples",
        "format": "JSON",
        "size": size,
        "sort": "file_name:asc",
    }

    response = _send_gdc_request(
        endpoint="files",
        params=params,
    )

    hits = response.get("data", {}).get("hits", [])
    files: list[dict[str, Any]] = []

    for hit in hits:
        analysis = hit.get("analysis") or {}

        files.append(
            {
                "file_id": hit.get("file_id") or hit.get("id"),
                "file_name": hit.get("file_name"),
                "file_size": hit.get("file_size"),
                "md5sum": hit.get("md5sum"),
                "access": hit.get("access"),
                "data_format": hit.get("data_format"),
                "experimental_strategy": hit.get(
                    "experimental_strategy"
                ),
                "workflow_type": analysis.get("workflow_type"),
                "workflow_version": analysis.get(
                    "workflow_version"
                ),
                "cases": hit.get("cases", []),
            }
        )

    return {
        "patient": normalized_patient,
        "file_count": len(files),
        "files": files,
    }


def _download_with_curl(
    file_id: str,
    destination: Path,
) -> None:
    """
    Download a GDC file using curl.

    curl is used because it successfully downloaded GDC data in the
    current Windows environment when Python requests was being reset.
    """

    curl_command = shutil.which("curl.exe") or shutil.which("curl")

    if curl_command is None:
        raise RuntimeError(
            "curl was not found. Install curl or ensure curl.exe "
            "is available in the Windows PATH."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_file = destination.with_suffix(
        destination.suffix + ".part"
    )

    command = [
        curl_command,
        "-L",
        "--fail",
        "--retry",
        "3",
        "--connect-timeout",
        "30",
        "--max-time",
        str(DOWNLOAD_TIMEOUT),
        "-o",
        str(temporary_file),
        f"{GDC_BASE_URL}/data/{file_id}",
    ]

    completed_process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed_process.returncode != 0:
        if temporary_file.exists():
            temporary_file.unlink()

        error_message = (
            completed_process.stderr.strip()
            or completed_process.stdout.strip()
            or "Unknown curl error."
        )

        raise RuntimeError(
            f"The mutation file could not be downloaded. "
            f"curl returned code {completed_process.returncode}. "
            f"Details: {error_message}"
        )

    if not temporary_file.exists():
        raise RuntimeError(
            "curl completed, but the downloaded file was not found."
        )

    temporary_file.replace(destination)


def download_patient_maf(
    submitter_id: str,
    file_id: str | None = None,
) -> dict[str, Any]:
    """
    Download and cache one mutation MAF file for a patient.
    """

    patient_files = get_patient_maf_files(
        submitter_id=submitter_id,
    )

    files = patient_files["files"]

    if not files:
        raise FileNotFoundError(
            f"No open mutation MAF files were found for "
            f"patient '{submitter_id}'."
        )

    selected_file: dict[str, Any] | None = None

    if file_id:
        selected_file = next(
            (
                item
                for item in files
                if item["file_id"] == file_id.strip()
            ),
            None,
        )

        if selected_file is None:
            raise FileNotFoundError(
                f"Mutation file '{file_id}' was not found for "
                f"patient '{submitter_id}'."
            )
    else:
        selected_file = files[0]

    selected_file_id = selected_file["file_id"]
    selected_file_name = (
        selected_file.get("file_name")
        or f"{selected_file_id}.maf.gz"
    )

    safe_patient = submitter_id.strip().upper()
    patient_directory = MUTATION_DATA_DIR / safe_patient

    destination = patient_directory / selected_file_name

    downloaded_now = False

    if not destination.exists():
        _download_with_curl(
            file_id=selected_file_id,
            destination=destination,
        )
        downloaded_now = True

    return {
        "patient": safe_patient,
        "file_id": selected_file_id,
        "file_name": selected_file_name,
        "local_path": str(destination),
        "downloaded_now": downloaded_now,
        "file_size": destination.stat().st_size,
        "workflow_type": selected_file.get("workflow_type"),
    }


def _open_maf_file(file_path: Path):
    """
    Open either a compressed or uncompressed MAF text file.
    """

    if file_path.name.lower().endswith(".gz"):
        return gzip.open(
            file_path,
            mode="rt",
            encoding="utf-8",
            errors="replace",
            newline="",
        )

    return file_path.open(
        mode="r",
        encoding="utf-8",
        errors="replace",
        newline="",
    )


def _create_maf_reader(file_handle) -> csv.DictReader:
    """
    Skip MAF comment lines and return a tab-delimited reader.
    """

    while True:
        position = file_handle.tell()
        line = file_handle.readline()

        if line == "":
            raise ValueError(
                "The MAF file does not contain a valid header."
            )

        if line.startswith("#"):
            continue

        file_handle.seek(position)
        break

    reader = csv.DictReader(
        file_handle,
        delimiter="\t",
    )

    if not reader.fieldnames:
        raise ValueError(
            "The MAF header could not be read."
        )

    return reader


def _clean_maf_value(
    row: dict[str, str],
    *column_names: str,
) -> str | None:
    """
    Return the first non-empty value from possible MAF columns.
    """

    for column_name in column_names:
        value = row.get(column_name)

        if value is not None:
            cleaned = value.strip()

            if cleaned and cleaned not in {".", "NA"}:
                return cleaned

    return None


def get_patient_mutations(
    submitter_id: str,
    file_id: str | None = None,
    genes: list[str] | None = None,
    max_results: int = 500,
) -> dict[str, Any]:
    """
    Download, parse, and return somatic mutations for one patient.
    """

    normalized_patient = submitter_id.strip().upper()

    download_result = download_patient_maf(
        submitter_id=normalized_patient,
        file_id=file_id,
    )

    maf_path = Path(download_result["local_path"])

    requested_genes = {
        gene.strip().upper()
        for gene in (genes or [])
        if gene.strip()
    }

    mutations: list[dict[str, Any]] = []
    total_patient_mutations = 0

    with _open_maf_file(maf_path) as file_handle:
        reader = _create_maf_reader(file_handle)

        for row in reader:
            tumor_barcode = _clean_maf_value(
                row,
                "Tumor_Sample_Barcode",
                "Tumor_Aliquot_Barcode",
            )

            # The MAF may be patient-specific or aggregated. When a
            # barcode exists, retain rows belonging to this patient.
            if (
                tumor_barcode
                and not tumor_barcode.upper().startswith(
                    normalized_patient
                )
            ):
                continue

            gene = _clean_maf_value(
                row,
                "Hugo_Symbol",
                "SYMBOL",
            )

            if not gene:
                continue

            total_patient_mutations += 1

            if (
                requested_genes
                and gene.upper() not in requested_genes
            ):
                continue

            mutation = {
                "gene": gene,
                "entrez_gene_id": _clean_maf_value(
                    row,
                    "Entrez_Gene_Id",
                ),
                "variant_classification": _clean_maf_value(
                    row,
                    "Variant_Classification",
                ),
                "variant_type": _clean_maf_value(
                    row,
                    "Variant_Type",
                ),
                "chromosome": _clean_maf_value(
                    row,
                    "Chromosome",
                ),
                "start_position": _clean_maf_value(
                    row,
                    "Start_Position",
                ),
                "end_position": _clean_maf_value(
                    row,
                    "End_Position",
                ),
                "reference_allele": _clean_maf_value(
                    row,
                    "Reference_Allele",
                ),
                "tumor_allele": _clean_maf_value(
                    row,
                    "Tumor_Seq_Allele2",
                    "Tumor_Seq_Allele1",
                ),
                "dbsnp_id": _clean_maf_value(
                    row,
                    "dbSNP_RS",
                    "Existing_variation",
                ),
                "transcript_id": _clean_maf_value(
                    row,
                    "Transcript_ID",
                    "Feature",
                ),
                "protein_change": _clean_maf_value(
                    row,
                    "HGVSp_Short",
                    "HGVSp",
                ),
                "coding_change": _clean_maf_value(
                    row,
                    "HGVSc",
                ),
                "tumor_sample_barcode": tumor_barcode,
            }

            mutations.append(mutation)

            if len(mutations) >= max_results:
                break

    mutated_genes = sorted(
        {
            mutation["gene"]
            for mutation in mutations
            if mutation.get("gene")
        }
    )

    return {
        "patient": normalized_patient,
        "source_file": download_result["file_name"],
        "source_file_id": download_result["file_id"],
        "source_path": download_result["local_path"],
        "workflow_type": download_result["workflow_type"],
        "downloaded_now": download_result["downloaded_now"],
        "gene_filter": sorted(requested_genes),
        "total_patient_mutations_in_file": total_patient_mutations,
        "returned_mutation_count": len(mutations),
        "mutated_gene_count": len(mutated_genes),
        "mutated_genes": mutated_genes,
        "results_truncated": len(mutations) >= max_results,
        "mutations": mutations,
    }