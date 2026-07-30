import json
from collections.abc import Iterator
from typing import Any

import requests

GDC_BASE_URL = "https://api.gdc.cancer.gov"
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120


def _send_gdc_request(
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Send a GET request to a GDC search endpoint.
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
            "The GDC API returned an invalid JSON response."
        ) from exc


def get_projects(size: int = 10) -> dict[str, Any]:
    """
    Retrieve projects from the NCI Genomic Data Commons.
    """

    params = {
        "fields": (
            "project_id,"
            "name,"
            "program.name,"
            "primary_site,"
            "disease_type,"
            "summary.case_count,"
            "summary.file_count"
        ),
        "format": "JSON",
        "size": size,
    }

    return _send_gdc_request(
        endpoint="projects",
        params=params,
    )


def get_breast_projects() -> dict[str, Any]:
    """
    Retrieve projects containing breast cancer data.
    """

    filters = {
        "op": "in",
        "content": {
            "field": "primary_site",
            "value": ["Breast"],
        },
    }

    params = {
        "filters": json.dumps(filters),
        "fields": (
            "project_id,"
            "name,"
            "program.name,"
            "primary_site,"
            "disease_type,"
            "summary.case_count,"
            "summary.file_count"
        ),
        "format": "JSON",
        "size": 100,
    }

    return _send_gdc_request(
        endpoint="projects",
        params=params,
    )


def get_tcga_brca_project() -> dict[str, Any]:
    """
    Retrieve the TCGA Breast Invasive Carcinoma project.
    """

    filters = {
        "op": "in",
        "content": {
            "field": "project_id",
            "value": ["TCGA-BRCA"],
        },
    }

    params = {
        "filters": json.dumps(filters),
        "fields": (
            "project_id,"
            "name,"
            "program.name,"
            "primary_site,"
            "disease_type,"
            "summary.case_count,"
            "summary.file_count"
        ),
        "format": "JSON",
        "size": 1,
    }

    return _send_gdc_request(
        endpoint="projects",
        params=params,
    )


def get_tcga_brca_cases(
    size: int = 25,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Retrieve patient-level cases from TCGA-BRCA.
    """

    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": ["TCGA-BRCA"],
        },
    }

    params = {
        "filters": json.dumps(filters),
        "fields": (
            "case_id,"
            "submitter_id,"
            "primary_site,"
            "disease_type,"
            "demographic.gender,"
            "demographic.race,"
            "demographic.ethnicity,"
            "demographic.vital_status,"
            "demographic.days_to_birth,"
            "demographic.days_to_death,"
            "diagnoses.diagnosis_id,"
            "diagnoses.primary_diagnosis,"
            "diagnoses.tumor_stage,"
            "diagnoses.tumor_grade,"
            "diagnoses.age_at_diagnosis,"
            "diagnoses.days_to_last_follow_up,"
            "diagnoses.last_known_disease_status,"
            "diagnoses.tissue_or_organ_of_origin,"
            "samples.sample_id,"
            "samples.submitter_id,"
            "samples.sample_type,"
            "samples.tissue_type"
        ),
        "expand": "demographic,diagnoses,samples",
        "format": "JSON",
        "size": size,
        "from": offset,
        "sort": "submitter_id:asc",
    }

    return _send_gdc_request(
        endpoint="cases",
        params=params,
    )


def get_tcga_brca_case(
    submitter_id: str,
) -> dict[str, Any]:
    """
    Retrieve one TCGA-BRCA case using its TCGA identifier.
    """

    normalized_submitter_id = submitter_id.strip().upper()

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "project.project_id",
                    "value": ["TCGA-BRCA"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "submitter_id",
                    "value": [normalized_submitter_id],
                },
            },
        ],
    }

    params = {
        "filters": json.dumps(filters),
        "fields": (
            "case_id,"
            "submitter_id,"
            "primary_site,"
            "disease_type,"
            "demographic.gender,"
            "demographic.race,"
            "demographic.ethnicity,"
            "demographic.vital_status,"
            "demographic.days_to_birth,"
            "demographic.days_to_death,"
            "diagnoses.diagnosis_id,"
            "diagnoses.primary_diagnosis,"
            "diagnoses.tumor_stage,"
            "diagnoses.tumor_grade,"
            "diagnoses.age_at_diagnosis,"
            "diagnoses.days_to_last_follow_up,"
            "diagnoses.last_known_disease_status,"
            "diagnoses.tissue_or_organ_of_origin,"
            "samples.sample_id,"
            "samples.submitter_id,"
            "samples.sample_type,"
            "samples.tissue_type"
        ),
        "expand": "demographic,diagnoses,samples",
        "format": "JSON",
        "size": 1,
    }

    return _send_gdc_request(
        endpoint="cases",
        params=params,
    )


def _build_file_filters(
    submitter_id: str | None = None,
    data_category: str | None = None,
    data_type: str | None = None,
) -> dict[str, Any]:
    """
    Build reusable filters for TCGA-BRCA file searches.
    """

    conditions: list[dict[str, Any]] = [
        {
            "op": "in",
            "content": {
                "field": "cases.project.project_id",
                "value": ["TCGA-BRCA"],
            },
        }
    ]

    if submitter_id:
        conditions.append(
            {
                "op": "in",
                "content": {
                    "field": "cases.submitter_id",
                    "value": [submitter_id.strip().upper()],
                },
            }
        )

    if data_category:
        conditions.append(
            {
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": [data_category],
                },
            }
        )

    if data_type:
        conditions.append(
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": [data_type],
                },
            }
        )

    return {
        "op": "and",
        "content": conditions,
    }


def get_tcga_brca_files(
    size: int = 25,
    offset: int = 0,
    data_category: str | None = None,
    data_type: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve files associated with TCGA-BRCA.
    """

    filters = _build_file_filters(
        data_category=data_category,
        data_type=data_type,
    )

    params = {
        "filters": json.dumps(filters),
        "fields": (
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
            "cases.case_id,"
            "cases.submitter_id,"
            "cases.samples.sample_id,"
            "cases.samples.submitter_id,"
            "cases.samples.sample_type,"
            "cases.samples.tissue_type"
        ),
        "expand": "analysis,cases,cases.samples",
        "format": "JSON",
        "size": size,
        "from": offset,
        "sort": "file_name:asc",
    }

    return _send_gdc_request(
        endpoint="files",
        params=params,
    )


def get_tcga_brca_case_files(
    submitter_id: str,
    size: int = 100,
    offset: int = 0,
    data_category: str | None = None,
    data_type: str | None = None,
) -> dict[str, Any]:
    """
    Retrieve files associated with one TCGA-BRCA patient.
    """

    filters = _build_file_filters(
        submitter_id=submitter_id,
        data_category=data_category,
        data_type=data_type,
    )

    params = {
        "filters": json.dumps(filters),
        "fields": (
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
            "cases.case_id,"
            "cases.submitter_id,"
            "cases.samples.sample_id,"
            "cases.samples.submitter_id,"
            "cases.samples.sample_type,"
            "cases.samples.tissue_type"
        ),
        "expand": "analysis,cases,cases.samples",
        "format": "JSON",
        "size": size,
        "from": offset,
        "sort": "file_name:asc",
    }

    return _send_gdc_request(
        endpoint="files",
        params=params,
    )


def get_gdc_file_metadata(
    file_id: str,
) -> dict[str, Any]:
    """
    Retrieve metadata for one GDC file.
    """

    normalized_file_id = file_id.strip()

    filters = {
        "op": "in",
        "content": {
            "field": "file_id",
            "value": [normalized_file_id],
        },
    }

    params = {
        "filters": json.dumps(filters),
        "fields": (
            "file_id,"
            "file_name,"
            "file_size,"
            "md5sum,"
            "data_category,"
            "data_type,"
            "data_format,"
            "experimental_strategy,"
            "access,"
            "state"
        ),
        "format": "JSON",
        "size": 1,
    }

    return _send_gdc_request(
        endpoint="files",
        params=params,
    )


def preview_gdc_text_file(
    file_id: str,
    line_count: int = 15,
) -> dict[str, Any]:
    """
    Stream and return the first lines of an open-access text file.
    """

    metadata_response = get_gdc_file_metadata(file_id)
    hits = metadata_response.get("data", {}).get("hits", [])

    if not hits:
        raise FileNotFoundError(
            f"No GDC file was found for file ID '{file_id}'."
        )

    metadata = hits[0]

    if metadata.get("access") != "open":
        raise PermissionError(
            "This GDC file is controlled-access and cannot be "
            "previewed without authorization."
        )

    try:
        response = requests.get(
            f"{GDC_BASE_URL}/data/{file_id}",
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
        )

        response.raise_for_status()

        lines: list[str] = []

        for raw_line in response.iter_lines(
            decode_unicode=True,
        ):
            if raw_line is None:
                continue

            lines.append(raw_line)

            if len(lines) >= line_count:
                break

        response.close()

        return {
            "file_id": metadata.get("file_id"),
            "file_name": metadata.get("file_name"),
            "file_size": metadata.get("file_size"),
            "data_category": metadata.get("data_category"),
            "data_type": metadata.get("data_type"),
            "data_format": metadata.get("data_format"),
            "experimental_strategy": metadata.get(
                "experimental_strategy"
            ),
            "access": metadata.get("access"),
            "preview_line_count": len(lines),
            "lines": lines,
        }

    except requests.Timeout as exc:
        raise RuntimeError(
            "The GDC file preview request timed out."
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            "The GDC file could not be previewed."
        ) from exc


def open_gdc_file_stream(
    file_id: str,
) -> tuple[
    Iterator[bytes],
    str,
    str,
]:
    """
    Open a streaming response for one open-access GDC file.

    Returns:
        A byte iterator, file name, and media type.
    """

    metadata_response = get_gdc_file_metadata(file_id)
    hits = metadata_response.get("data", {}).get("hits", [])

    if not hits:
        raise FileNotFoundError(
            f"No GDC file was found for file ID '{file_id}'."
        )

    metadata = hits[0]

    if metadata.get("access") != "open":
        raise PermissionError(
            "This GDC file is controlled-access and requires "
            "GDC authorization."
        )

    try:
        response = requests.get(
            f"{GDC_BASE_URL}/data/{file_id}",
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
        )

        response.raise_for_status()

    except requests.Timeout as exc:
        raise RuntimeError(
            "The GDC download request timed out."
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            "The GDC file could not be downloaded."
        ) from exc

    def file_iterator() -> Iterator[bytes]:
        try:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024,
            ):
                if chunk:
                    yield chunk
        finally:
            response.close()

    file_name = metadata.get("file_name") or f"{file_id}.dat"
    media_type = response.headers.get(
        "Content-Type",
        "application/octet-stream",
    )

    return file_iterator(), file_name, media_type