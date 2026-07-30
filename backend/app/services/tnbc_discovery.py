from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests


GDC_BASE_URL = "https://api.gdc.cancer.gov"
REQUEST_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 180

PROJECT_ID = "TCGA-BRCA"

CLINICAL_DIRECTORY = Path(
    "data/clinical/tcga-brca"
)


class TNBCDiscoveryError(RuntimeError):
    """Raised when TNBC receptor information cannot be scanned."""


def _request_clinical_supplement_files(
    limit: int,
) -> list[dict[str, Any]]:
    """
    Retrieve TCGA-BRCA clinical-supplement file metadata.

    The query does not restrict data_format because GDC may
    label TCGA clinical supplements using values such as
    BCR XML rather than XML.
    """

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [PROJECT_ID],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_category",
                    "value": ["Clinical"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "data_type",
                    "value": ["Clinical Supplement"],
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
        "fields": (
            "file_id,"
            "file_name,"
            "file_size,"
            "md5sum,"
            "data_category,"
            "data_type,"
            "data_format,"
            "access,"
            "cases.case_id,"
            "cases.submitter_id"
        ),
        "expand": "cases",
        "format": "JSON",
        "size": limit,
        "sort": "file_name:asc",
    }

    try:
        response = requests.get(
            f"{GDC_BASE_URL}/files",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        payload = response.json()

    except requests.Timeout as exc:
        raise TNBCDiscoveryError(
            "The GDC clinical-file search timed out."
        ) from exc

    except requests.RequestException as exc:
        raise TNBCDiscoveryError(
            "The GDC clinical-file search failed."
        ) from exc

    except ValueError as exc:
        raise TNBCDiscoveryError(
            "The GDC returned invalid clinical-file JSON."
        ) from exc

    return (
        payload
        .get("data", {})
        .get("hits", [])
    )


def _calculate_md5(
    path: Path,
) -> str:
    """
    Calculate the MD5 checksum of a downloaded file.
    """

    digest = hashlib.md5()

    with path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _download_xml_file(
    file_record: dict[str, Any],
) -> tuple[Path, bool]:
    """
    Download one clinical-supplement file.

    Existing files are reused when their MD5 checksum matches.
    """

    file_id = str(
        file_record.get("file_id") or ""
    ).strip()

    if not file_id:
        raise TNBCDiscoveryError(
            "A clinical file is missing its GDC file ID."
        )

    file_name = (
        file_record.get("file_name")
        or f"{file_id}.xml"
    )

    destination = (
        CLINICAL_DIRECTORY
        / file_id
        / file_name
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_md5 = (
        str(file_record.get("md5sum") or "")
        .strip()
        .lower()
    )

    if destination.exists():
        if not expected_md5:
            return destination, False

        observed_md5 = _calculate_md5(
            destination
        )

        if observed_md5 == expected_md5:
            return destination, False

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    try:
        with requests.get(
            f"{GDC_BASE_URL}/data/{file_id}",
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
        ) as response:
            response.raise_for_status()

            with temporary_path.open(
                mode="wb"
            ) as output_file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024,
                ):
                    if chunk:
                        output_file.write(chunk)

    except requests.Timeout as exc:
        temporary_path.unlink(
            missing_ok=True
        )

        raise TNBCDiscoveryError(
            f"Clinical file '{file_id}' timed out."
        ) from exc

    except requests.RequestException as exc:
        temporary_path.unlink(
            missing_ok=True
        )

        raise TNBCDiscoveryError(
            f"Clinical file '{file_id}' could not be downloaded."
        ) from exc

    if expected_md5:
        observed_md5 = _calculate_md5(
            temporary_path
        )

        if observed_md5 != expected_md5:
            temporary_path.unlink(
                missing_ok=True
            )

            raise TNBCDiscoveryError(
                f"MD5 verification failed for '{file_id}'."
            )

    temporary_path.replace(destination)

    return destination, True


def _local_xml_name(
    tag: str,
) -> str:
    """
    Remove an XML namespace from an element tag.
    """

    if "}" in tag:
        tag = tag.split("}", 1)[1]

    return tag.strip().lower()


def _normalize_text(
    value: str | None,
) -> str:
    """
    Normalize XML text for matching.
    """

    if value is None:
        return ""

    normalized = value.strip().lower()
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("—", "-")

    return re.sub(
        r"\s+",
        " ",
        normalized,
    )


def _identify_receptor(
    field_name: str,
) -> str | None:
    """
    Determine which receptor an XML field describes.
    """

    normalized = field_name.lower()

    er_patterns = [
        "estrogen_receptor",
        "estrogen receptor",
        "er_status",
        "er_ihc",
    ]

    pr_patterns = [
        "progesterone_receptor",
        "progesterone receptor",
        "pr_status",
        "pr_ihc",
    ]

    her2_patterns = [
        "her2",
        "her_2",
        "her-2",
        "her2_neu",
        "her_2_neu",
        "her-2-neu",
    ]

    if any(
        pattern in normalized
        for pattern in er_patterns
    ):
        return "er"

    if any(
        pattern in normalized
        for pattern in pr_patterns
    ):
        return "pr"

    if any(
        pattern in normalized
        for pattern in her2_patterns
    ):
        return "her2"

    return None


def _classify_receptor_value(
    value: str,
) -> str:
    """
    Classify receptor text as positive, negative, equivocal,
    or unknown.
    """

    normalized = _normalize_text(value)

    if not normalized:
        return "unknown"

    negative_values = {
        "negative",
        "neg",
        "0",
        "0.0",
        "not detected",
        "absent",
        "no",
    }

    positive_values = {
        "positive",
        "pos",
        "1",
        "1.0",
        "detected",
        "present",
        "yes",
    }

    equivocal_values = {
        "equivocal",
        "indeterminate",
        "borderline",
        "2+",
    }

    unknown_values = {
        "unknown",
        "not available",
        "not reported",
        "not evaluated",
        "not tested",
        "not applicable",
        "na",
        "n/a",
    }

    if normalized in negative_values:
        return "negative"

    if normalized in positive_values:
        return "positive"

    if normalized in equivocal_values:
        return "equivocal"

    if normalized in unknown_values:
        return "unknown"

    if "negative" in normalized:
        return "negative"

    if "positive" in normalized:
        return "positive"

    if "equivocal" in normalized:
        return "equivocal"

    return "unknown"


def _aggregate_receptor_status(
    classified_values: list[str],
) -> str:
    """
    Combine multiple receptor observations for one patient.
    """

    informative = {
        value
        for value in classified_values
        if value != "unknown"
    }

    if not informative:
        return "unknown"

    if (
        "positive" in informative
        and "negative" in informative
    ):
        return "conflicting"

    if "positive" in informative:
        return "positive"

    if "negative" in informative:
        return "negative"

    if "equivocal" in informative:
        return "equivocal"

    return "unknown"


def _extract_receptor_fields(
    xml_path: Path,
) -> dict[str, Any]:
    """
    Extract receptor-related XML fields from one supplement.
    """

    try:
        tree = ET.parse(xml_path)

    except ET.ParseError as exc:
        raise TNBCDiscoveryError(
            f"Invalid XML in '{xml_path}'."
        ) from exc

    receptor_fields: dict[
        str,
        list[dict[str, str]],
    ] = {
        "er": [],
        "pr": [],
        "her2": [],
    }

    submitter_ids: set[str] = set()

    for element in tree.iter():
        field_name = _local_xml_name(
            element.tag
        )

        field_value = _normalize_text(
            element.text
        )

        if (
            "patient" in field_name
            and "barcode" in field_name
            and field_value
        ):
            submitter_ids.add(
                field_value.upper()
            )

        if (
            field_name
            in {
                "bcr_patient_barcode",
                "submitter_id",
            }
            and field_value.startswith("tcga-")
        ):
            submitter_ids.add(
                field_value.upper()
            )

        receptor = _identify_receptor(
            field_name
        )

        if receptor is None or not field_value:
            continue

        receptor_fields[receptor].append(
            {
                "xml_field": field_name,
                "raw_value": field_value,
                "classification": (
                    _classify_receptor_value(
                        field_value
                    )
                ),
            }
        )

    receptor_status = {
        receptor: _aggregate_receptor_status(
            [
                item["classification"]
                for item in observations
            ]
        )
        for receptor, observations
        in receptor_fields.items()
    }

    confirmed_tnbc = all(
        receptor_status[receptor]
        == "negative"
        for receptor in [
            "er",
            "pr",
            "her2",
        ]
    )

    return {
        "xml_submitter_ids": sorted(
            submitter_ids
        ),
        "receptor_status": receptor_status,
        "confirmed_tnbc": confirmed_tnbc,
        "receptor_fields": receptor_fields,
    }


def _extract_case_ids(
    file_record: dict[str, Any],
) -> list[str]:
    """
    Extract TCGA submitter IDs associated with a file.
    """

    submitter_ids: list[str] = []

    for case in file_record.get(
        "cases"
    ) or []:
        submitter_id = (
            case.get("submitter_id")
            or ""
        ).strip().upper()

        if submitter_id:
            submitter_ids.append(
                submitter_id
            )

    return sorted(
        set(submitter_ids)
    )


def scan_tcga_brca_receptor_supplements(
    limit: int = 25,
) -> dict[str, Any]:
    """
    Download and inspect TCGA-BRCA clinical supplements.

    This discovery step identifies ER, PR, and HER2 fields
    available in the source files.
    """

    if limit < 1 or limit > 1000:
        raise ValueError(
            "The clinical-supplement limit must be between "
            "1 and 1000."
        )

    files = _request_clinical_supplement_files(
        limit=limit
    )

    scanned_files: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    patient_results: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    downloaded_now_count = 0

    for file_record in files:
        file_id = str(
            file_record.get("file_id") or ""
        )

        try:
            xml_path, downloaded_now = (
                _download_xml_file(
                    file_record
                )
            )

            if downloaded_now:
                downloaded_now_count += 1

            extracted = _extract_receptor_fields(
                xml_path
            )

            case_submitter_ids = (
                _extract_case_ids(
                    file_record
                )
            )

            patient_ids = sorted(
                set(case_submitter_ids)
                | set(
                    extracted[
                        "xml_submitter_ids"
                    ]
                )
            )

            file_result = {
                "file_id": file_id,
                "file_name": file_record.get(
                    "file_name"
                ),
                "data_format": file_record.get(
                    "data_format"
                ),
                "xml_path": str(xml_path),
                "downloaded_now": downloaded_now,
                "patient_ids": patient_ids,
                **extracted,
            }

            scanned_files.append(
                file_result
            )

            for patient_id in patient_ids:
                patient_results[
                    patient_id
                ].append(file_result)

        except (
            TNBCDiscoveryError,
            OSError,
            ValueError,
        ) as exc:
            failures.append(
                {
                    "file_id": file_id,
                    "file_name": str(
                        file_record.get(
                            "file_name"
                        )
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": str(exc),
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "file_id": file_id,
                    "file_name": str(
                        file_record.get(
                            "file_name"
                        )
                    ),
                    "error_type": (
                        type(exc).__name__
                    ),
                    "error": (
                        "Unexpected clinical-file "
                        f"processing error: {exc}"
                    ),
                }
            )

    confirmed_tnbc_patients: list[
        dict[str, Any]
    ] = []

    patients_with_complete_receptors = 0
    patients_with_any_receptor_data = 0

    for patient_id, patient_files in sorted(
        patient_results.items()
    ):
        receptor_classifications = {
            "er": [],
            "pr": [],
            "her2": [],
        }

        receptor_observations = {
            "er": [],
            "pr": [],
            "her2": [],
        }

        for file_result in patient_files:
            for receptor in [
                "er",
                "pr",
                "her2",
            ]:
                receptor_classifications[
                    receptor
                ].append(
                    file_result[
                        "receptor_status"
                    ][receptor]
                )

                receptor_observations[
                    receptor
                ].extend(
                    file_result[
                        "receptor_fields"
                    ][receptor]
                )

        combined_status = {
            receptor: (
                _aggregate_receptor_status(
                    values
                )
            )
            for receptor, values
            in receptor_classifications.items()
        }

        has_any_data = any(
            status != "unknown"
            for status in combined_status.values()
        )

        has_complete_data = all(
            status != "unknown"
            for status in combined_status.values()
        )

        confirmed_tnbc = all(
            combined_status[receptor]
            == "negative"
            for receptor in [
                "er",
                "pr",
                "her2",
            ]
        )

        if has_any_data:
            patients_with_any_receptor_data += 1

        if has_complete_data:
            patients_with_complete_receptors += 1

        if confirmed_tnbc:
            confirmed_tnbc_patients.append(
                {
                    "patient_id": patient_id,
                    "receptor_status": (
                        combined_status
                    ),
                    "source_file_count": len(
                        patient_files
                    ),
                    "receptor_observations": (
                        receptor_observations
                    ),
                }
            )

    return {
        "project": PROJECT_ID,
        "scan_status": (
            "complete"
            if not failures
            else "complete_with_failures"
        ),
        "requested_file_limit": limit,
        "clinical_files_found": len(files),
        "clinical_files_scanned": len(
            scanned_files
        ),
        "downloaded_now_count": (
            downloaded_now_count
        ),
        "reused_download_count": (
            len(scanned_files)
            - downloaded_now_count
        ),
        "failed_file_count": len(
            failures
        ),
        "patient_count": len(
            patient_results
        ),
        "patients_with_any_receptor_data": (
            patients_with_any_receptor_data
        ),
        "patients_with_complete_receptors": (
            patients_with_complete_receptors
        ),
        "confirmed_tnbc_patient_count": len(
            confirmed_tnbc_patients
        ),
        "confirmed_tnbc_patients": (
            confirmed_tnbc_patients
        ),
        "failures": failures,
        "scanned_files": scanned_files,
        "next_step": (
            "Review the receptor XML field names and use the "
            "confirmed TNBC patient IDs to build a "
            "TNBC-specific molecular cohort."
        ),
        "research_use_only": True,
    }