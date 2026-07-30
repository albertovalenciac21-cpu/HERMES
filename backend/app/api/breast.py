from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.app.services.gdc import (
    get_breast_projects,
    get_tcga_brca_case,
    get_tcga_brca_case_files,
    get_tcga_brca_cases,
    get_tcga_brca_files,
    get_tcga_brca_project,
    open_gdc_file_stream,
    preview_gdc_text_file,
)

router = APIRouter(
    prefix="/breast",
    tags=["Breast Cancer"],
)


@router.get("/projects")
def breast_projects():
    try:
        return get_breast_projects()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.get("/tcga-brca")
def tcga_brca_project():
    try:
        return get_tcga_brca_project()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.get("/tcga-brca/cases")
def tcga_brca_cases(
    size: int = Query(default=25, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        return get_tcga_brca_cases(
            size=size,
            offset=offset,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.get("/tcga-brca/files")
def tcga_brca_files(
    size: int = Query(default=25, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    data_category: str | None = Query(default=None),
    data_type: str | None = Query(default=None),
):
    try:
        return get_tcga_brca_files(
            size=size,
            offset=offset,
            data_category=data_category,
            data_type=data_type,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.get("/tcga-brca/files/{file_id}/preview")
def preview_tcga_brca_file(
    file_id: str,
    line_count: int = Query(
        default=15,
        ge=1,
        le=100,
        description="Number of lines to preview.",
    ),
):
    """
    Preview the first lines of an open-access GDC text file.
    """

    try:
        return preview_gdc_text_file(
            file_id=file_id,
            line_count=line_count,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.get("/tcga-brca/files/{file_id}/download")
def download_tcga_brca_file(
    file_id: str,
):
    """
    Stream an open-access GDC file to the user.
    """

    try:
        file_iterator, file_name, media_type = (
            open_gdc_file_stream(file_id)
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except PermissionError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    encoded_name = quote(file_name)

    return StreamingResponse(
        file_iterator,
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f"attachment; filename*=UTF-8''{encoded_name}"
            )
        },
    )


@router.get("/tcga-brca/cases/{submitter_id}")
def tcga_brca_case(
    submitter_id: str,
):
    try:
        result = get_tcga_brca_case(
            submitter_id=submitter_id,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    hits = result.get("data", {}).get("hits", [])

    if not hits:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No TCGA-BRCA case was found for "
                f"submitter ID '{submitter_id}'."
            ),
        )

    return result


@router.get("/tcga-brca/cases/{submitter_id}/files")
def tcga_brca_case_files(
    submitter_id: str,
    size: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    data_category: str | None = Query(default=None),
    data_type: str | None = Query(default=None),
):
    try:
        result = get_tcga_brca_case_files(
            submitter_id=submitter_id,
            size=size,
            offset=offset,
            data_category=data_category,
            data_type=data_type,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    hits = result.get("data", {}).get("hits", [])

    if not hits:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No files were found for TCGA-BRCA patient "
                f"'{submitter_id}' using the selected filters."
            ),
        )

    return result