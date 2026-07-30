from fastapi import APIRouter, HTTPException, Query

from backend.app.services.mutations import (
    get_patient_maf_files,
    get_patient_mutations,
    get_tcga_brca_maf_files,
    summarize_tcga_brca_maf_files,
)


router = APIRouter(
    prefix="/mutations",
    tags=["Somatic Mutations"],
)


@router.get("/tcga-brca/files")
def tcga_brca_mutation_files():
    """
    Retrieve full GDC metadata for open TCGA-BRCA MAF files.
    """

    try:
        return get_tcga_brca_maf_files()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc


@router.get("/tcga-brca/files/summary")
def tcga_brca_mutation_file_summary():
    """
    Retrieve a simplified list of TCGA-BRCA MAF files.
    """

    try:
        result = summarize_tcga_brca_maf_files()

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    if result["file_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "No open-access TCGA-BRCA masked somatic "
                "mutation MAF files were found."
            ),
        )

    return result


@router.get("/patient/{submitter_id}/files")
def patient_mutation_files(
    submitter_id: str,
):
    """
    Find mutation MAF files associated with one patient.
    """

    try:
        result = get_patient_maf_files(
            submitter_id=submitter_id,
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    if result["file_count"] == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No open mutation MAF files were found for "
                f"patient '{submitter_id}'."
            ),
        )

    return result


@router.get("/patient/{submitter_id}")
def patient_mutations(
    submitter_id: str,
    file_id: str | None = Query(
        default=None,
        description=(
            "Optional GDC mutation file ID. When omitted, "
            "the first matching patient MAF file is used."
        ),
    ),
    genes: str | None = Query(
        default=None,
        description=(
            "Optional comma-separated gene filter. "
            "Example: TP53,PIK3CA,BRCA1"
        ),
    ),
    max_results: int = Query(
        default=500,
        ge=1,
        le=5000,
        description="Maximum number of mutations to return.",
    ),
):
    """
    Download and extract somatic mutations for one patient.
    """

    gene_list = []

    if genes:
        gene_list = [
            gene.strip()
            for gene in genes.split(",")
            if gene.strip()
        ]

    try:
        return get_patient_mutations(
            submitter_id=submitter_id,
            file_id=file_id,
            genes=gene_list,
            max_results=max_results,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc