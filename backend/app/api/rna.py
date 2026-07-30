from fastapi import APIRouter, HTTPException, Query

from backend.app.services.rna import (
    get_gene_expression,
    preview_rna_file,
)


router = APIRouter(
    prefix="/rna",
    tags=["RNA Expression"],
)


@router.get("/preview")
def local_rna_preview(
    row_count: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Number of RNA-seq rows to display.",
    ),
):
    """
    Preview the locally downloaded GDC RNA-seq TSV file.
    """

    try:
        return preview_rna_file(
            row_count=row_count,
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


@router.get("/patient/{submitter_id}/expression")
def patient_gene_expression(
    submitter_id: str,
    genes: str = Query(
        default="TP53,BRCA1,BRCA2,ERBB2,CD274,PIK3CA",
        description=(
            "Comma-separated gene symbols. "
            "Example: TP53,BRCA1,ERBB2,CD274"
        ),
    ),
    expression_column: str = Query(
        default="tpm_unstranded",
        description=(
            "Expression measurement to return. "
            "TPM is recommended for an individual sample."
        ),
    ),
):
    """
    Extract selected gene-expression values from the local RNA-seq file.
    """

    gene_list = [
        gene.strip()
        for gene in genes.split(",")
        if gene.strip()
    ]

    try:
        result = get_gene_expression(
            genes=gene_list,
            expression_column=expression_column,
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

    return {
        "patient": submitter_id.strip().upper(),
        **result,
    }