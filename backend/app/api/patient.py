from fastapi import APIRouter, HTTPException, Query

from backend.app.services.patient_profile import (
    DEFAULT_PROFILE_GENES,
    build_patient_profile,
)


router = APIRouter(
    prefix="/patient",
    tags=["Unified Patient Profile"],
)


@router.get("/{submitter_id}/profile")
def unified_patient_profile(
    submitter_id: str,
    genes: str = Query(
        default=",".join(DEFAULT_PROFILE_GENES),
        description=(
            "Comma-separated genes to include in the RNA-expression "
            "section. Example: TP53,BRCA1,ERBB2,CD274"
        ),
    ),
    mutation_max_results: int = Query(
        default=500,
        ge=1,
        le=5000,
        description=(
            "Maximum number of somatic mutations to include."
        ),
    ),
):
    """
    Build a unified clinical, RNA-expression, and mutation profile.
    """

    gene_list = [
        gene.strip()
        for gene in genes.split(",")
        if gene.strip()
    ]

    try:
        return build_patient_profile(
            submitter_id=submitter_id,
            genes=gene_list,
            mutation_max_results=mutation_max_results,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc