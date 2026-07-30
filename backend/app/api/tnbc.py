from fastapi import APIRouter, HTTPException, Query

from backend.app.services.cohort import CohortDiscoveryError
from backend.app.services.cohort_builder import DEFAULT_GENES, CohortBuildError
from backend.app.services.tnbc_cohort_builder import (
    TNBCCohortBuildError,
    build_tcga_tnbc_cohort,
)
from backend.app.services.tnbc_discovery import (
    TNBCDiscoveryError,
    scan_tcga_brca_receptor_supplements,
)


router = APIRouter(
    prefix="/tnbc",
    tags=["TNBC Discovery and Cohort Builder"],
)


def _parse_gene_list(genes: str) -> list[str]:
    """Convert comma-separated gene symbols into a clean list."""

    return [
        gene.strip()
        for gene in genes.split(",")
        if gene.strip()
    ]


@router.get("/tcga-brca/scan-clinical-supplements")
def scan_clinical_supplements(
    limit: int = Query(
        default=25,
        ge=1,
        le=1000,
        description=(
            "Maximum number of TCGA-BRCA clinical XML supplements to inspect."
        ),
    ),
):
    """Scan TCGA-BRCA clinical supplements for ER, PR, and HER2."""

    try:
        return scan_tcga_brca_receptor_supplements(limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TNBCDiscoveryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected TNBC clinical-supplement scan error: {exc}",
        ) from exc


@router.post("/tcga-brca/build-cohort")
def build_tnbc_cohort(
    target_count: int | None = Query(
        default=None,
        ge=1,
        le=500,
        description=(
            "Number of molecularly eligible confirmed TNBC patients to "
            "attempt. Leave blank to process every eligible TNBC patient."
        ),
    ),
    clinical_file_limit: int = Query(
        default=1000,
        ge=1,
        le=1000,
        description=(
            "Number of TCGA-BRCA clinical XML supplements used to identify "
            "ER-negative, PR-negative, and HER2-negative cases."
        ),
    ),
    output_name: str | None = Query(
        default="tcga_tnbc_cohort",
        description="Filename stem for the TNBC dataset and manifest.",
    ),
    genes: str = Query(
        default=",".join(DEFAULT_GENES),
        description=(
            "Comma-separated genes to extract from RNA and mutation files."
        ),
    ),
):
    """
    Build a TNBC-specific molecular dataset from confirmed TCGA-BRCA cases.

    The endpoint rescans or reuses cached clinical supplements, intersects
    confirmed TNBC IDs with available RNA and mutation files, applies
    patient-specific MAF filtering, and writes a CSV plus JSON manifest.
    """

    try:
        return build_tcga_tnbc_cohort(
            target_count=target_count,
            genes=_parse_gene_list(genes),
            clinical_file_limit=clinical_file_limit,
            output_name=output_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (
        TNBCCohortBuildError,
        TNBCDiscoveryError,
        CohortBuildError,
        CohortDiscoveryError,
    ) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected TNBC cohort-build error: {exc}",
        ) from exc
