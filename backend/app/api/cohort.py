from fastapi import APIRouter, HTTPException, Query

from backend.app.services.cohort import (
    CohortDiscoveryError,
    discover_tcga_brca_cohort,
)
from backend.app.services.cohort_builder import (
    DEFAULT_GENES,
    CohortBuildError,
    build_tcga_brca_pilot_cohort,
)
from backend.app.services.dataset_validator import (
    DatasetValidationError,
    validate_tcga_brca_dataset,
)
from backend.app.services.model_training import (
    ModelTrainingError,
    train_tcga_brca_baseline_model,
)
from backend.app.services.preprocessing import (
    DatasetPreprocessingError,
    preprocess_cohort_dataset,
    preprocess_tcga_brca_dataset,
)


router = APIRouter(
    prefix="/cohort",
    tags=["Cohort Builder"],
)


def _parse_gene_list(
    genes: str,
) -> list[str]:
    """
    Convert comma-separated gene symbols into a list.
    """

    return [
        gene.strip()
        for gene in genes.split(",")
        if gene.strip()
    ]


@router.get("/tcga-brca/discover")
def discover_breast_cancer_cohort(
    limit: int = Query(
        default=10,
        ge=1,
        le=500,
        description=(
            "Maximum number of eligible patients to return."
        ),
    ),
    require_rna: bool = Query(
        default=True,
        description=(
            "Only include patients with RNA-expression data."
        ),
    ),
    require_mutations: bool = Query(
        default=True,
        description=(
            "Only include patients with somatic-mutation data."
        ),
    ),
):
    """
    Discover TCGA-BRCA patients without downloading files.
    """

    try:
        return discover_tcga_brca_cohort(
            limit=limit,
            require_rna=require_rna,
            require_mutations=require_mutations,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except CohortDiscoveryError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected cohort-discovery error: "
                f"{exc}"
            ),
        ) from exc


@router.post("/tcga-brca/build-pilot")
def build_pilot_cohort(
    limit: int = Query(
        default=5,
        ge=1,
        le=25,
        description=(
            "Number of eligible patients to process."
        ),
    ),
    genes: str = Query(
        default=",".join(DEFAULT_GENES),
        description=(
            "Comma-separated genes to extract from RNA and "
            "mutation files."
        ),
    ),
):
    """
    Download and process a small machine-learning cohort.
    """

    try:
        return build_tcga_brca_pilot_cohort(
            limit=limit,
            genes=_parse_gene_list(genes),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except (
        CohortBuildError,
        CohortDiscoveryError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected cohort-build error: "
                f"{exc}"
            ),
        ) from exc


@router.get("/tcga-brca/validate-pilot")
def validate_pilot_cohort(
    dataset_path: str | None = Query(
        default=None,
        description=(
            "Optional path to a cohort CSV."
        ),
    ),
    genes: str = Query(
        default=",".join(DEFAULT_GENES),
        description=(
            "Comma-separated genes expected in the dataset."
        ),
    ),
):
    """
    Validate a cohort dataset before preprocessing.
    """

    try:
        return validate_tcga_brca_dataset(
            dataset_path=dataset_path,
            genes=_parse_gene_list(genes),
        )

    except DatasetValidationError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected dataset-validation error: "
                f"{exc}"
            ),
        ) from exc


@router.post("/preprocess")
def preprocess_cohort(
    dataset_path: str = Query(
        ...,
        description="Path to any compatible validated cohort CSV.",
    ),
    output_name: str | None = Query(
        default=None,
        description=(
            "Optional base name for generated files. Do not include a file "
            "extension. When omitted, the input dataset name is used."
        ),
    ),
    genes: str = Query(
        default=",".join(DEFAULT_GENES),
        description="Comma-separated genes expected in the dataset.",
    ),
):
    """Preprocess any compatible cohort into an ML-ready matrix."""

    try:
        resolved_output_path = None
        resolved_metadata_path = None

        if output_name:
            cleaned_name = output_name.strip()

            if not cleaned_name:
                raise ValueError("output_name cannot be blank.")

            invalid_characters = set('\\/:*?"<>|')
            if any(character in invalid_characters for character in cleaned_name):
                raise ValueError(
                    "output_name contains a character that is not allowed in "
                    "Windows file names."
                )

            if cleaned_name.lower().endswith((".csv", ".json")):
                raise ValueError(
                    "Provide output_name without a .csv or .json extension."
                )

            resolved_output_path = (
                f"data/processed/{cleaned_name}.csv"
                if cleaned_name.endswith("_ml_ready")
                else f"data/processed/{cleaned_name}_ml_ready.csv"
            )
            metadata_stem = (
                cleaned_name.removesuffix("_ml_ready")
                if cleaned_name.endswith("_ml_ready")
                else cleaned_name
            )
            resolved_metadata_path = (
                f"data/processed/{metadata_stem}_preprocessing.json"
            )

        return preprocess_cohort_dataset(
            dataset_path=dataset_path,
            output_path=resolved_output_path,
            metadata_path=resolved_metadata_path,
            genes=_parse_gene_list(genes),
        )

    except (DatasetPreprocessingError, DatasetValidationError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected dataset-preprocessing error: {exc}",
        ) from exc


@router.post("/tcga-brca/preprocess-pilot")
def preprocess_pilot_cohort(
    dataset_path: str | None = Query(
        default=(
            "data/datasets/"
            "tcga_brca_pilot_25_patients.csv"
        ),
        description=(
            "Path to the validated cohort CSV."
        ),
    ),
    output_path: str | None = Query(
        default=None,
        description=(
            "Optional path for the ML-ready CSV."
        ),
    ),
    metadata_path: str | None = Query(
        default=None,
        description=(
            "Optional path for preprocessing metadata."
        ),
    ),
    genes: str = Query(
        default=",".join(DEFAULT_GENES),
        description=(
            "Comma-separated genes expected in the dataset."
        ),
    ),
):
    """
    Transform a validated cohort into an ML-ready matrix.
    """

    try:
        return preprocess_tcga_brca_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            metadata_path=metadata_path,
            genes=_parse_gene_list(genes),
        )

    except (
        DatasetPreprocessingError,
        DatasetValidationError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected dataset-preprocessing error: "
                f"{exc}"
            ),
        ) from exc


@router.post("/tcga-brca/train-baseline")
def train_baseline_model(
    dataset_path: str | None = Query(
        default=(
            "data/processed/"
            "tcga_brca_pilot_25_patients_ml_ready.csv"
        ),
        description=(
            "Path to the preprocessed ML-ready CSV."
        ),
    ),
    model_path: str | None = Query(
        default=None,
        description=(
            "Optional output path for the trained model."
        ),
    ),
    report_path: str | None = Query(
        default=None,
        description=(
            "Optional output path for the evaluation report."
        ),
    ),
    random_state: int = Query(
        default=42,
        ge=0,
        description=(
            "Random seed used for reproducible "
            "cross-validation."
        ),
    ),
):
    """
    Train and evaluate the first baseline classification model.
    """

    try:
        return train_tcga_brca_baseline_model(
            dataset_path=dataset_path,
            model_path=model_path,
            report_path=report_path,
            random_state=random_state,
        )

    except ModelTrainingError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected baseline-model training error: "
                f"{exc}"
            ),
        ) from exc