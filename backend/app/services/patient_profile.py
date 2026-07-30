from typing import Any

from backend.app.services.gdc import get_tcga_brca_case
from backend.app.services.mutations import get_patient_mutations
from backend.app.services.rna import get_gene_expression


LOCAL_RNA_PATIENT_ID = "TCGA-3C-AAAU"

DEFAULT_PROFILE_GENES = [
    "TP53",
    "BRCA1",
    "BRCA2",
    "ERBB2",
    "CD274",
    "PIK3CA",
    "ESR1",
    "PGR",
    "EGFR",
    "PTEN",
    "RB1",
    "MYC",
]


def _extract_first_diagnosis(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract the first available diagnosis from a GDC case.
    """

    diagnoses = case.get("diagnoses") or []

    if not diagnoses:
        return {}

    diagnosis = diagnoses[0]

    return {
        "diagnosis_id": diagnosis.get("diagnosis_id"),
        "primary_diagnosis": diagnosis.get("primary_diagnosis"),
        "tumor_stage": diagnosis.get("tumor_stage"),
        "tumor_grade": diagnosis.get("tumor_grade"),
        "age_at_diagnosis": diagnosis.get("age_at_diagnosis"),
        "days_to_last_follow_up": diagnosis.get(
            "days_to_last_follow_up"
        ),
        "days_to_last_known_disease_status": diagnosis.get(
            "days_to_last_known_disease_status"
        ),
        "last_known_disease_status": diagnosis.get(
            "last_known_disease_status"
        ),
        "tissue_or_organ_of_origin": diagnosis.get(
            "tissue_or_organ_of_origin"
        ),
    }


def _extract_demographic(
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract demographic and survival-related information.
    """

    demographic = case.get("demographic") or {}

    days_to_birth = demographic.get("days_to_birth")
    age_at_index = None

    if isinstance(days_to_birth, (int, float)):
        age_at_index = round(abs(days_to_birth) / 365.25, 1)

    return {
        "gender": demographic.get("gender"),
        "race": demographic.get("race"),
        "ethnicity": demographic.get("ethnicity"),
        "vital_status": demographic.get("vital_status"),
        "days_to_birth": days_to_birth,
        "estimated_age_years": age_at_index,
        "days_to_death": demographic.get("days_to_death"),
    }


def _extract_samples(
    case: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return a simplified list of patient samples.
    """

    samples = case.get("samples") or []

    return [
        {
            "sample_id": sample.get("sample_id"),
            "submitter_id": sample.get("submitter_id"),
            "sample_type": sample.get("sample_type"),
            "tissue_type": sample.get("tissue_type"),
        }
        for sample in samples
    ]


def _simplify_expression(
    expression_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert the RNA service response into a compact gene-value map.
    """

    simplified_genes: dict[str, Any] = {}

    for gene, result in expression_result.get("genes", {}).items():
        simplified_genes[gene] = {
            "value": result.get("value"),
            "measurement": result.get("measurement"),
            "gene_id": result.get("gene_id"),
            "gene_type": result.get("gene_type"),
        }

    return {
        "source_file": expression_result.get("source_file"),
        "measurement": expression_result.get(
            "expression_measurement"
        ),
        "requested_gene_count": expression_result.get(
            "requested_gene_count"
        ),
        "found_gene_count": expression_result.get(
            "found_gene_count"
        ),
        "missing_genes": expression_result.get(
            "missing_genes",
            [],
        ),
        "genes": simplified_genes,
    }


def _simplify_mutations(
    mutation_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Return the mutation information needed for the patient profile.
    """

    return {
        "source_file": mutation_result.get("source_file"),
        "source_file_id": mutation_result.get("source_file_id"),
        "workflow_type": mutation_result.get("workflow_type"),
        "total_mutation_count": mutation_result.get(
            "total_patient_mutations_in_file"
        ),
        "returned_mutation_count": mutation_result.get(
            "returned_mutation_count"
        ),
        "mutated_gene_count": mutation_result.get(
            "mutated_gene_count"
        ),
        "mutated_genes": mutation_result.get(
            "mutated_genes",
            [],
        ),
        "mutations": mutation_result.get(
            "mutations",
            [],
        ),
    }


def build_patient_profile(
    submitter_id: str,
    genes: list[str] | None = None,
    mutation_max_results: int = 500,
) -> dict[str, Any]:
    """
    Build one unified clinical and molecular patient profile.

    The current RNA implementation uses one locally downloaded file
    belonging to TCGA-3C-AAAU. Other patients will be supported after
    the automated cohort downloader is implemented.
    """

    normalized_patient = submitter_id.strip().upper()

    if normalized_patient != LOCAL_RNA_PATIENT_ID:
        raise ValueError(
            "The current local RNA-seq file belongs to "
            f"{LOCAL_RNA_PATIENT_ID}. A profile cannot yet be built "
            f"for {normalized_patient} without downloading that "
            "patient's matching RNA-seq file."
        )

    selected_genes = genes or DEFAULT_PROFILE_GENES

    clinical_response = get_tcga_brca_case(
        submitter_id=normalized_patient,
    )

    clinical_hits = (
        clinical_response
        .get("data", {})
        .get("hits", [])
    )

    if not clinical_hits:
        raise FileNotFoundError(
            f"No TCGA-BRCA clinical record was found for "
            f"patient '{normalized_patient}'."
        )

    case = clinical_hits[0]

    expression_result = get_gene_expression(
        genes=selected_genes,
        expression_column="tpm_unstranded",
    )

    mutation_result = get_patient_mutations(
        submitter_id=normalized_patient,
        genes=None,
        max_results=mutation_max_results,
    )

    clinical_profile = {
        "case_id": case.get("case_id"),
        "submitter_id": case.get("submitter_id"),
        "primary_site": case.get("primary_site"),
        "disease_type": case.get("disease_type"),
        "demographic": _extract_demographic(case),
        "diagnosis": _extract_first_diagnosis(case),
        "samples": _extract_samples(case),
    }

    return {
        "patient": normalized_patient,
        "project": "TCGA-BRCA",
        "profile_status": "complete",
        "clinical": clinical_profile,
        "rna_expression": _simplify_expression(
            expression_result
        ),
        "somatic_mutations": _simplify_mutations(
            mutation_result
        ),
        "available_modalities": {
            "clinical": True,
            "rna_expression": True,
            "somatic_mutations": True,
            "copy_number": False,
            "methylation": False,
            "proteomics": False,
            "pathology": False,
        },
        "limitations": [
            (
                "RNA expression currently comes from one locally "
                "downloaded TCGA sample."
            ),
            (
                "The profile is research-oriented and must not be "
                "used for clinical decision-making."
            ),
            (
                "ER, PR, and HER2 receptor status have not yet been "
                "integrated."
            ),
        ],
    }