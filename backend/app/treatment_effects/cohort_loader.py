"""
HERMES 2.0 NeoTRIP Cohort Loader

Loads and validates the randomized NeoTRIP baseline cohort.

The canonical patient-level representation is:

    D_i = (X_i, T_i, Y_i, Z_i)

where:

    X_i = baseline transcriptomic state
    T_i = randomized treatment
    Y_i = pathologic complete response
    Z_i = contextual variables such as TNBC subtype and batch

Treatment encoding:
    0 = CT
    1 = CT/A

Outcome encoding:
    0 = residual disease
    1 = pCR
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from backend.app.treatment_effects.schema import (
    ClinicalOutcome,
    DataProvenance,
    DiseaseSetting,
    HermesPatientRecord,
    MolecularState,
    OutcomeType,
    SampleTimepoint,
    Treatment,
    TreatmentGroup,
)


DEFAULT_EXPRESSION_PATH = Path(
    "data/hermes2/cohorts/neotrip/"
    "GSE319641_NeoTRIP_baseline_D1C2_TPM_ComBat_all_samples.txt.gz"
)

DEFAULT_CLINICAL_PATH = Path(
    "data/hermes2/cohorts/neotrip/authors_code/data/"
    "NeoTRIP_baseline_D1C2_pheno_and_clinical.txt"
)


@dataclass
class NeoTRIPCohort:
    records: List[HermesPatientRecord]
    expression: pd.DataFrame
    clinical: pd.DataFrame

    @property
    def n_patients(self) -> int:
        return len(self.records)

    @property
    def n_genes(self) -> int:
        return self.expression.shape[1]


def _encode_treatment(arm: str) -> Treatment:
    arm = str(arm).strip()

    if arm == "CT":
        return Treatment(
            treatment_indicator=0,
            group=TreatmentGroup.CONTROL,
            ici_agent=None,
            chemotherapy=["carboplatin", "nab-paclitaxel"],
            treatment_arm_label="CT",
        )

    if arm == "CT/A":
        return Treatment(
            treatment_indicator=1,
            group=TreatmentGroup.ICI,
            ici_agent="atezolizumab",
            chemotherapy=["carboplatin", "nab-paclitaxel"],
            treatment_arm_label="CT/A",
        )

    raise ValueError(f"Unknown NeoTRIP treatment arm: {arm}")


def _encode_outcome(row: pd.Series) -> ClinicalOutcome:
    value = int(row["pCR.num"])

    return ClinicalOutcome(
        outcome_type=OutcomeType.PCR,
        binary_outcome=value,
        response_category=str(row["pCR"]),
    )


def load_neotrip_baseline(
    expression_path: Path = DEFAULT_EXPRESSION_PATH,
    clinical_path: Path = DEFAULT_CLINICAL_PATH,
) -> NeoTRIPCohort:
    """
    Load the 241-patient NeoTRIP pretreatment cohort.

    Returns one row per patient and one column per gene in the expression
    matrix, together with canonical HERMES patient records.
    """

    expression_path = Path(expression_path)
    clinical_path = Path(clinical_path)

    if not expression_path.exists():
        raise FileNotFoundError(
            f"NeoTRIP expression file not found: {expression_path}"
        )

    if not clinical_path.exists():
        raise FileNotFoundError(
            f"NeoTRIP clinical file not found: {clinical_path}"
        )

    clinical_all = pd.read_csv(
        clinical_path,
        sep="\t",
    )

    clinical = clinical_all.loc[
        clinical_all["Timepoint"].eq("Baseline")
    ].copy()

    if clinical["Patient_ID"].duplicated().any():
        duplicates = clinical.loc[
            clinical["Patient_ID"].duplicated(),
            "Patient_ID",
        ].tolist()

        raise ValueError(
            f"Duplicate baseline NeoTRIP patients detected: {duplicates}"
        )

    expression_raw = pd.read_csv(
        expression_path,
        sep="\t",
        compression="gzip",
    )

    required_gene_columns = [
        "HGNC_approved_symbol",
        "HGNC_id",
        "entrez_gene_id",
    ]

    missing_gene_columns = [
        column
        for column in required_gene_columns
        if column not in expression_raw.columns
    ]

    if missing_gene_columns:
        raise ValueError(
            "NeoTRIP expression matrix is missing expected gene columns: "
            f"{missing_gene_columns}"
        )

    baseline_columns = [
        column
        for column in expression_raw.columns
        if column.endswith("_Baseline")
    ]

    expression_patient_ids = {
        column.removesuffix("_Baseline")
        for column in baseline_columns
    }

    clinical_patient_ids = set(
        clinical["Patient_ID"].astype(str)
    )

    expression_only = sorted(
        expression_patient_ids - clinical_patient_ids
    )

    clinical_only = sorted(
        clinical_patient_ids - expression_patient_ids
    )

    if expression_only or clinical_only:
        raise ValueError(
            "NeoTRIP expression/clinical mismatch. "
            f"Expression-only patients: {expression_only}; "
            f"Clinical-only patients: {clinical_only}"
        )

    gene_symbols = (
        expression_raw["HGNC_approved_symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    valid_gene_mask = gene_symbols.ne("")

    expression_values = expression_raw.loc[
        valid_gene_mask,
        baseline_columns,
    ].copy()

    expression_values.index = gene_symbols.loc[
        valid_gene_mask
    ]

    # If a gene symbol appears more than once, average duplicate rows.
    expression_values = expression_values.groupby(
        level=0
    ).mean()

    expression = expression_values.T

    expression.index = [
        sample_id.removesuffix("_Baseline")
        for sample_id in expression.index
    ]

    expression.index.name = "Patient_ID"

    clinical = clinical.set_index("Patient_ID", drop=False)

    expression = expression.loc[
        clinical.index
    ]

    if expression.isna().any().any():
        n_missing = int(expression.isna().sum().sum())

        raise ValueError(
            f"NeoTRIP baseline expression contains "
            f"{n_missing} missing values."
        )

    records: List[HermesPatientRecord] = []

    for patient_id, row in clinical.iterrows():
        treatment = _encode_treatment(row["Arm"])
        outcome = _encode_outcome(row)

        sample_id = str(row["Sample_ID"])

        molecular_state = MolecularState(
            gene_expression_id=sample_id,
            clinical_features={
                "tnbc_type": str(row["TNBCtype"]),
                "batch_correlation": str(
                    row["Batch_correlation"]
                ),
            },
        )

        provenance = DataProvenance(
            cohort="NeoTRIP",
            study_accession="GSE319641",
            sample_accession=sample_id,
            platform="Illumina NextSeq 500",
            batch=str(row["Batch_correlation"]),
            source="GEO + published NeoTRIP analysis repository",
        )

        record = HermesPatientRecord(
            patient_id=str(patient_id),
            sample_id=sample_id,
            disease="TNBC",
            disease_setting=DiseaseSetting.EARLY,
            sample_timepoint=SampleTimepoint.PRETREATMENT,
            treatment=treatment,
            outcome=outcome,
            molecular_state=molecular_state,
            provenance=provenance,
            metadata={
                "original_arm": str(row["Arm"]),
                "original_pcr": str(row["pCR"]),
                "tnbc_type": str(row["TNBCtype"]),
            },
        )

        record.validate()
        records.append(record)

    cohort = NeoTRIPCohort(
        records=records,
        expression=expression,
        clinical=clinical,
    )

    validate_neotrip_cohort(cohort)

    return cohort


def validate_neotrip_cohort(
    cohort: NeoTRIPCohort,
) -> None:
    """
    Validate structural and randomized-treatment integrity.
    """

    if cohort.n_patients != 241:
        raise ValueError(
            f"Expected 241 NeoTRIP baseline patients, "
            f"found {cohort.n_patients}."
        )

    if cohort.expression.shape[0] != cohort.n_patients:
        raise ValueError(
            "Expression matrix row count does not match "
            "patient record count."
        )

    patient_ids = [
        record.patient_id
        for record in cohort.records
    ]

    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError(
            "Duplicate NeoTRIP patient records detected."
        )

    treatment_values = [
        record.treatment.treatment_indicator
        for record in cohort.records
        if record.treatment is not None
    ]

    outcome_values = [
        record.outcome.binary_outcome
        for record in cohort.records
        if record.outcome is not None
    ]

    if set(treatment_values) != {0, 1}:
        raise ValueError(
            "NeoTRIP cohort must contain both randomized treatment arms."
        )

    if set(outcome_values) != {0, 1}:
        raise ValueError(
            "NeoTRIP cohort must contain both pCR and residual disease."
        )


def summarize_neotrip_cohort(
    cohort: NeoTRIPCohort,
) -> dict:
    """
    Return publication-friendly cohort summary statistics.
    """

    clinical = cohort.clinical

    treatment_counts = (
        clinical["Arm"]
        .value_counts()
        .to_dict()
    )

    outcome_counts = (
        clinical["pCR"]
        .value_counts()
        .to_dict()
    )

    pcr_by_arm = (
        clinical
        .groupby("Arm")["pCR.num"]
        .mean()
        .to_dict()
    )

    return {
        "patients": cohort.n_patients,
        "genes": cohort.n_genes,
        "treatment_counts": treatment_counts,
        "outcome_counts": outcome_counts,
        "pcr_rate_by_arm": pcr_by_arm,
    }


if __name__ == "__main__":
    cohort = load_neotrip_baseline()
    summary = summarize_neotrip_cohort(cohort)

    print("=== HERMES 2.0 NeoTRIP Loader ===")
    print(f"Patients: {summary['patients']}")
    print(f"Genes: {summary['genes']}")
    print(
        f"Treatment counts: "
        f"{summary['treatment_counts']}"
    )
    print(
        f"Outcome counts: "
        f"{summary['outcome_counts']}"
    )
    print(
        f"pCR rate by arm: "
        f"{summary['pcr_rate_by_arm']}"
    )