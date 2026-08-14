"""
HERMES 2.0 Treatment-Effect Data Schema

Canonical representation of patients, tumor samples, treatments, outcomes,
and molecular features used by the HERMES treatment-effect framework.

Core estimand:

    tau(x) = E[Y(1) - Y(0) | X = x]

where:

    X = pretreatment patient/tumor state
    T = treatment assignment
    Y = observed clinical outcome

HERMES ultimately seeks to estimate patient-specific treatment effects
while preserving cohort provenance, biological context, and uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# ENUMERATIONS
# ---------------------------------------------------------------------


class DiseaseSetting(str, Enum):
    EARLY = "early"
    LOCALLY_ADVANCED = "locally_advanced"
    METASTATIC = "metastatic"
    UNKNOWN = "unknown"


class SampleTimepoint(str, Enum):
    PRETREATMENT = "pretreatment"
    ON_TREATMENT = "on_treatment"
    POST_TREATMENT = "post_treatment"
    PROGRESSION = "progression"
    UNKNOWN = "unknown"


class TreatmentGroup(str, Enum):
    CONTROL = "control"
    ICI = "ici"
    OTHER = "other"


class OutcomeType(str, Enum):
    PCR = "pcr"
    RESPONSE = "response"
    PFS = "pfs"
    OS = "os"
    EFS = "efs"
    DFS = "dfs"
    OTHER = "other"


# ---------------------------------------------------------------------
# TREATMENT
# ---------------------------------------------------------------------


@dataclass
class Treatment:
    """
    Observed treatment received by a patient.

    treatment_indicator is the mathematical T variable used in
    treatment-effect estimation.

        T = 0 -> control / chemotherapy backbone
        T = 1 -> chemotherapy + immune-checkpoint inhibition

    More complex treatment encodings can be introduced in HERMES 3.0.
    """

    treatment_indicator: int

    group: TreatmentGroup

    ici_agent: Optional[str] = None
    chemotherapy: List[str] = field(default_factory=list)

    treatment_arm_label: Optional[str] = None

    additional_agents: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.treatment_indicator not in (0, 1):
            raise ValueError(
                "treatment_indicator must currently be 0 or 1."
            )


# ---------------------------------------------------------------------
# CLINICAL OUTCOME
# ---------------------------------------------------------------------


@dataclass
class ClinicalOutcome:
    """
    Observed clinical outcome Y.

    Different trials may contribute different endpoints.
    Keeping these fields explicit prevents inappropriate mixing
    of binary and time-to-event outcomes.
    """

    outcome_type: OutcomeType

    binary_outcome: Optional[int] = None

    time_to_event: Optional[float] = None
    event_observed: Optional[int] = None

    response_category: Optional[str] = None

    units: Optional[str] = None

    def validate(self) -> None:
        if self.binary_outcome is not None:
            if self.binary_outcome not in (0, 1):
                raise ValueError(
                    "binary_outcome must be 0 or 1."
                )

        if self.event_observed is not None:
            if self.event_observed not in (0, 1):
                raise ValueError(
                    "event_observed must be 0 or 1."
                )

        if self.time_to_event is not None:
            if self.time_to_event < 0:
                raise ValueError(
                    "time_to_event cannot be negative."
                )


# ---------------------------------------------------------------------
# MOLECULAR STATE X
# ---------------------------------------------------------------------


@dataclass
class MolecularState:
    """
    Molecular representation of the tumor.

    This object represents X in the HERMES framework.

    Raw high-dimensional data should generally not be stored directly
    inside this object. Instead, identifiers and derived biological
    representations can be attached here.
    """

    gene_expression_id: Optional[str] = None
    mutation_profile_id: Optional[str] = None
    copy_number_profile_id: Optional[str] = None

    pathway_features: Dict[str, float] = field(default_factory=dict)
    immune_features: Dict[str, float] = field(default_factory=dict)
    tumor_features: Dict[str, float] = field(default_factory=dict)

    clinical_features: Dict[str, Any] = field(default_factory=dict)

    embedding: Optional[List[float]] = None


# ---------------------------------------------------------------------
# DATA PROVENANCE
# ---------------------------------------------------------------------


@dataclass
class DataProvenance:
    """
    Tracks where every HERMES observation originated.

    This becomes critical for:
        - external validation
        - batch-effect analysis
        - leakage prevention
        - reproducibility
        - publication reporting
    """

    cohort: str

    study_accession: Optional[str] = None
    sample_accession: Optional[str] = None

    platform: Optional[str] = None
    batch: Optional[str] = None

    source: Optional[str] = None
    source_version: Optional[str] = None

    notes: Optional[str] = None


# ---------------------------------------------------------------------
# CANONICAL HERMES PATIENT RECORD
# ---------------------------------------------------------------------


@dataclass
class HermesPatientRecord:
    """
    Canonical patient-level record used by HERMES.

    Conceptually:

        D_i = (X_i, T_i, Y_i)

    where:

        X_i = pretreatment molecular/clinical state
        T_i = observed treatment
        Y_i = observed outcome

    Randomized trial cohorts allow HERMES to learn treatment
    interactions rather than prognosis alone.
    """

    patient_id: str
    sample_id: str

    disease: str = "TNBC"

    disease_setting: DiseaseSetting = DiseaseSetting.UNKNOWN
    sample_timepoint: SampleTimepoint = SampleTimepoint.UNKNOWN

    treatment: Optional[Treatment] = None
    outcome: Optional[ClinicalOutcome] = None

    molecular_state: MolecularState = field(
        default_factory=MolecularState
    )

    provenance: Optional[DataProvenance] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.patient_id:
            raise ValueError("patient_id cannot be empty.")

        if not self.sample_id:
            raise ValueError("sample_id cannot be empty.")

        if self.treatment is not None:
            self.treatment.validate()

        if self.outcome is not None:
            self.outcome.validate()


# ---------------------------------------------------------------------
# TREATMENT-EFFECT OUTPUT
# ---------------------------------------------------------------------


@dataclass
class TreatmentEffectEstimate:
    """
    Standard HERMES treatment-effect prediction.

    For binary outcomes:

        mu_1(x) = P(Y=1 | X=x, T=1)
        mu_0(x) = P(Y=1 | X=x, T=0)

        tau(x) = mu_1(x) - mu_0(x)

    tau > 0 suggests predicted incremental benefit from treatment 1.
    """

    patient_id: str

    predicted_outcome_treated: float
    predicted_outcome_control: float

    treatment_effect: float

    uncertainty: Optional[float] = None

    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    out_of_distribution_score: Optional[float] = None

    mechanism_scores: Dict[str, float] = field(default_factory=dict)

    model_version: Optional[str] = None

    def validate(self) -> None:
        expected_effect = (
            self.predicted_outcome_treated
            - self.predicted_outcome_control
        )

        tolerance = 1e-8

        if abs(expected_effect - self.treatment_effect) > tolerance:
            raise ValueError(
                "treatment_effect must equal "
                "predicted_outcome_treated - predicted_outcome_control."
            )