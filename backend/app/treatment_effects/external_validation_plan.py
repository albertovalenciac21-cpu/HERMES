"""
HERMES 2.0
Locked External Validation Plan
===============================

Purpose
-------
Freeze the external-validation strategy BEFORE inspecting external treatment-
effect results.

The primary public replication cohort is I-SPY2 (GSE194040), restricted to
triple-negative breast cancer (HR-negative/HER2-negative), comparing
pembrolizumab + chemotherapy with the contemporaneous chemotherapy control
and using pretreatment gene expression with pCR as the endpoint.

Why this is a replication rather than direct model transport
--------------------------------------------------------------
NeoTRIP and I-SPY2 use different transcriptomic platforms and different ICI/
chemotherapy regimens. NeoTRIP is RNA-seq and uses atezolizumab; the public
I-SPY2 resource uses Agilent expression arrays and the relevant arm uses
pembrolizumab.

Therefore HERMES 2.0 will NOT apply NeoTRIP raw-gene means/SDs directly to
I-SPY2. That would create an invalid cross-platform coordinate system.

Instead the public I-SPY2 analysis is prespecified as:

    external methodological + biological replication

It will:
    * use the same HERMES treatment-effect architecture;
    * preserve pCR and randomized treatment contrast;
    * use platform-appropriate pathway representations;
    * test locked biological hypotheses from NeoTRIP;
    * use independent cross-fitting and permutation validation;
    * report direction/concordance rather than claiming coefficient transport.

A future same-agent / controlled-access cohort such as IMpassion031 can serve
as a stronger transport/generalization test if suitable molecular data access
is obtained.

No external result may be used to change the locked NeoTRIP primary analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalCohortSpec:
    cohort_id: str
    study_name: str
    accession: str
    access: str
    disease: str
    setting: str
    sample_timepoint: str
    experimental_treatment: str
    control_treatment: str
    endpoint: str
    transcriptomic_platform: str
    validation_role: str
    direct_raw_scale_transport_allowed: bool
    notes: str


@dataclass(frozen=True)
class LockedExternalValidationPlan:
    """Prespecified HERMES 2.0 external-validation contract."""

    plan_name: str = "hermes2_external_validation_locked_v1"
    source_primary_plan: str = "hermes2_neotrip_primary_locked_v1"
    source_engine_tag: str = "hermes-2.0-engine-v1.0"

    primary_public_cohort_id: str = "ISPY2_PEMBRO_TNBC"
    secondary_controlled_cohort_id: str = "IMPASSION031"

    primary_endpoint: str = "pCR"
    primary_estimand: str = (
        "heterogeneity in incremental pCR benefit from adding an immune "
        "checkpoint inhibitor to neoadjuvant chemotherapy"
    )

    # Cross-platform external replication rules.
    representation_strategy: str = (
        "platform-appropriate pathway scoring; no direct NeoTRIP raw-gene "
        "mean/SD transport to microarray data"
    )
    primary_replication_type: str = (
        "methodological_and_biological_replication"
    )
    direct_neotrip_model_transport_to_ispy2: bool = False

    n_repeats: int = 100
    n_splits: int = 5
    regularization_C: float = 0.10
    max_iter: int = 10000
    base_random_state: int = 42

    n_permutations: int = 1000
    permutation_n_repeats: int = 10
    permutation_mode: str = "feature_permutation"

    # These hypotheses were identified in the already-locked NeoTRIP
    # biological characterization and are fixed before external analysis.
    locked_negative_pathway_hypotheses: tuple[str, ...] = (
        "HALLMARK_COAGULATION",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_MYOGENESIS",
        "HALLMARK_APICAL_JUNCTION",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_KRAS_SIGNALING_UP",
    )

    primary_directional_hypothesis: str = (
        "Higher activity of the locked mesenchymal/stromal pathways is "
        "associated with lower estimated incremental ICI benefit."
    )

    multiplicity_strategy: str = (
        "Benjamini-Hochberg FDR within the locked pathway hypothesis family; "
        "all remaining Hallmark analyses labeled exploratory"
    )
    hypothesis_fdr_threshold: float = 0.10

    success_criteria: tuple[str, ...] = (
        "Primary public cohort passes treatment/outcome/sample integrity audit.",
        "External analysis uses no NeoTRIP outcome information for fitting.",
        "Locked pathway-hypothesis directions are reported regardless of significance.",
        "Permutation-null results are reported for heterogeneity statistics.",
        "Cross-fit stability and applicability are reported.",
        "No post-hoc pathway substitution is allowed in the confirmatory hypothesis family.",
    )

    analysis_scope: str = "external_research_validation"
    clinical_claims_allowed: bool = False


PRIMARY_PUBLIC_COHORT = ExternalCohortSpec(
    cohort_id="ISPY2_PEMBRO_TNBC",
    study_name="I-SPY2 pembrolizumab neoadjuvant cohort",
    accession="GSE194040",
    access="public",
    disease="triple-negative breast cancer",
    setting="high-risk early breast cancer / neoadjuvant",
    sample_timepoint="pretreatment",
    experimental_treatment=(
        "paclitaxel + pembrolizumab followed by doxorubicin/cyclophosphamide"
    ),
    control_treatment=(
        "paclitaxel followed by doxorubicin/cyclophosphamide"
    ),
    endpoint="pathologic complete response",
    transcriptomic_platform="Agilent 44K gene-expression array",
    validation_role="primary public methodological_and_biological_replication",
    direct_raw_scale_transport_allowed=False,
    notes=(
        "Restrict to HR-negative/HER2-negative patients. Published analyses "
        "of this public resource have used 29 TNBC pembrolizumab patients and "
        "56 TNBC chemotherapy controls; the loader must verify counts from "
        "the downloaded source rather than hard-code them."
    ),
)


SECONDARY_CONTROLLED_COHORT = ExternalCohortSpec(
    cohort_id="IMPASSION031",
    study_name="IMpassion031",
    accession="EGAS50000000974",
    access="controlled",
    disease="triple-negative breast cancer",
    setting="stage II/III early TNBC / neoadjuvant",
    sample_timepoint="baseline",
    experimental_treatment="atezolizumab + neoadjuvant chemotherapy",
    control_treatment="placebo + neoadjuvant chemotherapy",
    endpoint="pathologic complete response",
    transcriptomic_platform="controlled-access biomarker dataset",
    validation_role="future same_agent_external_validation",
    direct_raw_scale_transport_allowed=False,
    notes=(
        "Randomized phase III atezolizumab study and therefore scientifically "
        "closer to NeoTRIP. Molecular/individual-level access requires the "
        "relevant controlled-data agreements."
    ),
)


LOCKED_EXTERNAL_VALIDATION_PLAN = LockedExternalValidationPlan()


def external_validation_plan_payload(
    plan: LockedExternalValidationPlan = LOCKED_EXTERNAL_VALIDATION_PLAN,
) -> dict[str, Any]:
    """Return the complete machine-readable external-validation contract."""

    return {
        "plan": asdict(plan),
        "cohorts": {
            PRIMARY_PUBLIC_COHORT.cohort_id: asdict(PRIMARY_PUBLIC_COHORT),
            SECONDARY_CONTROLLED_COHORT.cohort_id: asdict(
                SECONDARY_CONTROLLED_COHORT
            ),
        },
    }


def hash_external_validation_plan(
    plan: LockedExternalValidationPlan = LOCKED_EXTERNAL_VALIDATION_PLAN,
) -> str:
    """Deterministic SHA-256 fingerprint of the locked validation plan."""

    payload = external_validation_plan_payload(plan)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def validate_external_validation_plan(
    plan: LockedExternalValidationPlan = LOCKED_EXTERNAL_VALIDATION_PLAN,
) -> None:
    """Fail loudly if key scientific guardrails are weakened."""

    if plan.direct_neotrip_model_transport_to_ispy2:
        raise ValueError(
            "Direct NeoTRIP raw-scale model transport to I-SPY2 is prohibited."
        )

    if PRIMARY_PUBLIC_COHORT.direct_raw_scale_transport_allowed:
        raise ValueError(
            "Cross-platform raw-scale transport must remain disabled."
        )

    if len(plan.locked_negative_pathway_hypotheses) < 2:
        raise ValueError(
            "External validation requires a nontrivial locked hypothesis family."
        )

    if len(set(plan.locked_negative_pathway_hypotheses)) != len(
        plan.locked_negative_pathway_hypotheses
    ):
        raise ValueError("Locked pathway hypotheses must be unique.")

    if not (0.0 < plan.hypothesis_fdr_threshold < 1.0):
        raise ValueError("Invalid hypothesis FDR threshold.")

    if plan.n_repeats < 2:
        raise ValueError("External repeated cross-fitting requires >=2 repeats.")

    if plan.n_splits < 2:
        raise ValueError("External cross-fitting requires >=2 folds.")

    if plan.n_permutations < 100:
        raise ValueError(
            "Locked external validation requires at least 100 permutations."
        )

    if plan.clinical_claims_allowed:
        raise ValueError(
            "Clinical claims are not allowed from this research validation."
        )


def export_external_validation_plan(
    output_dir: str | Path = "outputs/hermes2/external_validation",
    plan: LockedExternalValidationPlan = LOCKED_EXTERNAL_VALIDATION_PLAN,
) -> dict[str, Path]:
    """Write the plan before any external outcomes are analyzed."""

    validate_external_validation_plan(plan)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = external_validation_plan_payload(plan)
    plan_hash = hash_external_validation_plan(plan)

    plan_path = output_dir / "locked_external_validation_plan.json"
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest = {
        "plan_name": plan.plan_name,
        "plan_sha256": plan_hash,
        "source_primary_plan": plan.source_primary_plan,
        "source_engine_tag": plan.source_engine_tag,
        "primary_public_cohort": plan.primary_public_cohort_id,
        "secondary_controlled_cohort": plan.secondary_controlled_cohort_id,
        "external_results_inspected_before_lock": False,
        "direct_neotrip_model_transport_to_ispy2": (
            plan.direct_neotrip_model_transport_to_ispy2
        ),
        "clinical_claims_allowed": plan.clinical_claims_allowed,
    }

    manifest_path = output_dir / "external_validation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "locked_external_validation_plan": plan_path,
        "external_validation_manifest": manifest_path,
    }


def main() -> None:
    validate_external_validation_plan()

    generated = export_external_validation_plan()

    print("=== HERMES 2.0 LOCKED EXTERNAL VALIDATION PLAN ===")
    print()
    print(f"Plan: {LOCKED_EXTERNAL_VALIDATION_PLAN.plan_name}")
    print(
        "Plan SHA256: "
        f"{hash_external_validation_plan(LOCKED_EXTERNAL_VALIDATION_PLAN)}"
    )
    print()
    print(
        "Primary public replication cohort: "
        f"{PRIMARY_PUBLIC_COHORT.study_name} "
        f"({PRIMARY_PUBLIC_COHORT.accession})"
    )
    print(
        "Validation role: "
        f"{PRIMARY_PUBLIC_COHORT.validation_role}"
    )
    print(
        "Direct NeoTRIP raw-scale transport allowed: "
        f"{PRIMARY_PUBLIC_COHORT.direct_raw_scale_transport_allowed}"
    )
    print()
    print(
        "Secondary controlled cohort: "
        f"{SECONDARY_CONTROLLED_COHORT.study_name} "
        f"({SECONDARY_CONTROLLED_COHORT.accession})"
    )
    print()
    print("Locked directional pathway hypotheses:")
    for pathway in (
        LOCKED_EXTERNAL_VALIDATION_PLAN.locked_negative_pathway_hypotheses
    ):
        print(f"  - {pathway}")
    print()
    print(
        "IMPORTANT: I-SPY2 is a cross-platform, cross-ICI replication. "
        "It is not direct transport validation of NeoTRIP-fitted raw-scale "
        "coefficients."
    )
    print(
        "The external hypothesis family is frozen before external outcome "
        "analysis and may not be substituted post hoc."
    )
    print()
    print("Artifacts:")
    for name, path in generated.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()