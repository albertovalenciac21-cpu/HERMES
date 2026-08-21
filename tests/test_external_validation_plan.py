"""
HERMES 2.0
Locked External Validation Plan Tests
=====================================
"""

from __future__ import annotations

from dataclasses import replace
import json

from backend.app.treatment_effects.external_validation_plan import (
    LOCKED_EXTERNAL_VALIDATION_PLAN,
    PRIMARY_PUBLIC_COHORT,
    SECONDARY_CONTROLLED_COHORT,
    export_external_validation_plan,
    hash_external_validation_plan,
    validate_external_validation_plan,
)


def test_primary_public_cohort_is_ispy2_and_cross_platform() -> None:
    assert PRIMARY_PUBLIC_COHORT.accession == "GSE194040"
    assert PRIMARY_PUBLIC_COHORT.cohort_id == "ISPY2_PEMBRO_TNBC"
    assert PRIMARY_PUBLIC_COHORT.access == "public"
    assert PRIMARY_PUBLIC_COHORT.direct_raw_scale_transport_allowed is False


def test_secondary_cohort_preserves_same_agent_validation_path() -> None:
    assert SECONDARY_CONTROLLED_COHORT.cohort_id == "IMPASSION031"
    assert SECONDARY_CONTROLLED_COHORT.accession == "EGAS50000000974"
    assert SECONDARY_CONTROLLED_COHORT.access == "controlled"
    assert "atezolizumab" in (
        SECONDARY_CONTROLLED_COHORT.experimental_treatment.lower()
    )


def test_locked_pathway_hypothesis_family_is_fixed() -> None:
    expected = {
        "HALLMARK_COAGULATION",
        "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_MYOGENESIS",
        "HALLMARK_APICAL_JUNCTION",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_KRAS_SIGNALING_UP",
    }

    assert set(
        LOCKED_EXTERNAL_VALIDATION_PLAN.locked_negative_pathway_hypotheses
    ) == expected

    assert LOCKED_EXTERNAL_VALIDATION_PLAN.hypothesis_fdr_threshold == 0.10


def test_plan_hash_is_deterministic_and_sensitive() -> None:
    first = hash_external_validation_plan()
    second = hash_external_validation_plan()

    changed = hash_external_validation_plan(
        replace(
            LOCKED_EXTERNAL_VALIDATION_PLAN,
            n_repeats=99,
        )
    )

    assert first == second
    assert len(first) == 64
    assert first != changed


def test_validation_guardrails_reject_invalid_transport() -> None:
    validate_external_validation_plan()

    invalid = replace(
        LOCKED_EXTERNAL_VALIDATION_PLAN,
        direct_neotrip_model_transport_to_ispy2=True,
    )

    try:
        validate_external_validation_plan(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid direct cross-platform transport was not rejected."
        )


def test_external_validation_plan_export_contract(tmp_path) -> None:
    generated = export_external_validation_plan(
        output_dir=tmp_path,
    )

    assert {
        "locked_external_validation_plan",
        "external_validation_manifest",
    }.issubset(generated)

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0

    manifest = json.loads(
        generated["external_validation_manifest"].read_text(
            encoding="utf-8"
        )
    )

    assert manifest["external_results_inspected_before_lock"] is False
    assert manifest["direct_neotrip_model_transport_to_ispy2"] is False
    assert manifest["clinical_claims_allowed"] is False
    assert len(manifest["plan_sha256"]) == 64