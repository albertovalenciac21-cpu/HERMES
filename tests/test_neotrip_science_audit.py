"""
HERMES 2.0
Locked NeoTRIP Scientific Cohort Audit Tests
============================================

These tests validate the prespecified science-phase cohort audit before
HERMES treatment-effect results are interpreted.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.treatment_effects.neotrip_science_audit import (
    LockedNeoTRIPAnalysisSpec,
    export_locked_neotrip_science_audit,
    run_locked_neotrip_science_audit,
)


def test_locked_neotrip_cohort_counts_and_rates() -> None:
    audit = run_locked_neotrip_science_audit()
    s = audit.cohort_summary

    assert s["patients"] == 241
    assert s["control_patients_CT"] == 122
    assert s["experimental_patients_CT_A"] == 119

    assert s["pcr_patients"] == 121
    assert s["residual_disease_patients"] == 120

    assert np.isclose(
        s["pcr_rate_CT"],
        58 / 122,
    )

    assert np.isclose(
        s["pcr_rate_CT_A"],
        63 / 119,
    )

    assert np.isclose(
        s["absolute_pcr_rate_difference_CT_A_minus_CT"],
        (63 / 119) - (58 / 122),
    )


def test_locked_neotrip_context_distributions() -> None:
    audit = run_locked_neotrip_science_audit()

    expected_subtypes = {
        "BL1": 65,
        "M": 56,
        "IM": 44,
        "UNS": 30,
        "LAR": 20,
        "BL2": 14,
        "MSL": 12,
    }

    for subtype, expected_n in expected_subtypes.items():
        assert int(
            audit.subtype_table.loc[
                subtype,
                "n",
            ]
        ) == expected_n

    expected_batches = {
        "Batch 3": 200,
        "Batch 1": 23,
        "Batch 2": 18,
    }

    for batch, expected_n in expected_batches.items():
        assert int(
            audit.batch_table.loc[
                batch,
                "n",
            ]
        ) == expected_n


def test_locked_neotrip_transcriptomic_and_hallmark_integrity() -> None:
    audit = run_locked_neotrip_science_audit()
    s = audit.cohort_summary

    assert s["processed_expression_genes"] > 0
    assert s["processed_expression_genes"] <= s["raw_expression_genes"]

    assert s["hallmark_sets_loaded"] == 50
    assert s["hallmark_sets_retained"] == 50

    assert audit.hallmark_coverage.shape[0] == 50
    assert audit.hallmark_coverage["retained"].all()

    assert pd.api.types.is_numeric_dtype(
        audit.pcr_by_arm["pcr_rate"]
    )


def test_locked_neotrip_all_integrity_checks_pass() -> None:
    audit = run_locked_neotrip_science_audit()

    assert audit.integrity_checks
    assert all(
        audit.integrity_checks.values()
    )

    assert audit.cohort_summary[
        "all_integrity_checks_passed"
    ] is True


def test_analysis_spec_is_locked_before_interpretation() -> None:
    spec = LockedNeoTRIPAnalysisSpec()

    assert spec.treatment_control == "CT"
    assert spec.treatment_experimental == "CT/A"
    assert spec.endpoint == "pathologic_complete_response"
    assert spec.biological_representation == "MSigDB Hallmark pathways"
    assert spec.minimum_genes_per_pathway == 3
    assert np.isclose(
        spec.minimum_pathway_gene_coverage,
        0.50,
    )
    assert spec.treatment_effect_engine_tag == "hermes-2.0-engine-v1.0"
    assert spec.analysis_scope == "research_internal_validation"
    assert spec.external_validation_required is True


def test_audit_export_contract(tmp_path) -> None:
    audit = run_locked_neotrip_science_audit()

    artifacts = export_locked_neotrip_science_audit(
        audit,
        tmp_path,
    )

    required = {
        "treatment_outcome_table",
        "pcr_by_arm",
        "tnbc_subtype_distribution",
        "batch_distribution",
        "clinical_missingness",
        "transcriptomic_qc",
        "hallmark_coverage",
        "cohort_audit_summary",
        "locked_analysis_spec",
    }

    assert required.issubset(
        artifacts
    )

    for path in artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    payload = json.loads(
        artifacts[
            "cohort_audit_summary"
        ].read_text(
            encoding="utf-8"
        )
    )

    assert payload[
        "cohort_summary"
    ]["patients"] == 241

    assert payload[
        "cohort_summary"
    ]["all_integrity_checks_passed"] is True

    assert payload[
        "analysis_spec"
    ]["treatment_effect_engine_tag"] == (
        "hermes-2.0-engine-v1.0"
    )