"""
HERMES 2.0
Primary NeoTRIP Scientific Analysis Tests
=========================================

The full locked primary plan is intentionally computationally intensive
(100 repeated cross-fits and 1000 feature-profile permutations). These tests
therefore verify the science-plan contract and run a lightweight end-to-end
version through the same production code path.
"""

from __future__ import annotations

import json

import numpy as np

from backend.app.treatment_effects.primary_neotrip_analysis import (
    PRIMARY_NEOTRIP_PLAN,
    PrimaryNeoTRIPAnalysisPlan,
    export_primary_neotrip_analysis,
    hash_analysis_plan,
    plan_to_engine_config,
    run_primary_neotrip_analysis,
)


def _lightweight_plan() -> PrimaryNeoTRIPAnalysisPlan:
    """
    Lightweight but structurally valid production-path configuration.

    Robustness requires at least two distinct model scenarios, so the test
    uses two C values while keeping every other expensive component small.
    """
    return PrimaryNeoTRIPAnalysisPlan(
        plan_name="hermes2_neotrip_test_lightweight",
        n_repeats=3,
        robustness_n_repeats=2,
        robustness_C_values=(0.03, 0.10),
        robustness_n_splits_values=(5,),
        robustness_modifier_perturbations=2,
        n_permutations=3,
        permutation_n_repeats=2,
    )


def test_primary_plan_is_prespecified_and_publication_oriented() -> None:
    plan = PRIMARY_NEOTRIP_PLAN

    assert plan.plan_name == "hermes2_neotrip_primary_locked_v1"
    assert plan.engine_tag == "hermes-2.0-engine-v1.0"

    assert plan.n_repeats == 100
    assert plan.n_splits == 5
    assert np.isclose(plan.regularization_C, 0.10)

    assert plan.permutation_mode == "feature_permutation"
    assert plan.n_permutations == 1000
    assert plan.permutation_n_repeats == 10

    assert plan.modifier_fdr_threshold == 0.10
    assert plan.external_validation_required is True
    assert plan.analysis_scope == "research_internal_validation"


def test_analysis_plan_hash_is_deterministic_and_sensitive() -> None:
    first = hash_analysis_plan(
        PRIMARY_NEOTRIP_PLAN
    )
    second = hash_analysis_plan(
        PRIMARY_NEOTRIP_PLAN
    )

    changed = hash_analysis_plan(
        PrimaryNeoTRIPAnalysisPlan(
            n_repeats=99
        )
    )

    assert first == second
    assert len(first) == 64
    assert first != changed


def test_plan_maps_exactly_to_engine_configuration() -> None:
    plan = PRIMARY_NEOTRIP_PLAN
    config = plan_to_engine_config(
        plan
    )

    assert config.n_repeats == plan.n_repeats
    assert config.n_splits == plan.n_splits
    assert config.regularization_C == plan.regularization_C

    assert config.run_modifier_discovery is True
    assert config.run_robustness is True
    assert config.run_permutation is True
    assert config.run_applicability is True

    assert config.n_permutations == plan.n_permutations
    assert config.permutation_mode == plan.permutation_mode


def test_lightweight_primary_analysis_runs_on_locked_neotrip() -> None:
    result = run_primary_neotrip_analysis(
        plan=_lightweight_plan()
    )

    assert result.dataset.n_patients == 241
    assert result.dataset.n_features == 50

    assert all(
        result.audit.integrity_checks.values()
    )

    assert result.hermes.modifiers is not None
    assert result.hermes.robustness is not None
    assert result.hermes.permutation is not None
    assert result.hermes.applicability is not None

    assert result.hermes.patient_table.index.equals(
        result.dataset.X.index
    )

    assert result.hermes.repeated_crossfit.ite_by_repeat.shape == (
        241,
        3,
    )

    assert np.isfinite(
        result.hermes.repeated_crossfit.ite_by_repeat.to_numpy(
            dtype=float
        )
    ).all()

    assert result.science_summary["patients"] == 241
    assert result.science_summary["biological_features"] == 50


def test_primary_analysis_top_modifier_contract() -> None:
    result = run_primary_neotrip_analysis(
        plan=_lightweight_plan()
    )

    assert len(result.top_modifiers) <= 15

    required = {
        "feature",
        "interaction_coefficient",
        "interaction_p_value",
        "interaction_fdr",
    }

    assert required.issubset(
        result.top_modifiers.columns
    )


def test_primary_analysis_export_contract(tmp_path) -> None:
    result = run_primary_neotrip_analysis(
        plan=_lightweight_plan()
    )

    artifacts = export_primary_neotrip_analysis(
        result,
        tmp_path,
    )

    required = {
        "primary_analysis_plan",
        "primary_science_summary",
        "analysis_manifest",
        "top_modifiers",
        "engine__patient_level_results",
        "engine__modifier_discovery",
        "engine__permutation_observed_vs_null",
        "engine__patient_robustness",
        "engine__applicability",
        "audit__cohort_audit_summary",
        "audit__locked_analysis_spec",
    }

    assert required.issubset(
        artifacts
    )

    for path in artifacts.values():
        assert path.exists()
        assert path.stat().st_size > 0

    manifest = json.loads(
        artifacts[
            "analysis_manifest"
        ].read_text(
            encoding="utf-8"
        )
    )

    assert manifest["patients"] == 241
    assert manifest["biological_features"] == 50
    assert manifest["all_locked_audit_checks_passed"] is True
    assert len(manifest["plan_sha256"]) == 64