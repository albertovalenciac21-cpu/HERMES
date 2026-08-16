"""Tests for the unified HERMES 2.0 analysis pipeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from backend.app.treatment_effects.analysis_pipeline import (
    HermesAnalysisConfig,
    export_hermes_analysis,
    run_hermes_analysis,
)


def _synthetic_trial(
    *,
    n_per_joint_stratum: int = 24,
    n_features: int = 6,
    random_state: int = 2026,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    treatment_values: list[int] = []
    outcome_values: list[int] = []
    for treatment in (0, 1):
        for outcome in (0, 1):
            treatment_values.extend([treatment] * n_per_joint_stratum)
            outcome_values.extend([outcome] * n_per_joint_stratum)

    n = len(treatment_values)
    index = pd.Index([f"SIM_{i:04d}" for i in range(n)], name="Patient_ID")

    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        index=index,
        columns=[f"PATHWAY_{i:02d}" for i in range(n_features)],
    )
    X["PATHWAY_00"] += 0.35 * np.asarray(outcome_values)
    X["PATHWAY_01"] += 0.20 * np.asarray(treatment_values)

    treatment = pd.Series(treatment_values, index=index, name="T", dtype=int)
    outcome = pd.Series(outcome_values, index=index, name="Y", dtype=int)
    metadata = pd.DataFrame(
        {"synthetic_group": [f"G{i % 3}" for i in range(n)]},
        index=index,
    )
    return X, treatment, outcome, metadata


def _light_config() -> HermesAnalysisConfig:
    return HermesAnalysisConfig(
        n_repeats=3,
        n_splits=4,
        run_modifier_discovery=False,
        run_robustness=False,
        run_permutation=False,
        run_applicability=True,
    )


def test_core_pipeline_patient_alignment_and_outputs() -> None:
    X, treatment, outcome, metadata = _synthetic_trial()
    result = run_hermes_analysis(
        X,
        treatment,
        outcome,
        metadata=metadata,
        config=_light_config(),
    )

    assert result.patient_table.index.equals(X.index)
    assert result.repeated_crossfit.ite_by_repeat.shape == (len(X), 3)
    assert result.uncertainty.patient_table.shape[0] == len(X)
    assert result.applicability is not None
    assert result.patient_table.shape[0] == len(X)
    assert any(column.startswith("metadata__") for column in result.patient_table.columns)
    assert any(column.startswith("crossfit__") for column in result.patient_table.columns)
    assert any(column.startswith("uncertainty__") for column in result.patient_table.columns)
    assert any(column.startswith("applicability__") for column in result.patient_table.columns)
    assert result.summary["patients"] == len(X)
    assert result.summary["biological_features"] == X.shape[1]


def test_pipeline_is_deterministic() -> None:
    X, treatment, outcome, metadata = _synthetic_trial()
    config = _light_config()

    first = run_hermes_analysis(X, treatment, outcome, metadata=metadata, config=config)
    second = run_hermes_analysis(X, treatment, outcome, metadata=metadata, config=config)

    pd.testing.assert_frame_equal(first.patient_table, second.patient_table)
    pd.testing.assert_frame_equal(
        first.repeated_crossfit.ite_by_repeat,
        second.repeated_crossfit.ite_by_repeat,
    )
    assert first.summary == second.summary


def test_optional_modules_can_be_disabled() -> None:
    X, treatment, outcome, _ = _synthetic_trial()
    config = HermesAnalysisConfig(
        n_repeats=2,
        n_splits=3,
        run_modifier_discovery=False,
        run_robustness=False,
        run_permutation=False,
        run_applicability=False,
    )

    result = run_hermes_analysis(X, treatment, outcome, config=config)

    assert result.modifiers is None
    assert result.robustness is None
    assert result.permutation is None
    assert result.applicability is None
    assert result.applicability_reference is None


def test_full_validation_stack_runs_on_small_synthetic_data() -> None:
    X, treatment, outcome, _ = _synthetic_trial(n_per_joint_stratum=20, n_features=4)
    config = HermesAnalysisConfig(
        n_repeats=2,
        n_splits=3,
        run_modifier_discovery=True,
        run_robustness=True,
        robustness_C_values=(0.05, 0.10),
        robustness_n_splits_values=(3,),
        robustness_n_repeats=2,
        robustness_modifier_perturbations=2,
        robustness_modifier_subsample_fraction=0.80,
        run_permutation=True,
        n_permutations=2,
        permutation_n_repeats=2,
        run_applicability=True,
    )

    result = run_hermes_analysis(X, treatment, outcome, config=config)

    assert result.modifiers is not None
    assert result.robustness is not None
    assert result.permutation is not None
    assert result.applicability is not None
    assert result.robustness.patient_robustness.shape[0] == len(X)
    assert result.permutation.null_statistics.shape[0] == 2
    assert "fraction_robust_patients" in result.summary


def test_export_writes_reproducible_artifacts(tmp_path) -> None:
    X, treatment, outcome, metadata = _synthetic_trial()
    result = run_hermes_analysis(
        X,
        treatment,
        outcome,
        metadata=metadata,
        config=_light_config(),
    )

    exported = export_hermes_analysis(result, tmp_path / "hermes_output")

    assert "patient_level_results" in exported
    assert "analysis_summary" in exported
    assert exported["patient_level_results"].exists()
    assert exported["analysis_summary"].exists()

    payload = json.loads(exported["analysis_summary"].read_text(encoding="utf-8"))
    assert payload["summary"]["patients"] == len(X)
    assert payload["config"]["n_repeats"] == 3


def test_invalid_configuration_is_rejected() -> None:
    X, treatment, outcome, _ = _synthetic_trial()

    with pytest.raises(ValueError, match="n_repeats must be at least 2"):
        run_hermes_analysis(
            X,
            treatment,
            outcome,
            config=HermesAnalysisConfig(n_repeats=1),
        )