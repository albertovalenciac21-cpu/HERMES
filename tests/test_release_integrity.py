"""
HERMES 2.0 — Final Release / End-to-End Integrity Gate
======================================================

This test exercises the canonical NeoTRIP -> Hallmark -> repeated cross-fit ->
uncertainty -> applicability -> unified export path using the real development
dataset available in the HERMES repository.

It is intentionally a release-integrity test, not evidence of external
generalizability, causal identification beyond the randomized-trial design,
or clinical utility.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.treatment_effects.analysis_pipeline import (
    HermesAnalysisConfig,
    export_hermes_analysis,
    run_neotrip_hermes_analysis,
)


def _release_config() -> HermesAnalysisConfig:
    """Small but genuine end-to-end configuration for the release gate."""
    return HermesAnalysisConfig(
        n_repeats=3,
        n_splits=5,
        regularization_C=0.10,
        max_iter=10000,
        base_random_state=42,
        run_modifier_discovery=False,
        run_robustness=False,
        run_permutation=False,
        run_applicability=True,
    )


def _assert_all_numeric_finite(frame: pd.DataFrame, *, name: str) -> None:
    numeric = frame.select_dtypes(include=[np.number])
    assert not numeric.empty, f"{name} contains no numeric outputs to validate."
    assert np.isfinite(numeric.to_numpy(dtype=float)).all(), (
        f"{name} contains NaN or infinite numeric values."
    )


def test_neotrip_release_gate_end_to_end(tmp_path) -> None:
    """
    Final HERMES 2.0 computational-engine integrity gate.

    Verifies:
      * canonical NeoTRIP construction;
      * exact patient/order conservation;
      * binary treatment/outcome integrity;
      * repeated out-of-fold ITE dimensions;
      * finite patient-level numerical outputs;
      * applicability output alignment;
      * deterministic rerun under fixed seeds;
      * reproducible CSV/JSON export schema.
    """
    config = _release_config()

    dataset, first = run_neotrip_hermes_analysis(config=config)

    # ---- Canonical dataset integrity -------------------------------------
    dataset.validate()

    assert dataset.n_patients == 241
    assert dataset.n_features == 50
    assert dataset.X.index.is_unique
    assert dataset.X.columns.is_unique
    assert dataset.X.index.equals(dataset.T.index)
    assert dataset.X.index.equals(dataset.Y.index)
    assert dataset.X.index.equals(dataset.metadata.index)
    assert set(dataset.T.unique()) == {0, 1}
    assert set(dataset.Y.unique()) == {0, 1}
    assert np.isfinite(dataset.X.to_numpy(dtype=float)).all()

    # ---- Cross-fit / patient conservation -------------------------------
    assert first.patient_table.index.equals(dataset.X.index)
    assert first.repeated_crossfit.ite_by_repeat.index.equals(dataset.X.index)
    assert first.repeated_crossfit.ite_by_repeat.shape == (
        dataset.n_patients,
        config.n_repeats,
    )
    assert first.uncertainty.patient_table.index.equals(dataset.X.index)

    assert first.applicability is not None
    assert first.applicability_reference is not None
    assert first.applicability.patient_table.index.equals(dataset.X.index)

    # No patient may disappear or be duplicated in the unified output.
    assert len(first.patient_table) == dataset.n_patients
    assert first.patient_table.index.is_unique

    # ---- Numerical integrity --------------------------------------------
    _assert_all_numeric_finite(
        first.repeated_crossfit.ite_by_repeat,
        name="ITE-by-repeat matrix",
    )
    _assert_all_numeric_finite(
        first.repeated_crossfit.patient_summary,
        name="cross-fit patient summary",
    )
    _assert_all_numeric_finite(
        first.uncertainty.patient_table,
        name="uncertainty table",
    )
    _assert_all_numeric_finite(
        first.applicability.patient_table,
        name="applicability table",
    )

    assert np.isfinite(float(first.summary["cohort_mean_ite"]))
    assert np.isfinite(float(first.summary["cohort_median_ite"]))
    assert np.isfinite(float(first.summary["mean_patient_ite_sd"]))

    assert first.summary["patients"] == dataset.n_patients
    assert first.summary["biological_features"] == dataset.n_features
    assert first.summary["dataset_source"] == "NeoTRIP_baseline"
    assert first.summary["analysis_scope"] == "research_internal_validation"

    # ---- Deterministic reproducibility ----------------------------------
    dataset_second, second = run_neotrip_hermes_analysis(config=config)

    pd.testing.assert_index_equal(dataset.X.index, dataset_second.X.index)
    pd.testing.assert_index_equal(dataset.X.columns, dataset_second.X.columns)
    pd.testing.assert_frame_equal(dataset.X, dataset_second.X)
    pd.testing.assert_series_equal(dataset.T, dataset_second.T)
    pd.testing.assert_series_equal(dataset.Y, dataset_second.Y)

    pd.testing.assert_frame_equal(
        first.repeated_crossfit.ite_by_repeat,
        second.repeated_crossfit.ite_by_repeat,
    )
    pd.testing.assert_frame_equal(
        first.patient_table,
        second.patient_table,
    )
    assert first.summary == second.summary

    # ---- Export contract -------------------------------------------------
    output_dir = tmp_path / "hermes_2_release_gate"
    exported = export_hermes_analysis(first, output_dir)

    required_artifacts = {
        "patient_level_results",
        "ite_by_repeat",
        "repeat_summary",
        "uncertainty",
        "applicability",
        "applicability_feature_z_scores",
        "analysis_summary",
    }
    assert required_artifacts.issubset(exported)

    for artifact in required_artifacts:
        assert exported[artifact].exists()
        assert exported[artifact].stat().st_size > 0

    exported_patient_table = pd.read_csv(
        exported["patient_level_results"],
        index_col=0,
    )
    assert exported_patient_table.shape == first.patient_table.shape
    assert exported_patient_table.index.astype(str).tolist() == (
        first.patient_table.index.astype(str).tolist()
    )
    assert exported_patient_table.columns.tolist() == (
        first.patient_table.columns.tolist()
    )

    payload = json.loads(
        exported["analysis_summary"].read_text(encoding="utf-8")
    )
    assert payload["summary"]["patients"] == 241
    assert payload["summary"]["biological_features"] == 50
    assert payload["summary"]["dataset_source"] == "NeoTRIP_baseline"
    assert payload["summary"]["analysis_scope"] == "research_internal_validation"
    assert payload["config"]["n_repeats"] == config.n_repeats
    assert payload["config"]["n_splits"] == config.n_splits


def test_release_gate_configuration_is_research_scoped() -> None:
    """Keep the final release gate explicitly within HERMES 2.0 research scope."""
    config = _release_config()

    assert config.n_repeats >= 2
    assert config.n_splits >= 2
    assert config.run_applicability is True

    # Expensive inferential/sensitivity modules have their own dedicated tests.
    # They are disabled here so this gate remains a fast integration test.
    assert config.run_modifier_discovery is False
    assert config.run_robustness is False
    assert config.run_permutation is False