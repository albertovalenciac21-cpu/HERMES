"""
HERMES 2.0
Repeated Cross-Fitted Treatment-Effect Tests
============================================

Direct automated coverage for the repeated cross-fitting layer.

These tests verify:
- memory-safe deterministic seed generation
- seed validation
- patient/repeat matrix integrity
- counterfactual ITE arithmetic
- summary consistency
- deterministic repeated cross-fitting
- stable-patient table behavior

The integration tests use a deterministic synthetic randomized-trial-like
matrix so they do not depend on local NeoTRIP data files.

Passing these tests establishes implementation integrity and resampling
reproducibility. It does not establish causal validity, clinical utility,
or external generalizability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.treatment_effects.repeated_crossfit import (
    generate_random_states,
    repeated_crossfit_treatment_effect_model,
    stable_patient_table,
    validate_random_states,
)


def _synthetic_trial(
    *,
    n_per_joint_stratum: int = 30,
    n_features: int = 8,
    random_state: int = 2026,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create a balanced deterministic dataset for repeated-crossfit tests."""

    rng = np.random.default_rng(
        random_state
    )

    treatment_values: list[int] = []
    outcome_values: list[int] = []

    for treatment in (0, 1):
        for outcome in (0, 1):
            treatment_values.extend(
                [treatment] * n_per_joint_stratum
            )
            outcome_values.extend(
                [outcome] * n_per_joint_stratum
            )

    n_patients = len(
        treatment_values
    )

    X = pd.DataFrame(
        rng.normal(
            size=(
                n_patients,
                n_features,
            )
        ),
        index=pd.Index(
            [
                f"PATIENT_{i:04d}"
                for i in range(n_patients)
            ],
            name="Patient_ID",
        ),
        columns=[
            f"PATHWAY_{i:02d}"
            for i in range(n_features)
        ],
    )

    X.loc[:, "PATHWAY_00"] += (
        0.35
        * np.asarray(outcome_values)
    )

    X.loc[:, "PATHWAY_01"] += (
        0.20
        * np.asarray(treatment_values)
    )

    treatment = pd.Series(
        treatment_values,
        index=X.index,
        name="T",
        dtype=int,
    )

    outcome = pd.Series(
        outcome_values,
        index=X.index,
        name="Y",
        dtype=int,
    )

    return X, treatment, outcome


def _run_small_repeated_crossfit():
    X, treatment, outcome = _synthetic_trial()

    result = repeated_crossfit_treatment_effect_model(
        X=X,
        treatment=treatment,
        outcome=outcome,
        n_repeats=3,
        n_splits=5,
        C=0.1,
        base_random_state=42,
    )

    return X, treatment, outcome, result


def test_generate_random_states_is_unique_and_deterministic() -> None:
    first = generate_random_states(
        10,
        base_random_state=42,
    )

    second = generate_random_states(
        10,
        base_random_state=42,
    )

    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10
    assert all(
        isinstance(value, int)
        for value in first
    )
    assert all(
        1 <= value < 2_147_483_647
        for value in first
    )


def test_random_state_validation() -> None:
    assert validate_random_states(
        [11, 22, 33]
    ) == (
        11,
        22,
        33,
    )

    with pytest.raises(
        ValueError,
        match="At least two random states",
    ):
        validate_random_states(
            [11]
        )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        validate_random_states(
            [11, 11]
        )

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        validate_random_states(
            [11, -1]
        )

    with pytest.raises(
        ValueError,
        match="n_repeats must be at least 2",
    ):
        generate_random_states(
            1,
            base_random_state=42,
        )

    with pytest.raises(
        ValueError,
        match="base_random_state must be non-negative",
    ):
        generate_random_states(
            3,
            base_random_state=-1,
        )


def test_repeated_crossfit_matrix_and_summary_integrity() -> None:
    X, _, _, result = _run_small_repeated_crossfit()

    expected_shape = (
        len(X),
        3,
    )

    matrices = [
        result.ite_by_repeat,
        result.observed_probability_by_repeat,
        result.probability_control_by_repeat,
        result.probability_treated_by_repeat,
    ]

    for matrix in matrices:
        assert matrix.shape == expected_shape
        assert matrix.index.equals(
            X.index
        )
        assert not matrix.isna().any().any()
        assert np.isfinite(
            matrix.to_numpy(dtype=float)
        ).all()

    for probability_matrix in [
        result.observed_probability_by_repeat,
        result.probability_control_by_repeat,
        result.probability_treated_by_repeat,
    ]:
        assert (
            probability_matrix >= 0.0
        ).all().all()

        assert (
            probability_matrix <= 1.0
        ).all().all()

    expected_ite = (
        result.probability_treated_by_repeat
        - result.probability_control_by_repeat
    )

    pd.testing.assert_frame_equal(
        result.ite_by_repeat,
        expected_ite,
        check_exact=False,
        atol=1e-12,
        rtol=0.0,
    )

    assert result.n_patients == len(X)
    assert result.n_repeats == 3
    assert len(result.random_states) == 3
    assert len(set(result.random_states)) == 3

    assert result.patient_summary.index.equals(
        X.index
    )

    assert result.repeat_summary.shape[0] == 3

    np.testing.assert_allclose(
        result.patient_summary[
            "mean_ite"
        ].to_numpy(),
        result.ite_by_repeat.mean(
            axis=1
        ).to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    assert result.summary[
        "patients"
    ] == len(X)

    assert result.summary[
        "biological_features"
    ] == X.shape[1]

    assert result.summary[
        "n_repeats"
    ] == 3

    assert result.summary[
        "n_splits"
    ] == 5

    assert np.isclose(
        result.summary[
            "cohort_mean_ite"
        ],
        result.patient_summary[
            "mean_ite"
        ].mean(),
        atol=1e-12,
        rtol=0.0,
    )


def test_explicit_random_states_are_preserved() -> None:
    X, treatment, outcome = _synthetic_trial()

    explicit_states = (
        101,
        202,
        303,
    )

    result = repeated_crossfit_treatment_effect_model(
        X=X,
        treatment=treatment,
        outcome=outcome,
        n_repeats=99,
        n_splits=5,
        C=0.1,
        random_states=explicit_states,
    )

    assert result.random_states == explicit_states

    assert result.n_repeats == len(
        explicit_states
    )

    assert result.summary[
        "n_repeats"
    ] == len(explicit_states)


def test_repeated_crossfit_is_deterministic() -> None:
    X, treatment, outcome = _synthetic_trial()

    first = repeated_crossfit_treatment_effect_model(
        X=X,
        treatment=treatment,
        outcome=outcome,
        n_repeats=3,
        n_splits=5,
        C=0.1,
        base_random_state=2026,
    )

    second = repeated_crossfit_treatment_effect_model(
        X=X,
        treatment=treatment,
        outcome=outcome,
        n_repeats=3,
        n_splits=5,
        C=0.1,
        base_random_state=2026,
    )

    assert first.random_states == second.random_states

    pd.testing.assert_frame_equal(
        first.ite_by_repeat,
        second.ite_by_repeat,
        check_exact=True,
    )

    pd.testing.assert_frame_equal(
        first.patient_summary,
        second.patient_summary,
        check_exact=True,
    )

    pd.testing.assert_frame_equal(
        first.repeat_summary,
        second.repeat_summary,
        check_exact=True,
    )

    assert first.summary == second.summary


def test_stable_patient_table_sorting_and_validation() -> None:
    _, _, _, result = _run_small_repeated_crossfit()

    descending = stable_patient_table(
        result,
        sort_by="mean_ite",
        ascending=False,
    )

    ascending = stable_patient_table(
        result,
        sort_by="ite_std",
        ascending=True,
    )

    assert descending[
        "mean_ite"
    ].is_monotonic_decreasing

    assert ascending[
        "ite_std"
    ].is_monotonic_increasing

    assert set(descending.index) == set(
        result.patient_summary.index
    )

    with pytest.raises(
        ValueError,
        match="Unknown sort column",
    ):
        stable_patient_table(
            result,
            sort_by="not_a_real_column",
        )