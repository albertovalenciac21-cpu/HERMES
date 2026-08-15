"""
Tests for HERMES 2.0 multi-seed treatment-effect simulation study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.simulation_study import (
    RECOVERY_METRICS,
    generate_simulation_random_states,
    run_simulation_grid,
    run_simulation_study,
    run_single_simulation,
    summarize_interaction_strengths,
)


def test_random_state_generation() -> None:
    states = generate_simulation_random_states(
        n_simulations=10,
        base_random_state=2026,
    )

    assert len(states) == 10
    assert len(set(states)) == 10
    assert all(
        isinstance(state, int)
        for state in states
    )

    print(
        "PASS: simulation random-state generation"
    )


def test_random_state_reproducibility() -> None:
    first = generate_simulation_random_states(
        n_simulations=5,
        base_random_state=2026,
    )

    second = generate_simulation_random_states(
        n_simulations=5,
        base_random_state=2026,
    )

    assert first == second

    print(
        "PASS: simulation random-state reproducibility"
    )


def test_single_simulation_structure() -> None:
    record = run_single_simulation(
        interaction_strength=1.5,
        data_random_state=2026,
        n_patients=200,
        n_features=10,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        model_random_state=42,
    )

    assert record[
        "interaction_strength"
    ] == 1.5

    assert record[
        "data_random_state"
    ] == 2026

    for metric in RECOVERY_METRICS:
        assert metric in record

    print(
        "PASS: single-simulation structure"
    )


def test_simulation_grid_dimensions() -> None:
    table = run_simulation_grid(
        interaction_strengths=(
            0.0,
            1.0,
        ),
        random_states=(
            101,
            202,
            303,
        ),
        n_patients=150,
        n_features=10,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        model_random_state=42,
        verbose=False,
    )

    assert isinstance(
        table,
        pd.DataFrame,
    )

    assert len(table) == 6

    assert set(
        table[
            "interaction_strength"
        ]
    ) == {
        0.0,
        1.0,
    }

    assert table[
        "data_random_state"
    ].nunique() == 3

    print(
        "PASS: simulation-grid dimensions"
    )


def test_simulation_grid_reproducibility() -> None:
    kwargs = dict(
        interaction_strengths=(
            0.0,
            1.0,
        ),
        random_states=(
            101,
            202,
        ),
        n_patients=150,
        n_features=10,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        model_random_state=42,
        verbose=False,
    )

    first = run_simulation_grid(
        **kwargs
    )

    second = run_simulation_grid(
        **kwargs
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

    print(
        "PASS: simulation-grid reproducibility"
    )


def test_summary_structure() -> None:
    table = run_simulation_grid(
        interaction_strengths=(
            0.0,
            1.5,
        ),
        random_states=(
            101,
            202,
            303,
        ),
        n_patients=150,
        n_features=10,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        model_random_state=42,
        verbose=False,
    )

    summary = summarize_interaction_strengths(
        table
    )

    assert isinstance(
        summary,
        pd.DataFrame,
    )

    assert list(
        summary.index
    ) == [
        0.0,
        1.5,
    ]

    assert (
        summary[
            "n_simulations"
        ]
        == 3
    ).all()

    assert (
        "ite_pearson_correlation_mean"
        in summary.columns
    )

    assert (
        "ite_pearson_correlation_mcse"
        in summary.columns
    )

    assert (
        "fraction_positive_pearson"
        in summary.columns
    )

    print(
        "PASS: interaction-summary structure"
    )


def test_probability_metrics_are_bounded() -> None:
    result = run_simulation_study(
        interaction_strengths=(
            0.0,
            1.5,
        ),
        n_simulations=3,
        n_patients=150,
        n_features=10,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        simulation_base_random_state=2026,
        model_random_state=42,
        verbose=False,
    )

    probability_columns = [
        "fraction_positive_pearson",
        "fraction_pearson_ge_0_50",
        "fraction_positive_spearman",
        "fraction_sign_accuracy_ge_0_75",
        "fraction_top_overlap_ge_0_50",
        "fraction_positive_true_separation",
    ]

    for column in probability_columns:

        values = (
            result
            .interaction_summary[
                column
            ]
        )

        assert (
            (
                values >= 0.0
            )
            & (
                values <= 1.0
            )
        ).all()

    print(
        "PASS: probability metrics bounded"
    )


def test_positive_control_improves_recovery() -> None:
    """
    Strong injected HTE should produce substantially better average
    recovery than the zero-interaction condition.

    This is intentionally tested across multiple independent synthetic
    trials rather than relying on one favorable random seed.
    """

    result = run_simulation_study(
        interaction_strengths=(
            0.0,
            1.5,
        ),
        n_simulations=5,
        n_patients=300,
        n_features=15,
        n_repeats=3,
        n_splits=5,
        C=0.1,
        simulation_base_random_state=2026,
        model_random_state=42,
        verbose=False,
    )

    summary = (
        result.interaction_summary
    )

    null_pearson = float(
        summary.loc[
            0.0,
            "ite_pearson_correlation_mean",
        ]
    )

    signal_pearson = float(
        summary.loc[
            1.5,
            "ite_pearson_correlation_mean",
        ]
    )

    null_separation = float(
        summary.loc[
            0.0,
            "true_ite_top_bottom_separation_mean",
        ]
    )

    signal_separation = float(
        summary.loc[
            1.5,
            "true_ite_top_bottom_separation_mean",
        ]
    )

    assert signal_pearson > null_pearson
    assert signal_pearson > 0.40

    assert (
        signal_separation
        > null_separation
    )

    assert signal_separation > 0.10

    print(
        "PASS: injected HTE improves HERMES recovery"
    )


def test_strong_signal_recovery_direction() -> None:
    """
    With a strong treatment interaction, HERMES should recover positive
    ranking information on average.
    """

    result = run_simulation_study(
        interaction_strengths=(
            1.5,
        ),
        n_simulations=5,
        n_patients=300,
        n_features=15,
        n_repeats=3,
        n_splits=5,
        C=0.1,
        simulation_base_random_state=2026,
        model_random_state=42,
        verbose=False,
    )

    summary = (
        result.interaction_summary
    )

    pearson = float(
        summary.loc[
            1.5,
            "ite_pearson_correlation_mean",
        ]
    )

    spearman = float(
        summary.loc[
            1.5,
            "ite_spearman_correlation_mean",
        ]
    )

    overlap = float(
        summary.loc[
            1.5,
            "top_quartile_overlap_mean",
        ]
    )

    assert pearson > 0.40
    assert spearman > 0.40
    assert overlap > 0.40

    print(
        "PASS: strong-signal treatment-effect recovery"
    )


def test_complete_study_reproducibility() -> None:
    kwargs = dict(
        interaction_strengths=(
            0.0,
            1.0,
        ),
        n_simulations=3,
        n_patients=150,
        n_features=10,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        simulation_base_random_state=2026,
        model_random_state=42,
        verbose=False,
    )

    first = run_simulation_study(
        **kwargs
    )

    second = run_simulation_study(
        **kwargs
    )

    pd.testing.assert_frame_equal(
        first.simulation_table,
        second.simulation_table,
    )

    pd.testing.assert_frame_equal(
        first.interaction_summary,
        second.interaction_summary,
    )

    assert (
        first.overall_summary
        == second.overall_summary
    )

    print(
        "PASS: complete simulation-study reproducibility"
    )


def main() -> None:
    print(
        "=== HERMES 2.0 MULTI-SEED "
        "SIMULATION STUDY TESTS ==="
    )

    test_random_state_generation()
    test_random_state_reproducibility()
    test_single_simulation_structure()
    test_simulation_grid_dimensions()
    test_simulation_grid_reproducibility()
    test_summary_structure()
    test_probability_metrics_are_bounded()
    test_positive_control_improves_recovery()
    test_strong_signal_recovery_direction()
    test_complete_study_reproducibility()

    print()

    print(
        "=============================================="
    )

    print(
        "ALL MULTI-SEED SIMULATION STUDY TESTS PASSED"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":
    main()