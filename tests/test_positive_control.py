"""
HERMES 2.0
Positive-Control Simulation Tests
=================================

Validation suite for known-ground-truth treatment-effect recovery.

These tests verify:

1. Synthetic trial dimensions
2. Randomized treatment encoding
3. Binary outcome encoding
4. Valid potential-outcome probabilities
5. Correct true ITE arithmetic
6. Deterministic data generation
7. Known interaction produces real HTE
8. Zero interaction removes biology-dependent HTE
9. Patient-table integrity
10. HERMES positive-control recovery
11. Ranking recovery
12. Sign recovery
13. Estimated heterogeneity is non-degenerate
14. Summary consistency
15. Deterministic HERMES recovery

Passing these tests establishes implementation integrity and
known-signal recovery under controlled simulation.

It does NOT establish heterogeneous treatment effects in NeoTRIP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.positive_control import (
    build_positive_control_patient_table,
    calculate_positive_control_metrics,
    generate_positive_control_dataset,
    pearson_correlation,
    run_positive_control,
    sigmoid,
    spearman_correlation,
)


def assert_close(
    a: float,
    b: float,
    *,
    atol: float = 1e-12,
) -> None:
    assert np.isclose(
        a,
        b,
        atol=atol,
        rtol=0.0,
    ), f"{a} != {b}"


def run_tests() -> None:

    print(
        "=== HERMES 2.0 "
        "Positive-Control Simulation Tests ==="
    )

    # =========================================================
    # 1. Sigmoid validity
    # =========================================================

    values = np.array(
        [
            -100.0,
            -1.0,
            0.0,
            1.0,
            100.0,
        ]
    )

    probabilities = sigmoid(
        values
    )

    assert (
        probabilities >= 0
    ).all()

    assert (
        probabilities <= 1
    ).all()

    assert_close(
        float(
            probabilities[2]
        ),
        0.5,
    )

    print(
        "PASS: sigmoid validity"
    )

    # =========================================================
    # 2. Generate known positive-control dataset
    # =========================================================

    dataset = (
        generate_positive_control_dataset(
            n_patients=500,
            n_features=20,
            treatment_interaction=1.50,
            random_state=2026,
        )
    )

    assert dataset.X.shape == (
        500,
        20,
    )

    assert len(
        dataset.T
    ) == 500

    assert len(
        dataset.Y
    ) == 500

    assert len(
        dataset.true_ite
    ) == 500

    assert (
        dataset.signal_feature
        == "FEATURE_001"
    )

    print(
        "PASS: synthetic trial dimensions"
    )

    # =========================================================
    # 3. Index alignment
    # =========================================================

    assert (
        dataset.X.index.equals(
            dataset.T.index
        )
    )

    assert (
        dataset.X.index.equals(
            dataset.Y.index
        )
    )

    assert (
        dataset.X.index.equals(
            dataset.true_ite.index
        )
    )

    assert (
        dataset.X.index.equals(
            dataset.probability_control.index
        )
    )

    assert (
        dataset.X.index.equals(
            dataset.probability_treated.index
        )
    )

    print(
        "PASS: patient alignment"
    )

    # =========================================================
    # 4. Randomized treatment encoding
    # =========================================================

    assert set(
        dataset.T.unique()
    ).issubset(
        {
            0,
            1,
        }
    )

    treatment_fraction = float(
        dataset.T.mean()
    )

    assert (
        0.35
        < treatment_fraction
        < 0.65
    )

    print(
        "PASS: randomized treatment encoding"
    )

    # =========================================================
    # 5. Binary outcome encoding
    # =========================================================

    assert set(
        dataset.Y.unique()
    ).issubset(
        {
            0,
            1,
        }
    )

    print(
        "PASS: binary outcome encoding"
    )

    # =========================================================
    # 6. Potential-outcome probability bounds
    # =========================================================

    for probability in [
        dataset.probability_control,
        dataset.probability_treated,
    ]:

        assert (
            probability >= 0.0
        ).all()

        assert (
            probability <= 1.0
        ).all()

        assert not (
            probability.isna().any()
        )

    print(
        "PASS: potential-outcome probability bounds"
    )

    # =========================================================
    # 7. True ITE arithmetic
    # =========================================================

    expected_true_ite = (
        dataset.probability_treated
        - dataset.probability_control
    )

    np.testing.assert_allclose(
        dataset.true_ite.to_numpy(),
        expected_true_ite.to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    assert (
        dataset.true_ite >= -1.0
    ).all()

    assert (
        dataset.true_ite <= 1.0
    ).all()

    print(
        "PASS: true ITE arithmetic"
    )

    # =========================================================
    # 8. Known interaction creates heterogeneity
    # =========================================================

    assert (
        dataset.true_ite.std()
        > 0.10
    )

    signal_true_ite_correlation = (
        pearson_correlation(
            dataset.X[
                dataset.signal_feature
            ],
            dataset.true_ite,
        )
    )

    assert (
        signal_true_ite_correlation
        > 0.70
    )

    print(
        "PASS: injected interaction creates HTE"
    )

    # =========================================================
    # 9. Deterministic simulation
    # =========================================================

    dataset_repeat = (
        generate_positive_control_dataset(
            n_patients=500,
            n_features=20,
            treatment_interaction=1.50,
            random_state=2026,
        )
    )

    pd.testing.assert_frame_equal(
        dataset.X,
        dataset_repeat.X,
    )

    pd.testing.assert_series_equal(
        dataset.T,
        dataset_repeat.T,
    )

    pd.testing.assert_series_equal(
        dataset.Y,
        dataset_repeat.Y,
    )

    pd.testing.assert_series_equal(
        dataset.true_ite,
        dataset_repeat.true_ite,
    )

    print(
        "PASS: deterministic synthetic trial"
    )

    # =========================================================
    # 10. Zero treatment interaction removes predictive HTE
    #
    # With beta_TX = 0, treatment may still have an average
    # treatment effect, but FEATURE_001 should no longer modify
    # that treatment effect.
    # =========================================================

    null_interaction_dataset = (
        generate_positive_control_dataset(
            n_patients=500,
            n_features=20,
            treatment_interaction=0.0,
            random_state=2026,
        )
    )

    # Logistic non-collapsibility / probability-scale effects can
    # still produce some variation in risk differences even when
    # the log-odds interaction is zero, so we do NOT demand exact
    # zero variance.
    #
    # Instead, heterogeneity should be substantially smaller than
    # under the strong injected interaction.
    assert (
        null_interaction_dataset
        .true_ite
        .std()
        <
        dataset
        .true_ite
        .std()
    )

    print(
        "PASS: zero interaction reduces HTE"
    )

    # =========================================================
    # 11. Correlation helper functions
    # =========================================================

    perfect = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )

    assert_close(
        pearson_correlation(
            perfect,
            perfect,
        ),
        1.0,
    )

    assert_close(
        spearman_correlation(
            perfect,
            perfect,
        ),
        1.0,
    )

    print(
        "PASS: correlation helpers"
    )

    # =========================================================
    # 12. Run HERMES positive control
    #
    # Five repeated cross-fits are sufficient for this validation
    # suite and keep runtime reasonable.
    # =========================================================

    result = run_positive_control(
        n_patients=500,
        n_features=20,
        treatment_interaction=1.50,
        n_repeats=5,
        n_splits=5,
        C=0.1,
        data_random_state=2026,
        model_random_state=42,
    )

    print(
        "PASS: HERMES positive-control experiment"
    )

    # =========================================================
    # 13. Patient table
    # =========================================================

    patient_table = (
        result.patient_table
    )

    assert len(
        patient_table
    ) == 500

    assert patient_table.index.equals(
        result.dataset.X.index
    )

    required_columns = {
        "signal_feature_value",
        "T",
        "Y",
        "true_probability_control",
        "true_probability_treated",
        "true_ite",
        "estimated_ite",
        "estimated_ite_std",
        "estimated_sign_stability",
        "true_benefit",
        "estimated_benefit",
    }

    assert set(
        patient_table.columns
    ) == required_columns

    assert not (
        patient_table.isna().any().any()
    )

    print(
        "PASS: positive-control patient table"
    )

    # =========================================================
    # 14. Strong true-vs-estimated ITE correlation
    #
    # These thresholds are intentionally below the ~0.83 recovery
    # observed in the development experiment. They verify recovery
    # without overfitting the test to one exact numerical value.
    # =========================================================

    assert (
        result.metrics[
            "ite_pearson_correlation"
        ]
        > 0.70
    )

    assert (
        result.metrics[
            "ite_spearman_correlation"
        ]
        > 0.70
    )

    print(
        "PASS: individualized treatment-effect recovery"
    )

    # =========================================================
    # 15. Injected feature recovery
    # =========================================================

    assert (
        result.metrics[
            "signal_feature_estimated_ite_correlation"
        ]
        > 0.60
    )

    print(
        "PASS: injected modifier recovery"
    )

    # =========================================================
    # 16. Direction recovery
    # =========================================================

    assert (
        result.metrics[
            "ite_sign_accuracy"
        ]
        > 0.75
    )

    print(
        "PASS: treatment-effect direction recovery"
    )

    # =========================================================
    # 17. Patient ranking recovery
    # =========================================================

    assert (
        result.metrics[
            "top_quartile_overlap"
        ]
        > 0.50
    )

    assert (
        result.metrics[
            "true_ite_top_bottom_separation"
        ]
        > 0.20
    )

    print(
        "PASS: high-benefit patient ranking"
    )

    # =========================================================
    # 18. Estimated heterogeneity
    # =========================================================

    assert (
        result.metrics[
            "estimated_ite_sd"
        ]
        > 0.05
    )

    print(
        "PASS: non-degenerate estimated HTE"
    )

    # =========================================================
    # 19. Estimated dispersion reasonably tracks truth
    # =========================================================

    true_sd = float(
        result.metrics[
            "true_ite_sd"
        ]
    )

    estimated_sd = float(
        result.metrics[
            "estimated_ite_sd"
        ]
    )

    sd_ratio = (
        estimated_sd
        / true_sd
    )

    assert (
        0.50
        < sd_ratio
        < 1.50
    )

    print(
        "PASS: treatment-effect dispersion recovery"
    )

    # =========================================================
    # 20. Summary consistency
    # =========================================================

    assert (
        result.summary[
            "patients"
        ]
        == 500
    )

    assert (
        result.summary[
            "features"
        ]
        == 20
    )

    assert (
        result.summary[
            "signal_feature"
        ]
        == "FEATURE_001"
    )

    assert_close(
        result.summary[
            "injected_treatment_interaction"
        ],
        1.50,
    )

    assert_close(
        result.summary[
            "ite_pearson_correlation"
        ],
        result.metrics[
            "ite_pearson_correlation"
        ],
    )

    print(
        "PASS: summary consistency"
    )

    # =========================================================
    # 21. Recalculate metrics directly
    # =========================================================

    recalculated = (
        calculate_positive_control_metrics(
            patient_table
        )
    )

    for key in (
        result.metrics
    ):

        assert_close(
            result.metrics[
                key
            ],
            recalculated[
                key
            ],
        )

    print(
        "PASS: metric reproducibility"
    )

    # =========================================================
    # 22. Patient-table builder reproducibility
    # =========================================================

    rebuilt_table = (
        build_positive_control_patient_table(
            result.dataset,
            result.hermes_result,
        )
    )

    pd.testing.assert_frame_equal(
        patient_table,
        rebuilt_table,
    )

    print(
        "PASS: patient-table reproducibility"
    )

    # =========================================================
    # 23. Full deterministic positive-control experiment
    # =========================================================

    result_repeat = run_positive_control(
        n_patients=500,
        n_features=20,
        treatment_interaction=1.50,
        n_repeats=5,
        n_splits=5,
        C=0.1,
        data_random_state=2026,
        model_random_state=42,
    )

    pd.testing.assert_frame_equal(
        result.patient_table,
        result_repeat.patient_table,
    )

    for key in (
        result.metrics
    ):

        assert_close(
            result.metrics[
                key
            ],
            result_repeat.metrics[
                key
            ],
        )

    print(
        "PASS: deterministic positive-control recovery"
    )

    # =========================================================
    # Final
    # =========================================================

    print()

    print(
        "=============================================="
    )

    print(
        "ALL POSITIVE-CONTROL SIMULATION TESTS PASSED"
    )

    print(
        "=============================================="
    )

    print()

    print(
        f"Patients: "
        f"{result.summary['patients']}"
    )

    print(
        "Injected interaction: "
        f"{result.summary['injected_treatment_interaction']:.2f}"
    )

    print(
        "ITE Pearson correlation: "
        f"{result.metrics['ite_pearson_correlation']:.4f}"
    )

    print(
        "ITE Spearman correlation: "
        f"{result.metrics['ite_spearman_correlation']:.4f}"
    )

    print(
        "Sign accuracy: "
        f"{result.metrics['ite_sign_accuracy']:.4f}"
    )

    print(
        "Top-quartile overlap: "
        f"{result.metrics['top_quartile_overlap']:.4f}"
    )

    print()

    print(
        "NOTE: Positive-control success demonstrates known-signal "
        "recovery capability, not biological validation in NeoTRIP."
    )


if __name__ == "__main__":
    run_tests()