"""
HERMES 2.0
Treatment-Effect Uncertainty Tests
==================================

Validation tests for patient-level uncertainty estimation from repeated
cross-fitted individualized treatment-effect estimates.
"""

import numpy as np
import pandas as pd

from backend.app.treatment_effects.positive_control import (
    run_positive_control,
)

from backend.app.treatment_effects.uncertainty import (
    _extract_ite_matrix,
    build_uncertainty_table,
    quantify_treatment_effect_uncertainty,
    summarize_uncertainty_table,
    validate_uncertainty_against_truth,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)

    print(f"PASS: {message}")


def main():

    print(
        "=== HERMES 2.0 TREATMENT-EFFECT UNCERTAINTY TESTS ==="
    )
    print()

    # ---------------------------------------------------------
    # Small deterministic matrix tests
    # ---------------------------------------------------------

    ite_matrix = pd.DataFrame(
        {
            "repeat_1": [0.10, -0.20, 0.10],
            "repeat_2": [0.20, -0.10, -0.10],
            "repeat_3": [0.30, -0.30, 0.10],
            "repeat_4": [0.20, -0.20, -0.10],
        },
        index=[
            "patient_1",
            "patient_2",
            "patient_3",
        ],
    )

    table = build_uncertainty_table(
        ite_matrix,
        alpha=0.05,
        minimum_sign_stability=0.90,
        minimum_signal_uncertainty_ratio=1.0,
    )

    check(
        table.shape[0] == 3,
        "complete patient coverage",
    )

    check(
        table.index.equals(ite_matrix.index),
        "patient order preservation",
    )

    required_columns = {
        "mean_ite",
        "median_ite",
        "ite_std",
        "ite_lower",
        "ite_upper",
        "interval_width",
        "fraction_positive",
        "fraction_negative",
        "fraction_zero",
        "sign_stability",
        "signal_uncertainty_ratio",
        "interval_excludes_zero",
        "evidence_state",
    }

    check(
        required_columns.issubset(table.columns),
        "uncertainty-table schema",
    )

    check(
        np.allclose(
            table["mean_ite"].to_numpy(),
            ite_matrix.mean(axis=1).to_numpy(),
        ),
        "mean ITE arithmetic",
    )

    check(
        np.allclose(
            table["median_ite"].to_numpy(),
            ite_matrix.median(axis=1).to_numpy(),
        ),
        "median ITE arithmetic",
    )

    check(
        np.allclose(
            table["ite_std"].to_numpy(),
            ite_matrix.std(axis=1, ddof=1).to_numpy(),
        ),
        "ITE standard-deviation arithmetic",
    )

    check(
        (
            table["ite_upper"]
            >= table["ite_lower"]
        ).all(),
        "empirical interval ordering",
    )

    check(
        np.allclose(
            table["interval_width"].to_numpy(),
            (
                table["ite_upper"]
                - table["ite_lower"]
            ).to_numpy(),
        ),
        "interval-width arithmetic",
    )

    check(
        (
            (
                table["fraction_positive"]
                >= 0.0
            )
            &
            (
                table["fraction_positive"]
                <= 1.0
            )
        ).all(),
        "positive-fraction bounds",
    )

    check(
        (
            (
                table["fraction_negative"]
                >= 0.0
            )
            &
            (
                table["fraction_negative"]
                <= 1.0
            )
        ).all(),
        "negative-fraction bounds",
    )

    check(
        (
            (
                table["sign_stability"]
                >= 0.0
            )
            &
            (
                table["sign_stability"]
                <= 1.0
            )
        ).all(),
        "sign-stability bounds",
    )

    check(
        table.loc[
            "patient_1",
            "fraction_positive",
        ] == 1.0,
        "unanimous positive sign detection",
    )

    check(
        table.loc[
            "patient_2",
            "fraction_negative",
        ] == 1.0,
        "unanimous negative sign detection",
    )

    check(
        table.loc[
            "patient_3",
            "sign_stability",
        ] == 0.5,
        "unstable sign detection",
    )

    check(
        table.loc[
            "patient_1",
            "evidence_state",
        ] == "likely_benefit",
        "positive evidence-state classification",
    )

    check(
        table.loc[
            "patient_2",
            "evidence_state",
        ] == "likely_harm",
        "negative evidence-state classification",
    )

    check(
        table.loc[
            "patient_3",
            "evidence_state",
        ] == "indeterminate",
        "indeterminate evidence-state classification",
    )

    # ---------------------------------------------------------
    # Summary tests
    # ---------------------------------------------------------

    summary = summarize_uncertainty_table(
        table
    )

    check(
        summary["patients"] == 3,
        "summary patient count",
    )

    check(
        (
            summary["likely_benefit"]
            + summary["likely_harm"]
            + summary["indeterminate"]
        )
        == 3,
        "evidence-state count conservation",
    )

    check(
        np.isclose(
            summary["fraction_likely_benefit"]
            + summary["fraction_likely_harm"]
            + summary["fraction_indeterminate"],
            1.0,
        ),
        "evidence-state fraction conservation",
    )

    # ---------------------------------------------------------
    # Positive-control integration
    # ---------------------------------------------------------

    positive_control = run_positive_control(
        n_patients=250,
        n_features=20,
        treatment_interaction=1.5,
        n_repeats=10,
        n_splits=5,
        C=0.1,
        data_random_state=2026,
        model_random_state=42,
    )

    repeated_result = (
        positive_control.hermes_result
    )

    extracted = _extract_ite_matrix(
        repeated_result
    )

    check(
        isinstance(
            extracted,
            pd.DataFrame,
        ),
        "RepeatedCrossFitResult ITE extraction",
    )

    check(
        extracted.shape == (250, 10),
        "patient-by-repeat matrix dimensions",
    )

    uncertainty = (
        quantify_treatment_effect_uncertainty(
            repeated_result,
            alpha=0.05,
            minimum_sign_stability=0.90,
            minimum_signal_uncertainty_ratio=1.0,
        )
    )

    check(
        uncertainty.patient_table.shape[0]
        == 250,
        "positive-control patient coverage",
    )

    check(
        uncertainty.summary[
            "n_repeated_estimates"
        ]
        == 10,
        "repeat-count preservation",
    )

    check(
        uncertainty.patient_table[
            "mean_ite"
        ].std()
        > 0.0,
        "non-degenerate individualized treatment effects",
    )

    check(
        uncertainty.patient_table[
            "ite_std"
        ].mean()
        > 0.0,
        "non-degenerate patient uncertainty",
    )

    check(
        (
            uncertainty.patient_table[
                "interval_width"
            ]
            >= 0.0
        ).all(),
        "non-negative uncertainty intervals",
    )

    valid_states = {
        "likely_benefit",
        "likely_harm",
        "indeterminate",
    }

    check(
        set(
            uncertainty.patient_table[
                "evidence_state"
            ].unique()
        ).issubset(
            valid_states
        ),
        "valid evidence states",
    )

    # ---------------------------------------------------------
    # Synthetic truth validation
    # ---------------------------------------------------------

    validation = (
        validate_uncertainty_against_truth(
            uncertainty.patient_table,
            positive_control.dataset.true_ite,
            stable_threshold=0.90,
        )
    )

    check(
        0.0
        <= validation[
            "empirical_interval_coverage"
        ]
        <= 1.0,
        "empirical interval coverage bounds",
    )

    check(
        validation[
            "mean_absolute_ite_error"
        ]
        >= 0.0,
        "mean absolute ITE error bounds",
    )

    check(
        validation[
            "root_mean_squared_ite_error"
        ]
        >= 0.0,
        "RMSE bounds",
    )

    check(
        0.0
        <= validation[
            "overall_sign_accuracy"
        ]
        <= 1.0,
        "overall sign-accuracy bounds",
    )

    check(
        0.0
        <= validation[
            "fraction_stable_patients"
        ]
        <= 1.0,
        "stable-patient fraction bounds",
    )

    # ---------------------------------------------------------
    # Determinism
    # ---------------------------------------------------------

    positive_control_2 = run_positive_control(
        n_patients=250,
        n_features=20,
        treatment_interaction=1.5,
        n_repeats=10,
        n_splits=5,
        C=0.1,
        data_random_state=2026,
        model_random_state=42,
    )

    uncertainty_2 = (
        quantify_treatment_effect_uncertainty(
            positive_control_2.hermes_result,
            alpha=0.05,
            minimum_sign_stability=0.90,
            minimum_signal_uncertainty_ratio=1.0,
        )
    )

    check(
        uncertainty.patient_table.equals(
            uncertainty_2.patient_table
        ),
        "deterministic uncertainty estimation",
    )

    print()
    print(
        "============================================="
    )
    print(
        "ALL TREATMENT-EFFECT UNCERTAINTY TESTS PASSED"
    )
    print(
        "============================================="
    )

    print()

    print(
        f"Patients: "
        f"{uncertainty.summary['patients']}"
    )

    print(
        f"Repeats: "
        f"{uncertainty.summary['n_repeated_estimates']}"
    )

    print(
        "Mean patient ITE SD: "
        f"{uncertainty.summary['mean_ite_std']:.4f}"
    )

    print(
        "Fraction stable patients: "
        f"{validation['fraction_stable_patients']:.4f}"
    )

    print(
        "Overall sign accuracy: "
        f"{validation['overall_sign_accuracy']:.4f}"
    )

    print(
        "Stable-patient sign accuracy: "
        f"{validation['stable_patient_sign_accuracy']:.4f}"
    )

    print(
        "Unstable-patient sign accuracy: "
        f"{validation['unstable_patient_sign_accuracy']:.4f}"
    )

    print(
        "Stable-patient MAE: "
        f"{validation['stable_patient_mean_absolute_error']:.4f}"
    )

    print(
        "Unstable-patient MAE: "
        f"{validation['unstable_patient_mean_absolute_error']:.4f}"
    )

    print(
        "Empirical interval coverage: "
        f"{validation['empirical_interval_coverage']:.4f}"
    )

    print()
    print(
        "NOTE: Repeated-fit empirical intervals measure "
        "model/resampling stability and are not formal causal "
        "confidence intervals."
    )


if __name__ == "__main__":
    main()