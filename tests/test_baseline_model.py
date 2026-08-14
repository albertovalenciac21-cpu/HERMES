"""
HERMES 2.0
Baseline Treatment-Effect Model Tests
=====================================

Validates the first regularized treatment x biology interaction model.

These tests check:

1. Input integrity
2. Expected dimensions
3. Treatment/outcome preservation
4. Design-matrix construction
5. Counterfactual probability validity
6. Individualized treatment-effect arithmetic
7. Coefficient structure
8. Deterministic fitting
9. Treatment-effect table construction
10. Interaction table construction

This is an engineering validation suite.
It does NOT establish clinical validity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.feature_builder import (
    build_treatment_effect_dataset,
)

from backend.app.treatment_effects.baseline_model import (
    build_interaction_design,
    fit_baseline_treatment_effect_model,
    interaction_table,
    treatment_effect_table,
    validate_model_inputs,
)


EXPECTED_PATIENTS = 241
EXPECTED_BIOLOGICAL_FEATURES = 50
EXPECTED_DESIGN_FEATURES = 101


def main() -> None:

    print(
        "=== HERMES 2.0 "
        "Baseline Treatment-Effect Tests ==="
    )

    dataset = build_treatment_effect_dataset()

    X = dataset.X
    T = dataset.T
    Y = dataset.Y

    # ---------------------------------------------------------
    # 1. Input validation
    # ---------------------------------------------------------

    validate_model_inputs(
        X,
        T,
        Y,
    )

    print("PASS: model input validation")

    # ---------------------------------------------------------
    # 2. Fit model
    # ---------------------------------------------------------

    result = (
        fit_baseline_treatment_effect_model(
            X=X,
            treatment=T,
            outcome=Y,
            C=0.1,
        )
    )

    print("PASS: baseline model fitting")

    # ---------------------------------------------------------
    # 3. Dataset dimensions
    # ---------------------------------------------------------

    assert result.summary[
        "patients"
    ] == EXPECTED_PATIENTS

    assert result.summary[
        "biological_features"
    ] == EXPECTED_BIOLOGICAL_FEATURES

    assert result.summary[
        "design_features"
    ] == EXPECTED_DESIGN_FEATURES

    print("PASS: model dimensions")

    # ---------------------------------------------------------
    # 4. Treatment/outcome counts
    # ---------------------------------------------------------

    assert (
        result.summary["control_patients"]
        == 122
    )

    assert (
        result.summary["treated_patients"]
        == 119
    )

    assert (
        result.summary["residual_disease"]
        == 120
    )

    assert (
        result.summary["pcr"]
        == 121
    )

    print(
        "PASS: treatment/outcome preservation"
    )

    # ---------------------------------------------------------
    # 5. Interaction design dimensions
    # ---------------------------------------------------------

    X_scaled = (
        result.scaler.transform(
            X.to_numpy(dtype=float)
        )
    )

    design, names = (
        build_interaction_design(
            X_scaled,
            T.to_numpy(),
            X.columns,
        )
    )

    assert design.shape == (
        EXPECTED_PATIENTS,
        EXPECTED_DESIGN_FEATURES,
    )

    assert len(names) == (
        EXPECTED_DESIGN_FEATURES
    )

    assert names[0] == "T"

    print(
        "PASS: interaction design construction"
    )

    # ---------------------------------------------------------
    # 6. Probability validity
    # ---------------------------------------------------------

    probability_vectors = [
        result.observed_probability,
        result.probability_control,
        result.probability_treated,
    ]

    for probabilities in probability_vectors:

        assert len(
            probabilities
        ) == EXPECTED_PATIENTS

        assert np.isfinite(
            probabilities.to_numpy()
        ).all()

        assert (
            probabilities >= 0
        ).all()

        assert (
            probabilities <= 1
        ).all()

    print(
        "PASS: counterfactual probability validity"
    )

    # ---------------------------------------------------------
    # 7. ITE arithmetic
    # ---------------------------------------------------------

    expected_ite = (
        result.probability_treated
        - result.probability_control
    )

    np.testing.assert_allclose(
        result.ite.to_numpy(),
        expected_ite.to_numpy(),
        rtol=0,
        atol=1e-12,
    )

    assert (
        result.ite.index
        .equals(X.index)
    )

    assert np.isfinite(
        result.ite.to_numpy()
    ).all()

    assert (
        result.ite >= -1
    ).all()

    assert (
        result.ite <= 1
    ).all()

    print(
        "PASS: individualized treatment-effect arithmetic"
    )

    # ---------------------------------------------------------
    # 8. Coefficient structure
    # ---------------------------------------------------------

    assert (
        len(
            result.biological_coefficients
        )
        == EXPECTED_BIOLOGICAL_FEATURES
    )

    assert (
        len(
            result.interaction_coefficients
        )
        == EXPECTED_BIOLOGICAL_FEATURES
    )

    assert (
        result.biological_coefficients
        .index.tolist()
        == X.columns.tolist()
    )

    assert (
        result.interaction_coefficients
        .index.tolist()
        == X.columns.tolist()
    )

    assert np.isfinite(
        result.treatment_coefficient
    )

    print(
        "PASS: coefficient structure"
    )

    # ---------------------------------------------------------
    # 9. Treatment-effect table
    # ---------------------------------------------------------

    patient_table = (
        treatment_effect_table(
            result
        )
    )

    assert patient_table.shape == (
        EXPECTED_PATIENTS,
        6,
    )

    assert patient_table.index.equals(
        X.index
    )

    required_patient_columns = {
        "T",
        "Y",
        "observed_pcr_probability",
        "pcr_probability_CT",
        "pcr_probability_CT_A",
        "estimated_ite",
    }

    assert set(
        patient_table.columns
    ) == required_patient_columns

    print(
        "PASS: patient treatment-effect table"
    )

    # ---------------------------------------------------------
    # 10. Interaction ranking table
    # ---------------------------------------------------------

    interactions = (
        interaction_table(
            result
        )
    )

    assert interactions.shape[0] == (
        EXPECTED_BIOLOGICAL_FEATURES
    )

    assert {
        "biological_main_effect",
        "treatment_interaction",
        "absolute_treatment_interaction",
    }.issubset(
        interactions.columns
    )

    assert interactions[
        "absolute_treatment_interaction"
    ].is_monotonic_decreasing

    print(
        "PASS: interaction ranking table"
    )

    # ---------------------------------------------------------
    # 11. Deterministic fitting
    # ---------------------------------------------------------

    repeat = (
        fit_baseline_treatment_effect_model(
            X=X,
            treatment=T,
            outcome=Y,
            C=0.1,
        )
    )

    np.testing.assert_allclose(
        result.model.coef_,
        repeat.model.coef_,
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.ite.to_numpy(),
        repeat.ite.to_numpy(),
        rtol=0,
        atol=1e-12,
    )

    print(
        "PASS: deterministic model fitting"
    )

    # ---------------------------------------------------------
    # 12. Mean treatment effect sanity
    # ---------------------------------------------------------

    mean_ite = float(
        result.ite.mean()
    )

    assert -1.0 <= mean_ite <= 1.0

    print(
        "PASS: mean treatment-effect sanity"
    )

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print(
        "=========================================="
    )
    print(
        "ALL BASELINE TREATMENT-EFFECT TESTS PASSED"
    )
    print(
        "=========================================="
    )

    print()

    print(
        "Patients:",
        result.summary["patients"],
    )

    print(
        "Biological features:",
        result.summary[
            "biological_features"
        ],
    )

    print(
        "Design features:",
        result.summary[
            "design_features"
        ],
    )

    print(
        "Mean estimated ITE:",
        f"{result.ite.mean():.4f}",
    )

    print(
        "ITE range:",
        f"{result.ite.min():.4f}",
        "to",
        f"{result.ite.max():.4f}",
    )


if __name__ == "__main__":
    main()