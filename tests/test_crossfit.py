"""
HERMES 2.0
Cross-Fitted Treatment-Effect Tests
===================================

Validation suite for the HERMES cross-fitting engine.

These tests verify:
- correct patient coverage
- valid fold assignments
- treatment/outcome preservation
- valid probability estimates
- correct counterfactual ITE arithmetic
- joint-stratum balancing
- fold-level integrity
- deterministic cross-fitting
- absence of missing predictions
- summary consistency
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.crossfit import (
    build_joint_strata,
    crossfit_patient_table,
    crossfit_treatment_effect_model,
    validate_crossfit_configuration,
)

from backend.app.treatment_effects.feature_builder import (
    build_treatment_effect_dataset,
)


def assert_close(
    a: float,
    b: float,
    *,
    atol: float = 1e-10,
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
        "Cross-Fitted Treatment-Effect Tests ==="
    )

    dataset = build_treatment_effect_dataset()

    X = dataset.X
    T = dataset.T
    Y = dataset.Y

    # ---------------------------------------------------------
    # Basic configuration
    # ---------------------------------------------------------

    validate_crossfit_configuration(
        T,
        Y,
        n_splits=5,
    )

    print(
        "PASS: cross-fit configuration validation"
    )

    # ---------------------------------------------------------
    # Joint treatment x outcome strata
    # ---------------------------------------------------------

    strata = build_joint_strata(
        T,
        Y,
    )

    expected_strata = {
        "T0_Y0",
        "T0_Y1",
        "T1_Y0",
        "T1_Y1",
    }

    assert set(
        strata.unique()
    ) == expected_strata

    print(
        "PASS: joint treatment/outcome strata"
    )

    # ---------------------------------------------------------
    # Run cross-fitting
    # ---------------------------------------------------------

    result = crossfit_treatment_effect_model(
        X=X,
        treatment=T,
        outcome=Y,
        n_splits=5,
        C=0.1,
        random_state=42,
    )

    # ---------------------------------------------------------
    # Patient coverage
    # ---------------------------------------------------------

    assert result.n_patients == len(X)

    assert result.ite.index.equals(
        X.index
    )

    assert result.fold.index.equals(
        X.index
    )

    print(
        "PASS: complete patient coverage"
    )

    # ---------------------------------------------------------
    # Every patient assigned exactly one fold
    # ---------------------------------------------------------

    assert not result.fold.isna().any()

    assert result.fold.between(
        1,
        5,
    ).all()

    assert set(
        result.fold.unique()
    ) == {
        1,
        2,
        3,
        4,
        5,
    }

    print(
        "PASS: valid fold assignments"
    )

    # ---------------------------------------------------------
    # Fold sizes
    # ---------------------------------------------------------

    fold_sizes = (
        result.fold
        .value_counts()
    )

    assert (
        fold_sizes.max()
        - fold_sizes.min()
        <= 1
    )

    assert fold_sizes.sum() == len(X)

    print(
        "PASS: balanced fold sizes"
    )

    # ---------------------------------------------------------
    # Treatment/outcome preservation
    # ---------------------------------------------------------

    pd.testing.assert_series_equal(
        result.treatment,
        T.astype(int).rename("T"),
    )

    pd.testing.assert_series_equal(
        result.outcome,
        Y.astype(int).rename("Y"),
    )

    print(
        "PASS: treatment/outcome preservation"
    )

    # ---------------------------------------------------------
    # No missing predictions
    # ---------------------------------------------------------

    probability_objects = [
        result.observed_probability,
        result.probability_control,
        result.probability_treated,
        result.ite,
    ]

    for obj in probability_objects:
        assert not obj.isna().any()

    print(
        "PASS: no missing cross-fitted estimates"
    )

    # ---------------------------------------------------------
    # Probability validity
    # ---------------------------------------------------------

    for probability in [
        result.observed_probability,
        result.probability_control,
        result.probability_treated,
    ]:
        assert (
            probability >= 0.0
        ).all()

        assert (
            probability <= 1.0
        ).all()

    print(
        "PASS: probability bounds"
    )

    # ---------------------------------------------------------
    # Observed prediction must equal correct treatment-specific
    # counterfactual prediction.
    # ---------------------------------------------------------

    control_mask = (
        result.treatment == 0
    )

    treated_mask = (
        result.treatment == 1
    )

    np.testing.assert_allclose(
        result.observed_probability.loc[
            control_mask
        ].to_numpy(),
        result.probability_control.loc[
            control_mask
        ].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        result.observed_probability.loc[
            treated_mask
        ].to_numpy(),
        result.probability_treated.loc[
            treated_mask
        ].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    print(
        "PASS: observed/counterfactual consistency"
    )

    # ---------------------------------------------------------
    # ITE arithmetic
    # ---------------------------------------------------------

    expected_ite = (
        result.probability_treated
        - result.probability_control
    )

    np.testing.assert_allclose(
        result.ite.to_numpy(),
        expected_ite.to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    print(
        "PASS: cross-fitted ITE arithmetic"
    )

    # ---------------------------------------------------------
    # ITE theoretical probability bounds
    # ---------------------------------------------------------

    assert (
        result.ite >= -1.0
    ).all()

    assert (
        result.ite <= 1.0
    ).all()

    print(
        "PASS: ITE bounds"
    )

    # ---------------------------------------------------------
    # Joint-stratum balance across folds
    # ---------------------------------------------------------

    balance = pd.crosstab(
        result.fold,
        strata,
    )

    for column in balance.columns:
        assert (
            balance[column].max()
            - balance[column].min()
            <= 1
        )

    print(
        "PASS: joint-stratum fold balance"
    )

    # ---------------------------------------------------------
    # Fold summary integrity
    # ---------------------------------------------------------

    fold_summary = result.fold_summary

    assert len(fold_summary) == 5

    assert (
        fold_summary["test_n"].sum()
        == len(X)
    )

    assert (
        fold_summary[
            "test_CT"
        ].sum()
        == int((T == 0).sum())
    )

    assert (
        fold_summary[
            "test_CT_A"
        ].sum()
        == int((T == 1).sum())
    )

    assert (
        fold_summary[
            "test_RD"
        ].sum()
        == int((Y == 0).sum())
    )

    assert (
        fold_summary[
            "test_pCR"
        ].sum()
        == int((Y == 1).sum())
    )

    print(
        "PASS: fold summary integrity"
    )

    # ---------------------------------------------------------
    # Fold train/test disjointness implied by counts
    # ---------------------------------------------------------

    for _, row in (
        fold_summary.iterrows()
    ):
        assert (
            int(row["train_n"])
            + int(row["test_n"])
            == len(X)
        )

    print(
        "PASS: fold train/test size integrity"
    )

    # ---------------------------------------------------------
    # Patient table
    # ---------------------------------------------------------

    patient_table = (
        crossfit_patient_table(
            result
        )
    )

    assert len(patient_table) == len(X)

    assert patient_table.index.equals(
        X.index
    )

    required_columns = {
        "fold",
        "T",
        "Y",
        "observed_pcr_probability",
        "pcr_probability_CT",
        "pcr_probability_CT_A",
        "crossfitted_ite",
    }

    assert set(
        patient_table.columns
    ) == required_columns

    print(
        "PASS: patient treatment-effect table"
    )

    # ---------------------------------------------------------
    # Summary consistency
    # ---------------------------------------------------------

    summary = result.summary

    assert (
        summary["patients"]
        == len(X)
    )

    assert (
        summary["biological_features"]
        == X.shape[1]
    )

    assert (
        summary["n_splits"]
        == 5
    )

    assert_close(
        summary[
            "mean_crossfitted_ite"
        ],
        float(
            result.ite.mean()
        ),
    )

    assert_close(
        summary[
            "median_crossfitted_ite"
        ],
        float(
            result.ite.median()
        ),
    )

    assert_close(
        summary[
            "minimum_crossfitted_ite"
        ],
        float(
            result.ite.min()
        ),
    )

    assert_close(
        summary[
            "maximum_crossfitted_ite"
        ],
        float(
            result.ite.max()
        ),
    )

    print(
        "PASS: summary consistency"
    )

    # ---------------------------------------------------------
    # Observed randomized treatment difference
    # ---------------------------------------------------------

    expected_CT_rate = float(
        Y[T == 0].mean()
    )

    expected_CT_A_rate = float(
        Y[T == 1].mean()
    )

    expected_difference = (
        expected_CT_A_rate
        - expected_CT_rate
    )

    assert_close(
        summary[
            "observed_CT_pcr_rate"
        ],
        expected_CT_rate,
    )

    assert_close(
        summary[
            "observed_CT_A_pcr_rate"
        ],
        expected_CT_A_rate,
    )

    assert_close(
        summary[
            "observed_risk_difference"
        ],
        expected_difference,
    )

    print(
        "PASS: randomized treatment-effect summary"
    )

    # ---------------------------------------------------------
    # Determinism
    # ---------------------------------------------------------

    result_repeat = (
        crossfit_treatment_effect_model(
            X=X,
            treatment=T,
            outcome=Y,
            n_splits=5,
            C=0.1,
            random_state=42,
        )
    )

    pd.testing.assert_series_equal(
        result.fold,
        result_repeat.fold,
    )

    np.testing.assert_allclose(
        result.ite.to_numpy(),
        result_repeat.ite.to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    np.testing.assert_allclose(
        result.observed_probability.to_numpy(),
        result_repeat.observed_probability.to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    print(
        "PASS: deterministic cross-fitting"
    )

    # ---------------------------------------------------------
    # Biological heterogeneity sanity check
    #
    # This is NOT evidence that heterogeneity is real.
    # It merely confirms the estimator is not returning one
    # identical treatment effect for every patient.
    # ---------------------------------------------------------

    assert result.ite.std() > 0.0

    assert result.ite.nunique() > 1

    print(
        "PASS: non-degenerate ITE distribution"
    )

    print()

    print(
        "=========================================="
    )

    print(
        "ALL CROSS-FITTED TREATMENT-EFFECT "
        "TESTS PASSED"
    )

    print(
        "=========================================="
    )

    print()

    print(
        f"Patients: {len(X)}"
    )

    print(
        f"Folds: "
        f"{result.summary['n_splits']}"
    )

    print(
        "OOF AUC: "
        f"{result.summary['crossfitted_observed_auc']:.4f}"
    )

    print(
        "OOF Brier: "
        f"{result.summary['crossfitted_observed_brier']:.4f}"
    )

    print(
        "Mean cross-fitted ITE: "
        f"{result.ite.mean():.4f}"
    )

    print(
        "ITE SD: "
        f"{result.ite.std():.4f}"
    )

    print(
        "ITE range: "
        f"{result.ite.min():.4f} "
        f"to "
        f"{result.ite.max():.4f}"
    )


if __name__ == "__main__":
    run_tests()