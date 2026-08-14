"""
HERMES 2.0
Treatment-Effect Feature Builder Tests
======================================

Validation tests for the canonical NeoTRIP treatment-effect
modeling dataset.

These tests verify:

1. Dataset dimensions
2. Patient alignment
3. Treatment encoding
4. Outcome encoding
5. Randomized-arm counts
6. Outcome counts
7. Feature integrity
8. Hallmark representation integrity
9. Deterministic construction
10. Biological feature independence from treatment/outcome labels
"""

import numpy as np
import pandas as pd

from backend.app.treatment_effects.feature_builder import (
    build_treatment_effect_dataset,
)


EXPECTED_PATIENTS = 241
EXPECTED_FEATURES = 50

EXPECTED_TREATMENT_COUNTS = {
    0: 122,  # CT
    1: 119,  # CT/A
}

EXPECTED_OUTCOME_COUNTS = {
    0: 120,  # residual disease
    1: 121,  # pCR
}


def main() -> None:

    print(
        "=== HERMES 2.0 "
        "Feature Builder Tests ==="
    )

    dataset = build_treatment_effect_dataset()

    X = dataset.X
    T = dataset.T
    Y = dataset.Y

    # ---------------------------------------------------------
    # 1. Dataset dimensions
    # ---------------------------------------------------------

    assert X.shape == (
        EXPECTED_PATIENTS,
        EXPECTED_FEATURES,
    )

    assert len(T) == EXPECTED_PATIENTS
    assert len(Y) == EXPECTED_PATIENTS

    print("PASS: dataset dimensions")

    # ---------------------------------------------------------
    # 2. Patient alignment
    # ---------------------------------------------------------

    assert X.index.equals(T.index)
    assert X.index.equals(Y.index)
    assert X.index.equals(
        dataset.metadata.index
    )

    assert not X.index.duplicated().any()

    print("PASS: patient alignment")

    # ---------------------------------------------------------
    # 3. Treatment encoding
    # ---------------------------------------------------------

    assert set(T.unique()) == {0, 1}

    treatment_counts = (
        T.value_counts()
        .sort_index()
        .to_dict()
    )

    assert (
        treatment_counts
        == EXPECTED_TREATMENT_COUNTS
    )

    print("PASS: treatment encoding")

    # ---------------------------------------------------------
    # 4. Outcome encoding
    # ---------------------------------------------------------

    assert set(Y.unique()) == {0, 1}

    outcome_counts = (
        Y.value_counts()
        .sort_index()
        .to_dict()
    )

    assert (
        outcome_counts
        == EXPECTED_OUTCOME_COUNTS
    )

    print("PASS: outcome encoding")

    # ---------------------------------------------------------
    # 5. Known NeoTRIP arm-specific pCR counts
    # ---------------------------------------------------------

    audit = pd.DataFrame(
        {
            "T": T,
            "Y": Y,
        }
    )

    arm_table = pd.crosstab(
        audit["T"],
        audit["Y"],
    )

    # CT:
    # 64 RD / 58 pCR
    assert arm_table.loc[0, 0] == 64
    assert arm_table.loc[0, 1] == 58

    # CT/A:
    # 56 RD / 63 pCR
    assert arm_table.loc[1, 0] == 56
    assert arm_table.loc[1, 1] == 63

    print(
        "PASS: randomized treatment/outcome counts"
    )

    # ---------------------------------------------------------
    # 6. Feature integrity
    # ---------------------------------------------------------

    assert not X.isna().any().any()

    assert np.isfinite(
        X.to_numpy(dtype=float)
    ).all()

    assert not X.columns.duplicated().any()

    assert X.shape[1] == EXPECTED_FEATURES

    print("PASS: biological feature integrity")

    # ---------------------------------------------------------
    # 7. Hallmark naming
    # ---------------------------------------------------------

    assert all(
        str(column).startswith("HALLMARK_")
        for column in X.columns
    )

    print("PASS: Hallmark feature naming")

    # ---------------------------------------------------------
    # 8. Biological feature variation
    # ---------------------------------------------------------

    feature_std = X.std(axis=0)

    assert (
        feature_std > 0
    ).all()

    print("PASS: biological feature variation")

    # ---------------------------------------------------------
    # 9. Dataset summary
    # ---------------------------------------------------------

    assert (
        dataset.summary["patients"]
        == EXPECTED_PATIENTS
    )

    assert (
        dataset.summary["features"]
        == EXPECTED_FEATURES
    )

    assert (
        dataset.summary[
            "hallmark_sets_loaded"
        ]
        == EXPECTED_FEATURES
    )

    assert (
        dataset.summary[
            "hallmark_sets_retained"
        ]
        == EXPECTED_FEATURES
    )

    print("PASS: dataset summary")

    # ---------------------------------------------------------
    # 10. Deterministic construction
    # ---------------------------------------------------------

    dataset_repeat = (
        build_treatment_effect_dataset()
    )

    pd.testing.assert_frame_equal(
        dataset.X,
        dataset_repeat.X,
        check_exact=True,
    )

    pd.testing.assert_series_equal(
        dataset.T,
        dataset_repeat.T,
        check_exact=True,
    )

    pd.testing.assert_series_equal(
        dataset.Y,
        dataset_repeat.Y,
        check_exact=True,
    )

    print("PASS: deterministic construction")

    # ---------------------------------------------------------
    # 11. Feature layer contains biology only
    # ---------------------------------------------------------

    forbidden_exact = {
        "t",
        "y",
        "treatment",
        "treatment_label",
        "arm",
        "outcome",
        "outcome_label",
        "pcr",
        "response",
        "label",
    }

    normalized_columns = {
        str(column).strip().lower()
        for column in X.columns
    }

    assert normalized_columns.isdisjoint(
        forbidden_exact
    )

    print(
        "PASS: no treatment/outcome columns "
        "inside biological feature matrix"
    )

    # ---------------------------------------------------------
    # 12. Internal validation
    # ---------------------------------------------------------

    dataset.validate()

    print("PASS: internal dataset validation")

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print(
        "======================================"
    )
    print(
        "ALL FEATURE BUILDER TESTS PASSED"
    )
    print(
        "======================================"
    )

    print()
    print(
        f"Patients: {X.shape[0]}"
    )

    print(
        f"Biological features: {X.shape[1]}"
    )

    print(
        f"Feature matrix: {X.shape}"
    )

    print()

    print("Treatment x outcome:")

    display_table = pd.crosstab(
        dataset.metadata[
            "treatment_label"
        ],
        dataset.metadata[
            "outcome_label"
        ],
        margins=True,
    )

    print(display_table)


if __name__ == "__main__":
    main()