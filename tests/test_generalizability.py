"""
HERMES 2.0
Generalizability / Applicability / OOD Tests
============================================

Validation suite for the HERMES biological applicability framework.

The tests verify:

1. Reference-model construction
2. Reference z-score behavior
3. Mahalanobis-distance behavior
4. Patient applicability classification
5. Synthetic OOD positive controls
6. Cohort-shift detection
7. Deterministic holdout behavior
8. Feature compatibility enforcement
9. Deterministic applicability estimation
10. Lightweight NeoTRIP integration

These tests validate software capability and internal consistency.

They do NOT establish:
    - external clinical validation
    - causal transportability
    - clinical treatment safety
    - predictive-biomarker validity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.generalizability import (
    assess_applicability,
    compare_cohort_shift,
    fit_applicability_reference,
    generate_shifted_cohort,
    mahalanobis_distance,
    split_reference_holdout,
    standardized_feature_deviation,
)


# =============================================================
# Test helper
# =============================================================


def check(
    condition: bool,
    message: str,
) -> None:
    if not bool(condition):
        raise AssertionError(message)

    print(f"PASS: {message}")


# =============================================================
# Synthetic biological reference
# =============================================================


def generate_reference_cohort(
    *,
    n_patients: int = 300,
    n_features: int = 8,
    random_state: int = 2026,
) -> pd.DataFrame:
    """
    Generate a correlated synthetic biological reference cohort.
    """

    if n_patients < 50:
        raise ValueError(
            "n_patients must be at least 50."
        )

    if n_features < 2:
        raise ValueError(
            "n_features must be at least 2."
        )

    rng = np.random.default_rng(
        random_state
    )

    latent = rng.normal(
        0.0,
        1.0,
        size=(
            n_patients,
            3,
        ),
    )

    loading = rng.normal(
        0.0,
        0.5,
        size=(
            3,
            n_features,
        ),
    )

    noise = rng.normal(
        0.0,
        0.65,
        size=(
            n_patients,
            n_features,
        ),
    )

    X = (
        latent
        @ loading
        + noise
    )

    index = pd.Index(
        [
            f"REF_{i:05d}"
            for i in range(n_patients)
        ],
        name="Patient_ID",
    )

    columns = [
        f"PATHWAY_{i:02d}"
        for i in range(n_features)
    ]

    return pd.DataFrame(
        X,
        index=index,
        columns=columns,
    )


# =============================================================
# Main tests
# =============================================================


def main() -> None:

    print(
        "=== HERMES 2.0 GENERALIZABILITY / "
        "APPLICABILITY / OOD TESTS ==="
    )

    print()

    # =========================================================
    # 1. Reference construction
    # =========================================================

    X = generate_reference_cohort(
        n_patients=300,
        n_features=8,
        random_state=2026,
    )

    check(
        X.shape == (300, 8),
        "synthetic reference dimensions",
    )

    reference = fit_applicability_reference(
        X,
        borderline_quantile=0.95,
        ood_quantile=0.99,
    )

    check(
        reference.summary[
            "reference_patients"
        ]
        == 300,
        "reference patient count",
    )

    check(
        reference.summary[
            "biological_features"
        ]
        == 8,
        "reference biological feature count",
    )

    check(
        reference.feature_names
        == tuple(X.columns),
        "reference feature-name preservation",
    )

    check(
        reference.mean.index.equals(
            X.columns
        ),
        "reference mean feature alignment",
    )

    check(
        reference.standard_deviation.index.equals(
            X.columns
        ),
        "reference SD feature alignment",
    )

    check(
        reference.covariance.shape
        == (8, 8),
        "reference covariance dimensions",
    )

    check(
        reference.precision.shape
        == (8, 8),
        "reference precision dimensions",
    )

    check(
        np.isfinite(
            reference.precision.to_numpy()
        ).all(),
        "finite shrinkage precision matrix",
    )

    check(
        reference.mahalanobis_ood_threshold
        > reference.mahalanobis_borderline_threshold,
        "Mahalanobis threshold ordering",
    )

    check(
        reference.max_abs_z_ood_threshold
        > reference.max_abs_z_borderline_threshold,
        "maximum-z threshold ordering",
    )

    check(
        reference.mean_abs_z_ood_threshold
        > reference.mean_abs_z_borderline_threshold,
        "mean-z threshold ordering",
    )

    # =========================================================
    # 2. Standardized deviations
    # =========================================================

    z = standardized_feature_deviation(
        X,
        reference.mean,
        reference.standard_deviation,
    )

    check(
        z.shape == X.shape,
        "reference z-score dimensions",
    )

    check(
        np.isfinite(
            z.to_numpy()
        ).all(),
        "finite reference z-scores",
    )

    check(
        np.allclose(
            z.mean(axis=0).to_numpy(),
            np.zeros(
                X.shape[1]
            ),
            atol=1e-10,
        ),
        "reference z-score means approximately zero",
    )

    reference_z_sd = z.std(
        axis=0,
        ddof=1,
    )

    check(
        np.allclose(
            reference_z_sd.to_numpy(),
            np.ones(
                X.shape[1]
            ),
            atol=1e-10,
        ),
        "reference z-score SD approximately one",
    )

    # =========================================================
    # 3. Mahalanobis behavior
    # =========================================================

    distances = mahalanobis_distance(
        X,
        reference.mean,
        reference.precision,
    )

    check(
        len(distances) == 300,
        "reference Mahalanobis patient coverage",
    )

    check(
        np.isfinite(
            distances.to_numpy()
        ).all(),
        "finite Mahalanobis distances",
    )

    check(
        (
            distances
            >= 0.0
        ).all(),
        "non-negative Mahalanobis distances",
    )

    # Exact reference center should have distance ~0.
    center = pd.DataFrame(
        [
            reference.mean.to_numpy(
                dtype=float
            )
        ],
        index=pd.Index(
            ["CENTER"],
            name="Patient_ID",
        ),
        columns=X.columns,
    )

    center_distance = mahalanobis_distance(
        center,
        reference.mean,
        reference.precision,
    )

    check(
        np.isclose(
            center_distance.iloc[0],
            0.0,
            atol=1e-10,
        ),
        "reference center Mahalanobis distance approximately zero",
    )

    # =========================================================
    # 4. Internal applicability
    # =========================================================

    internal = assess_applicability(
        reference,
        X.iloc[:50].copy(),
    )

    check(
        internal.patient_table.shape[0]
        == 50,
        "internal applicability patient count",
    )

    required_patient_columns = {
        "mahalanobis_distance",
        "mahalanobis_reference_percentile",
        "max_abs_z",
        "max_abs_z_reference_percentile",
        "mean_abs_z",
        "mean_abs_z_reference_percentile",
        "fraction_features_abs_z_gt_2",
        "fraction_features_abs_z_gt_3",
        "n_ood_flags",
        "n_borderline_flags",
        "applicability_state",
        "applicability_score",
    }

    check(
        required_patient_columns.issubset(
            internal.patient_table.columns
        ),
        "patient applicability schema",
    )

    check(
        (
            internal.patient_table[
                "applicability_score"
            ]
            >= 0.0
        ).all()
        and (
            internal.patient_table[
                "applicability_score"
            ]
            <= 1.0
        ).all(),
        "applicability-score bounds",
    )

    check(
        set(
            internal.patient_table[
                "applicability_state"
            ].unique()
        ).issubset(
            {
                "in_distribution",
                "borderline",
                "out_of_distribution",
            }
        ),
        "valid applicability states",
    )

    check(
        (
            internal.summary[
                "in_distribution_n"
            ]
            + internal.summary[
                "borderline_n"
            ]
            + internal.summary[
                "out_of_distribution_n"
            ]
        )
        == 50,
        "applicability state count conservation",
    )

    # =========================================================
    # 5. Strong synthetic OOD positive control
    # =========================================================

    shifted = generate_shifted_cohort(
        X,
        n_patients=120,
        mean_shift=2.5,
        scale_multiplier=1.0,
        random_state=99,
    )

    check(
        shifted.shape
        == (120, 8),
        "synthetic shifted cohort dimensions",
    )

    shifted_assessment = assess_applicability(
        reference,
        shifted,
    )

    check(
        shifted_assessment.summary[
            "fraction_out_of_distribution"
        ]
        > internal.summary[
            "fraction_out_of_distribution"
        ],
        "synthetic shift increases OOD detection",
    )

    check(
        shifted_assessment.summary[
            "median_mahalanobis_distance"
        ]
        > internal.summary[
            "median_mahalanobis_distance"
        ],
        "synthetic shift increases Mahalanobis distance",
    )

    check(
        shifted_assessment.summary[
            "median_applicability_score"
        ]
        < internal.summary[
            "median_applicability_score"
        ],
        "synthetic shift decreases applicability score",
    )

    check(
        shifted_assessment.summary[
            "fraction_out_of_distribution"
        ]
        >= 0.75,
        "strong synthetic shift predominantly detected as OOD",
    )

    # =========================================================
    # 6. Shift-strength monotonicity
    # =========================================================

    shift_05 = generate_shifted_cohort(
        X,
        n_patients=200,
        mean_shift=0.5,
        random_state=123,
    )

    shift_15 = generate_shifted_cohort(
        X,
        n_patients=200,
        mean_shift=1.5,
        random_state=123,
    )

    shift_30 = generate_shifted_cohort(
        X,
        n_patients=200,
        mean_shift=3.0,
        random_state=123,
    )

    assessment_05 = assess_applicability(
        reference,
        shift_05,
    )

    assessment_15 = assess_applicability(
        reference,
        shift_15,
    )

    assessment_30 = assess_applicability(
        reference,
        shift_30,
    )

    check(
        assessment_05.summary[
            "median_mahalanobis_distance"
        ]
        < assessment_15.summary[
            "median_mahalanobis_distance"
        ]
        < assessment_30.summary[
            "median_mahalanobis_distance"
        ],
        "OOD distance increases with injected shift strength",
    )

    check(
        assessment_05.summary[
            "median_applicability_score"
        ]
        > assessment_15.summary[
            "median_applicability_score"
        ]
        >= assessment_30.summary[
            "median_applicability_score"
        ],
        "applicability decreases with injected shift strength",
    )

    # =========================================================
    # 7. Cohort-shift diagnostics
    # =========================================================

    cohort_shift = compare_cohort_shift(
        X,
        shifted,
        applicability_reference=reference,
    )

    check(
        cohort_shift.feature_shift_table.shape[0]
        == 8,
        "cohort-shift feature coverage",
    )

    required_shift_columns = {
        "reference_mean",
        "target_mean",
        "mean_difference",
        "standardized_mean_difference",
        "absolute_standardized_mean_difference",
        "reference_sd",
        "target_sd",
        "variance_ratio",
    }

    check(
        required_shift_columns.issubset(
            cohort_shift.feature_shift_table.columns
        ),
        "cohort-shift table schema",
    )

    check(
        cohort_shift.summary[
            "maximum_absolute_smd"
        ]
        >= cohort_shift.summary[
            "median_absolute_smd"
        ],
        "cohort-shift magnitude ordering",
    )

    check(
        cohort_shift.summary[
            "fraction_features_abs_smd_ge_0_50"
        ]
        > 0.50,
        "large synthetic shift detected across biological features",
    )

    # =========================================================
    # 8. Deterministic holdout splitting
    # =========================================================

    reference_a, holdout_a = (
        split_reference_holdout(
            X,
            holdout_fraction=0.20,
            random_state=42,
        )
    )

    reference_b, holdout_b = (
        split_reference_holdout(
            X,
            holdout_fraction=0.20,
            random_state=42,
        )
    )

    check(
        reference_a.equals(
            reference_b
        ),
        "deterministic reference split",
    )

    check(
        holdout_a.equals(
            holdout_b
        ),
        "deterministic holdout split",
    )

    check(
        len(reference_a)
        + len(holdout_a)
        == len(X),
        "holdout split patient conservation",
    )

    check(
        set(reference_a.index).isdisjoint(
            set(holdout_a.index)
        ),
        "reference and holdout patient separation",
    )

    # =========================================================
    # 9. Feature compatibility enforcement
    # =========================================================

    incorrect = shifted.copy()

    incorrect = incorrect.rename(
        columns={
            incorrect.columns[0]:
                "WRONG_FEATURE"
        }
    )

    compatibility_error = False

    try:
        assess_applicability(
            reference,
            incorrect,
        )

    except ValueError:
        compatibility_error = True

    check(
        compatibility_error,
        "feature mismatch correctly rejected",
    )

    reordered = shifted[
        list(
            reversed(
                shifted.columns
            )
        )
    ]

    order_error = False

    try:
        assess_applicability(
            reference,
            reordered,
        )

    except ValueError:
        order_error = True

    check(
        order_error,
        "feature-order mismatch correctly rejected",
    )

    # =========================================================
    # 10. Deterministic applicability
    # =========================================================

    first_assessment = assess_applicability(
        reference,
        shifted.iloc[:30],
    )

    second_assessment = assess_applicability(
        reference,
        shifted.iloc[:30],
    )

    pd.testing.assert_frame_equal(
        first_assessment.patient_table,
        second_assessment.patient_table,
        check_exact=True,
    )

    check(
        True,
        "deterministic patient applicability table",
    )

    pd.testing.assert_frame_equal(
        first_assessment.feature_z_scores,
        second_assessment.feature_z_scores,
        check_exact=True,
    )

    check(
        True,
        "deterministic feature z-scores",
    )

    check(
        first_assessment.summary
        == second_assessment.summary,
        "deterministic applicability summary",
    )

    # =========================================================
    # 11. Lightweight NeoTRIP integration
    # =========================================================

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    neotrip = build_treatment_effect_dataset()

    check(
        neotrip.X.shape[0]
        == 241,
        "NeoTRIP generalizability patient count",
    )

    check(
        neotrip.X.shape[1]
        == 50,
        "NeoTRIP generalizability biological feature count",
    )

    neotrip_reference_matrix, neotrip_holdout = (
        split_reference_holdout(
            neotrip.X,
            holdout_fraction=0.20,
            random_state=42,
        )
    )

    check(
        len(neotrip_reference_matrix)
        == 193,
        "NeoTRIP reference subset size",
    )

    check(
        len(neotrip_holdout)
        == 48,
        "NeoTRIP holdout subset size",
    )

    neotrip_reference = (
        fit_applicability_reference(
            neotrip_reference_matrix,
            borderline_quantile=0.95,
            ood_quantile=0.99,
        )
    )

    neotrip_holdout_assessment = (
        assess_applicability(
            neotrip_reference,
            neotrip_holdout,
        )
    )

    check(
        neotrip_holdout_assessment.patient_table.shape[
            0
        ]
        == 48,
        "complete NeoTRIP holdout applicability coverage",
    )

    check(
        np.isfinite(
            neotrip_holdout_assessment
            .patient_table[
                "mahalanobis_distance"
            ]
            .to_numpy()
        ).all(),
        "finite NeoTRIP holdout Mahalanobis distances",
    )

    check(
        (
            neotrip_holdout_assessment
            .patient_table[
                "applicability_score"
            ]
            >= 0.0
        ).all()
        and (
            neotrip_holdout_assessment
            .patient_table[
                "applicability_score"
            ]
            <= 1.0
        ).all(),
        "NeoTRIP applicability-score bounds",
    )

    neotrip_shifted = generate_shifted_cohort(
        neotrip_reference_matrix,
        n_patients=80,
        mean_shift=2.0,
        scale_multiplier=1.0,
        random_state=2026,
    )

    neotrip_shifted_assessment = (
        assess_applicability(
            neotrip_reference,
            neotrip_shifted,
        )
    )

    check(
        neotrip_shifted_assessment.summary[
            "median_mahalanobis_distance"
        ]
        > neotrip_holdout_assessment.summary[
            "median_mahalanobis_distance"
        ],
        "NeoTRIP synthetic OOD positive control",
    )

    check(
        neotrip_shifted_assessment.summary[
            "fraction_out_of_distribution"
        ]
        > neotrip_holdout_assessment.summary[
            "fraction_out_of_distribution"
        ],
        "NeoTRIP shifted cohort increases OOD rate",
    )

    neotrip_shift = compare_cohort_shift(
        neotrip_reference_matrix,
        neotrip_shifted,
        applicability_reference=(
            neotrip_reference
        ),
    )

    check(
        neotrip_shift.summary[
            "target_patients"
        ]
        == 80,
        "NeoTRIP shifted cohort count",
    )

    check(
        neotrip_shift.summary[
            "biological_features"
        ]
        == 50,
        "NeoTRIP cohort-shift feature count",
    )

    # =========================================================
    # Final report
    # =========================================================

    print()

    print(
        "============================================="
    )

    print(
        "ALL GENERALIZABILITY / APPLICABILITY / "
        "OOD TESTS PASSED"
    )

    print(
        "============================================="
    )

    print()

    print(
        "Synthetic reference:"
    )

    print(
        f"Patients: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    print(
        "Reference median Mahalanobis: "
        f"{reference.reference_mahalanobis.median():.4f}"
    )

    print()

    print(
        "Strong synthetic shift:"
    )

    print(
        "Median Mahalanobis: "
        f"{shifted_assessment.summary['median_mahalanobis_distance']:.4f}"
    )

    print(
        "Median applicability score: "
        f"{shifted_assessment.summary['median_applicability_score']:.4f}"
    )

    print(
        "Fraction OOD: "
        f"{shifted_assessment.summary['fraction_out_of_distribution']:.4f}"
    )

    print()

    print(
        "NeoTRIP holdout:"
    )

    print(
        f"Reference patients: {len(neotrip_reference_matrix)}"
    )

    print(
        f"Holdout patients: {len(neotrip_holdout)}"
    )

    print(
        "Holdout median Mahalanobis: "
        f"{neotrip_holdout_assessment.summary['median_mahalanobis_distance']:.4f}"
    )

    print(
        "Holdout fraction OOD: "
        f"{neotrip_holdout_assessment.summary['fraction_out_of_distribution']:.4f}"
    )

    print()

    print(
        "NeoTRIP synthetic shift:"
    )

    print(
        "Shifted median Mahalanobis: "
        f"{neotrip_shifted_assessment.summary['median_mahalanobis_distance']:.4f}"
    )

    print(
        "Shifted fraction OOD: "
        f"{neotrip_shifted_assessment.summary['fraction_out_of_distribution']:.4f}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "These tests demonstrate internal applicability/OOD detection "
        "behavior and synthetic-shift recovery."
    )

    print(
        "They do not constitute independent external validation."
    )


if __name__ == "__main__":
    main()