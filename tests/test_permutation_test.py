"""
HERMES 2.0
Permutation / Null Validation Tests
===================================

Validation suite for the HERMES treatment-effect heterogeneity
permutation framework.

These tests verify:

1. Permutation-mode validation
2. Treatment permutation preserves allocation counts
3. Treatment permutation is deterministic
4. Feature permutation preserves patient/profile dimensions
5. Feature permutation preserves complete biological profiles
6. Feature permutation is deterministic
7. Observed heterogeneity statistics are valid
8. Empirical p-value arithmetic uses the +1 correction
9. Development-scale permutation analysis runs end-to-end
10. Null-statistic dimensions are correct
11. Empirical p-values are bounded
12. Observed/null comparison table is internally consistent
13. Reproducibility of the full permutation experiment

These are engineering/statistical-integrity tests.

Passing them does NOT imply that the biological heterogeneity
detected by HERMES is real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.feature_builder import (
    build_treatment_effect_dataset,
)

from backend.app.treatment_effects.permutation_test import (
    _empirical_upper_tail_p_value,
    _extract_heterogeneity_statistics,
    _validate_mode,
    permutation_comparison_table,
    permute_feature_profiles,
    permute_treatment,
    run_permutation_test,
)

from backend.app.treatment_effects.repeated_crossfit import (
    repeated_crossfit_treatment_effect_model,
)


def assert_close(
    a: float,
    b: float,
    *,
    atol: float = 1e-12,
) -> None:
    """
    Assert numerical equality within a strict tolerance.
    """

    assert np.isclose(
        a,
        b,
        atol=atol,
        rtol=0.0,
    ), f"{a} != {b}"


def sorted_row_signatures(
    frame: pd.DataFrame,
) -> list[tuple]:
    """
    Represent dataframe rows as sortable tuples.

    Used to confirm that feature permutation reorders complete
    biological profiles rather than modifying the profiles.
    """

    rows = [
        tuple(row)
        for row in frame.to_numpy()
    ]

    return sorted(
        rows,
        key=repr,
    )


def run_tests() -> None:

    print(
        "=== HERMES 2.0 "
        "Permutation / Null Validation Tests ==="
    )

    dataset = build_treatment_effect_dataset()

    X = dataset.X
    T = dataset.T
    Y = dataset.Y

    # =========================================================
    # 1. Permutation-mode validation
    # =========================================================

    assert (
        _validate_mode(
            "feature_permutation"
        )
        == "feature_permutation"
    )

    assert (
        _validate_mode(
            "treatment_permutation"
        )
        == "treatment_permutation"
    )

    invalid_mode_failed = False

    try:
        _validate_mode(
            "invalid_mode"
        )

    except ValueError:
        invalid_mode_failed = True

    assert invalid_mode_failed

    print(
        "PASS: permutation-mode validation"
    )

    # =========================================================
    # 2. Treatment permutation preserves structure
    # =========================================================

    T_permuted = permute_treatment(
        T,
        random_state=12345,
    )

    assert T_permuted.index.equals(
        T.index
    )

    assert len(T_permuted) == len(T)

    assert (
        int(
            (T_permuted == 0).sum()
        )
        ==
        int(
            (T == 0).sum()
        )
    )

    assert (
        int(
            (T_permuted == 1).sum()
        )
        ==
        int(
            (T == 1).sum()
        )
    )

    print(
        "PASS: treatment permutation preserves allocation"
    )

    # =========================================================
    # 3. Treatment permutation determinism
    # =========================================================

    T_permuted_repeat = permute_treatment(
        T,
        random_state=12345,
    )

    pd.testing.assert_series_equal(
        T_permuted,
        T_permuted_repeat,
    )

    print(
        "PASS: deterministic treatment permutation"
    )

    # =========================================================
    # 4. Feature permutation dimensions / labels
    # =========================================================

    X_permuted = permute_feature_profiles(
        X,
        random_state=54321,
    )

    assert X_permuted.shape == X.shape

    assert X_permuted.index.equals(
        X.index
    )

    assert X_permuted.columns.equals(
        X.columns
    )

    print(
        "PASS: feature permutation dimensions"
    )

    # =========================================================
    # 5. Complete profiles must be preserved
    #
    # Feature permutation should shuffle ROWS as intact vectors.
    # It must not independently scramble pathway columns.
    # =========================================================

    original_profiles = (
        sorted_row_signatures(
            X
        )
    )

    permuted_profiles = (
        sorted_row_signatures(
            X_permuted
        )
    )

    assert (
        original_profiles
        == permuted_profiles
    )

    print(
        "PASS: complete biological profiles preserved"
    )

    # =========================================================
    # 6. Feature permutation determinism
    # =========================================================

    X_permuted_repeat = (
        permute_feature_profiles(
            X,
            random_state=54321,
        )
    )

    pd.testing.assert_frame_equal(
        X_permuted,
        X_permuted_repeat,
    )

    print(
        "PASS: deterministic feature permutation"
    )

    # =========================================================
    # 7. Heterogeneity-statistic extraction
    #
    # Use only 2 repeats here to keep the unit test reasonably
    # fast. This is an implementation test, not final analysis.
    # =========================================================

    observed_small = (
        repeated_crossfit_treatment_effect_model(
            X=X,
            treatment=T,
            outcome=Y,
            n_repeats=2,
            n_splits=5,
            C=0.1,
            base_random_state=42,
        )
    )

    observed_statistics = (
        _extract_heterogeneity_statistics(
            observed_small
        )
    )

    expected_statistics = {
        "ite_sd_across_patients",
        "ite_mean_absolute_deviation",
        "ite_iqr",
        "ite_absolute_90th_percentile",
        "ite_max_absolute",
        "fraction_sign_stability_ge_90pct",
        "fraction_unanimous_sign",
        "median_stability_signal_ratio",
        "cohort_mean_ite",
    }

    assert set(
        observed_statistics.index
    ) == expected_statistics

    assert np.isfinite(
        observed_statistics.to_numpy(
            dtype=float
        )
    ).all()

    assert (
        observed_statistics[
            "ite_sd_across_patients"
        ]
        >= 0
    )

    assert (
        observed_statistics[
            "ite_iqr"
        ]
        >= 0
    )

    assert (
        observed_statistics[
            "ite_max_absolute"
        ]
        >= 0
    )

    print(
        "PASS: heterogeneity-statistic extraction"
    )

    # =========================================================
    # 8. Empirical p-value +1 correction
    # =========================================================

    null_example = pd.Series(
        [
            0.10,
            0.20,
            0.30,
        ]
    )

    # observed = 0.25
    #
    # null >= observed:
    #     only 0.30 -> 1 exceedance
    #
    # p = (1 + 1) / (3 + 1)
    #   = 0.50
    example_p = (
        _empirical_upper_tail_p_value(
            observed=0.25,
            null_values=null_example,
        )
    )

    assert_close(
        example_p,
        0.50,
    )

    # If observed exceeds all null values:
    #
    # p = 1 / (B + 1)
    minimum_example_p = (
        _empirical_upper_tail_p_value(
            observed=1.0,
            null_values=null_example,
        )
    )

    assert_close(
        minimum_example_p,
        0.25,
    )

    print(
        "PASS: empirical p-value correction"
    )

    # =========================================================
    # 9. End-to-end DEVELOPMENT permutation test
    #
    # Three permutations x two repeated cross-fit runs.
    #
    # This is intentionally small so the unit suite does not
    # require a publication-scale computation.
    # =========================================================

    result = run_permutation_test(
        X=X,
        treatment=T,
        outcome=Y,
        permutation_mode=(
            "feature_permutation"
        ),
        n_permutations=3,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        base_random_state=2026,
        observed_base_random_state=42,
    )

    print(
        "PASS: end-to-end permutation analysis"
    )

    # =========================================================
    # 10. Result structure
    # =========================================================

    assert (
        result.permutation_mode
        == "feature_permutation"
    )

    assert result.n_permutations == 3

    assert (
        result.summary[
            "patients"
        ]
        == len(X)
    )

    assert (
        result.summary[
            "biological_features"
        ]
        == X.shape[1]
    )

    assert (
        result.summary[
            "n_permutations"
        ]
        == 3
    )

    assert (
        result.summary[
            "n_repeats_per_dataset"
        ]
        == 2
    )

    assert (
        result.summary[
            "n_splits"
        ]
        == 5
    )

    print(
        "PASS: permutation result structure"
    )

    # =========================================================
    # 11. Null-statistic matrix integrity
    # =========================================================

    null_statistics = (
        result.null_statistics
    )

    assert len(
        null_statistics
    ) == 3

    assert (
        "permutation_seed"
        in null_statistics.columns
    )

    for statistic in (
        expected_statistics
    ):

        assert (
            statistic
            in null_statistics.columns
        )

        assert not (
            null_statistics[
                statistic
            ].isna().any()
        )

    print(
        "PASS: null-statistic matrix integrity"
    )

    # =========================================================
    # 12. Observed-statistic integrity
    # =========================================================

    assert set(
        result.observed_statistics.index
    ) == expected_statistics

    assert np.isfinite(
        result.observed_statistics.to_numpy(
            dtype=float
        )
    ).all()

    print(
        "PASS: observed-statistic integrity"
    )

    # =========================================================
    # 13. Empirical p-value bounds
    # =========================================================

    assert len(
        result.empirical_p_values
    ) == 8

    assert (
        result.empirical_p_values
        > 0.0
    ).all()

    assert (
        result.empirical_p_values
        <= 1.0
    ).all()

    # With B=3 permutations and +1 correction,
    # the smallest possible p-value is 1/4.
    assert (
        result.empirical_p_values
        >= 0.25
    ).all()

    print(
        "PASS: empirical p-value bounds"
    )

    # =========================================================
    # 14. Verify p-value arithmetic against null table
    # =========================================================

    for statistic in (
        result.empirical_p_values.index
    ):

        observed = float(
            result.observed_statistics[
                statistic
            ]
        )

        null_values = (
            result.null_statistics[
                statistic
            ]
        )

        exceedances = int(
            (
                null_values
                >= observed
            ).sum()
        )

        expected_p = (
            exceedances + 1
        ) / (
            len(null_values) + 1
        )

        assert_close(
            float(
                result.empirical_p_values[
                    statistic
                ]
            ),
            float(
                expected_p
            ),
        )

    print(
        "PASS: empirical p-value arithmetic"
    )

    # =========================================================
    # 15. Comparison table
    # =========================================================

    comparison = (
        permutation_comparison_table(
            result
        )
    )

    assert len(
        comparison
    ) == 8

    required_comparison_columns = {
        "observed",
        "null_mean",
        "null_sd",
        "null_95th_percentile",
        "empirical_p_value",
    }

    assert set(
        comparison.columns
    ) == required_comparison_columns

    for statistic in (
        comparison.index
    ):

        assert_close(
            comparison.loc[
                statistic,
                "observed",
            ],
            result.observed_statistics[
                statistic
            ],
        )

        assert_close(
            comparison.loc[
                statistic,
                "empirical_p_value",
            ],
            result.empirical_p_values[
                statistic
            ],
        )

    print(
        "PASS: observed/null comparison table"
    )

    # =========================================================
    # 16. Full experiment determinism
    #
    # Re-run the same very small experiment.
    # =========================================================

    result_repeat = run_permutation_test(
        X=X,
        treatment=T,
        outcome=Y,
        permutation_mode=(
            "feature_permutation"
        ),
        n_permutations=3,
        n_repeats=2,
        n_splits=5,
        C=0.1,
        base_random_state=2026,
        observed_base_random_state=42,
    )

    pd.testing.assert_series_equal(
        result.observed_statistics,
        result_repeat.observed_statistics,
    )

    pd.testing.assert_frame_equal(
        result.null_statistics,
        result_repeat.null_statistics,
    )

    pd.testing.assert_series_equal(
        result.empirical_p_values,
        result_repeat.empirical_p_values,
    )

    print(
        "PASS: deterministic permutation experiment"
    )

    # =========================================================
    # 17. Treatment-permutation mode smoke test
    #
    # We do not rerun the expensive full experiment here.
    # The treatment-permutation transformation itself has already
    # been validated above.
    # =========================================================

    treatment_mode = _validate_mode(
        "treatment_permutation"
    )

    assert (
        treatment_mode
        == "treatment_permutation"
    )

    print(
        "PASS: treatment-permutation mode available"
    )

    # =========================================================
    # Final
    # =========================================================

    print()

    print(
        "==============================================="
    )

    print(
        "ALL PERMUTATION / NULL VALIDATION TESTS PASSED"
    )

    print(
        "==============================================="
    )

    print()

    print(
        f"Patients: {len(X)}"
    )

    print(
        "Development permutations: "
        f"{result.n_permutations}"
    )

    print(
        "Repeats per dataset: "
        f"{result.summary['n_repeats_per_dataset']}"
    )

    print()

    print(
        "Observed ITE SD: "
        f"{result.observed_statistics['ite_sd_across_patients']:.4f}"
    )

    print(
        "Null mean ITE SD: "
        f"{result.null_statistics['ite_sd_across_patients'].mean():.4f}"
    )

    print()

    print(
        "NOTE: These small permutation counts are "
        "for software validation only."
    )


if __name__ == "__main__":
    run_tests()