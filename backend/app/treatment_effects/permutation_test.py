"""
HERMES 2.0
Permutation / Null Validation for Treatment-Effect Heterogeneity
================================================================

Purpose
-------
Challenge the treatment-effect heterogeneity learned by HERMES.

The module implements two complementary null perturbations.

1. treatment_permutation
   ---------------------
   Randomly permutes treatment assignment while preserving the number
   of treated and control patients.

   This destroys:
       - the average treatment effect
       - biology x treatment relationships

   It is therefore a strong sharp-null stress test.

2. feature_permutation
   -------------------
   Randomly reassigns complete biological feature profiles across
   patients while leaving treatment and outcome labels unchanged.

   This preserves:
       - the observed randomized treatment/outcome relationship
       - treatment counts
       - outcome counts
       - correlation structure among biological features

   But it destroys:
       - correspondence between a patient's biology and treatment/outcome

   This is particularly useful as a stress test for whether HERMES is
   manufacturing apparent biology-dependent treatment heterogeneity.

IMPORTANT
---------
These permutation procedures are validation / falsification experiments.

They should not yet be interpreted as the final formal inferential test for
treatment-effect heterogeneity in a publication. Formal HTE inference may
require additional treatment of the population-average treatment effect as a
nuisance parameter and/or randomization-based inference designed specifically
for heterogeneous effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from backend.app.treatment_effects.repeated_crossfit import (
    RepeatedCrossFitResult,
    generate_random_states,
    repeated_crossfit_treatment_effect_model,
)


PermutationMode = Literal[
    "feature_permutation",
    "treatment_permutation",
]


@dataclass
class PermutationTestResult:
    """
    Results from HERMES null/permutation validation.
    """

    observed_result: RepeatedCrossFitResult

    null_statistics: pd.DataFrame

    observed_statistics: pd.Series

    empirical_p_values: pd.Series

    summary: dict[str, Any]

    permutation_mode: str

    @property
    def n_permutations(self) -> int:
        return int(
            self.null_statistics.shape[0]
        )


def _validate_mode(
    mode: str,
) -> PermutationMode:
    """
    Validate permutation mode.
    """

    allowed = {
        "feature_permutation",
        "treatment_permutation",
    }

    if mode not in allowed:
        raise ValueError(
            "permutation_mode must be one of "
            f"{sorted(allowed)}. "
            f"Found: {mode}"
        )

    return mode  # type: ignore[return-value]


def permute_treatment(
    treatment: pd.Series,
    *,
    random_state: int,
) -> pd.Series:
    """
    Permute treatment labels across patients.

    The number of treated and control patients is preserved exactly.
    """

    rng = np.random.default_rng(
        random_state
    )

    permuted_values = rng.permutation(
        treatment.to_numpy()
    )

    return pd.Series(
        permuted_values,
        index=treatment.index,
        name=treatment.name,
        dtype=treatment.dtype,
    )


def permute_feature_profiles(
    X: pd.DataFrame,
    *,
    random_state: int,
) -> pd.DataFrame:
    """
    Permute complete biological profiles across patients.

    Rows are permuted as intact vectors rather than gene/pathway columns
    independently.

    Therefore the covariance structure among biological features is
    preserved while patient-to-biology correspondence is destroyed.
    """

    rng = np.random.default_rng(
        random_state
    )

    permutation = rng.permutation(
        len(X)
    )

    values = X.to_numpy()[
        permutation,
        :
    ]

    return pd.DataFrame(
        values,
        index=X.index,
        columns=X.columns,
    )


def _extract_heterogeneity_statistics(
    result: RepeatedCrossFitResult,
) -> pd.Series:
    """
    Extract cohort-level statistics describing treatment-effect
    heterogeneity and stability.

    Statistics are deliberately chosen before looking at permutation
    results.

    Larger values generally imply stronger apparent heterogeneity or
    stability.
    """

    patient_summary = (
        result.patient_summary
    )

    mean_ite = (
        patient_summary[
            "mean_ite"
        ]
    )

    absolute_mean_ite = (
        mean_ite.abs()
    )

    statistics = pd.Series(
        {
            # Dispersion of patient-level repeated-CF effects
            "ite_sd_across_patients":
                float(
                    mean_ite.std()
                ),

            # Mean absolute deviation from cohort mean effect
            "ite_mean_absolute_deviation":
                float(
                    (
                        mean_ite
                        - mean_ite.mean()
                    )
                    .abs()
                    .mean()
                ),

            # Interquartile treatment-effect spread
            "ite_iqr":
                float(
                    mean_ite.quantile(
                        0.75
                    )
                    - mean_ite.quantile(
                        0.25
                    )
                ),

            # Upper-tail individualized effect magnitude
            "ite_absolute_90th_percentile":
                float(
                    absolute_mean_ite.quantile(
                        0.90
                    )
                ),

            # Maximum individualized effect magnitude
            "ite_max_absolute":
                float(
                    absolute_mean_ite.max()
                ),

            # Fraction whose direction is highly stable
            "fraction_sign_stability_ge_90pct":
                float(
                    (
                        patient_summary[
                            "sign_stability"
                        ]
                        >= 0.90
                    ).mean()
                ),

            # Fraction whose direction never changes
            "fraction_unanimous_sign":
                float(
                    (
                        patient_summary[
                            "sign_stability"
                        ]
                        == 1.0
                    ).mean()
                ),

            # Typical signal relative to split instability
            "median_stability_signal_ratio":
                float(
                    patient_summary[
                        "stability_signal_ratio"
                    ]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                    .median()
                ),

            # Cohort-average effect retained for context,
            # but NOT our primary heterogeneity statistic.
            "cohort_mean_ite":
                float(
                    mean_ite.mean()
                ),
        },
        dtype=float,
    )

    return statistics


def _empirical_upper_tail_p_value(
    observed: float,
    null_values: pd.Series,
) -> float:
    """
    Monte Carlo upper-tail empirical p-value.

    Uses the standard +1 correction:

        p = (1 + number(null >= observed)) / (B + 1)

    This prevents p=0 under finite Monte Carlo sampling.
    """

    null_values = (
        null_values
        .dropna()
        .astype(float)
    )

    if len(null_values) == 0:
        return float("nan")

    exceedances = int(
        (
            null_values
            >= observed
        ).sum()
    )

    return float(
        (
            exceedances + 1
        )
        / (
            len(null_values) + 1
        )
    )


def _build_empirical_p_values(
    observed_statistics: pd.Series,
    null_statistics: pd.DataFrame,
) -> pd.Series:
    """
    Compare each observed heterogeneity statistic against its
    permutation null distribution.

    Cohort mean ITE is excluded because the two permutation modes
    target different null hypotheses for the overall treatment effect.
    """

    test_statistics = [
        "ite_sd_across_patients",
        "ite_mean_absolute_deviation",
        "ite_iqr",
        "ite_absolute_90th_percentile",
        "ite_max_absolute",
        "fraction_sign_stability_ge_90pct",
        "fraction_unanimous_sign",
        "median_stability_signal_ratio",
    ]

    p_values = {}

    for statistic in test_statistics:

        p_values[
            statistic
        ] = (
            _empirical_upper_tail_p_value(
                float(
                    observed_statistics[
                        statistic
                    ]
                ),
                null_statistics[
                    statistic
                ],
            )
        )

    return pd.Series(
        p_values,
        name="empirical_p_value",
        dtype=float,
    )


def run_permutation_test(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    permutation_mode: PermutationMode = "feature_permutation",
    n_permutations: int = 100,
    n_repeats: int = 10,
    n_splits: int = 5,
    C: float = 0.1,
    max_iter: int = 10000,
    base_random_state: int = 2026,
    observed_base_random_state: int = 42,
) -> PermutationTestResult:
    """
    Run HERMES treatment-effect heterogeneity null validation.

    Parameters
    ----------
    X
        Biological feature matrix.

    treatment
        Binary randomized treatment.

    outcome
        Binary pCR outcome.

    permutation_mode
        "feature_permutation"
            Preserve observed treatment/outcome relationship while
            destroying patient-specific biology correspondence.

        "treatment_permutation"
            Destroy treatment/outcome and biology/treatment relationships.

    n_permutations
        Number of null datasets.

        For development, values such as 20-100 are useful.

        For final publication-quality Monte Carlo inference, this should
        later be increased substantially.

    n_repeats
        Number of cross-fitting repetitions within each observed/null run.

    n_splits
        Number of folds per repeat.

    C
        Logistic-model inverse regularization strength.

    base_random_state
        Generates permutation-specific random seeds.

    observed_base_random_state
        Controls repeated cross-fitting for the unpermuted observed data.
    """

    mode = _validate_mode(
        permutation_mode
    )

    if n_permutations < 1:
        raise ValueError(
            "n_permutations must be at least 1."
        )

    if n_repeats < 2:
        raise ValueError(
            "n_repeats must be at least 2."
        )

    # ---------------------------------------------------------
    # Fit real / observed data
    # ---------------------------------------------------------

    observed_result = (
        repeated_crossfit_treatment_effect_model(
            X=X,
            treatment=treatment,
            outcome=outcome,
            n_repeats=n_repeats,
            n_splits=n_splits,
            C=C,
            max_iter=max_iter,
            base_random_state=(
                observed_base_random_state
            ),
        )
    )

    observed_statistics = (
        _extract_heterogeneity_statistics(
            observed_result
        )
    )

    # ---------------------------------------------------------
    # Deterministic null seeds
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        base_random_state
    )

    permutation_seeds = generate_random_states(
        n_permutations,
        base_random_state=base_random_state,
    )

    records: list[
        dict[str, float | int]
    ] = []

    # ---------------------------------------------------------
    # Null experiments
    # ---------------------------------------------------------

    for permutation_number, seed_value in enumerate(
        permutation_seeds,
        start=1,
    ):

        permutation_seed = int(
            seed_value
        )

        if mode == "feature_permutation":

            X_null = (
                permute_feature_profiles(
                    X,
                    random_state=(
                        permutation_seed
                    ),
                )
            )

            treatment_null = (
                treatment.copy()
            )

        else:

            X_null = X.copy()

            treatment_null = (
                permute_treatment(
                    treatment,
                    random_state=(
                        permutation_seed
                    ),
                )
            )

        # Give the model-fitting resampling its own deterministic
        # seed, distinct from the permutation seed.
        model_seed = int(
            (
                permutation_seed
                + 7919
            )
            % 2_147_483_647
        )

        if model_seed == 0:
            model_seed = 1

        null_result = (
            repeated_crossfit_treatment_effect_model(
                X=X_null,
                treatment=treatment_null,
                outcome=outcome,
                n_repeats=n_repeats,
                n_splits=n_splits,
                C=C,
                max_iter=max_iter,
                base_random_state=(
                    model_seed
                ),
            )
        )

        null_stats = (
            _extract_heterogeneity_statistics(
                null_result
            )
        )

        record: dict[
            str,
            float | int
        ] = {
            "permutation":
                permutation_number,
            "permutation_seed":
                permutation_seed,
        }

        for key, value in (
            null_stats.items()
        ):

            record[
                str(key)
            ] = float(
                value
            )

        records.append(
            record
        )

        print(
            f"Permutation "
            f"{permutation_number}/"
            f"{n_permutations} complete"
        )

    # ---------------------------------------------------------
    # Assemble null distribution
    # ---------------------------------------------------------

    null_statistics = (
        pd.DataFrame(
            records
        )
        .set_index(
            "permutation"
        )
    )

    # ---------------------------------------------------------
    # Empirical comparisons
    # ---------------------------------------------------------

    empirical_p_values = (
        _build_empirical_p_values(
            observed_statistics,
            null_statistics,
        )
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary: dict[
        str,
        Any,
    ] = {
        "permutation_mode":
            mode,
        "patients":
            int(
                len(X)
            ),
        "biological_features":
            int(
                X.shape[1]
            ),
        "n_permutations":
            int(
                n_permutations
            ),
        "n_repeats_per_dataset":
            int(
                n_repeats
            ),
        "n_splits":
            int(
                n_splits
            ),
        "regularization_C":
            float(
                C
            ),
        "observed_ite_sd_across_patients":
            float(
                observed_statistics[
                    "ite_sd_across_patients"
                ]
            ),
        "null_mean_ite_sd_across_patients":
            float(
                null_statistics[
                    "ite_sd_across_patients"
                ].mean()
            ),
        "observed_ite_iqr":
            float(
                observed_statistics[
                    "ite_iqr"
                ]
            ),
        "null_mean_ite_iqr":
            float(
                null_statistics[
                    "ite_iqr"
                ].mean()
            ),
        "observed_fraction_unanimous_sign":
            float(
                observed_statistics[
                    "fraction_unanimous_sign"
                ]
            ),
        "null_mean_fraction_unanimous_sign":
            float(
                null_statistics[
                    "fraction_unanimous_sign"
                ].mean()
            ),
    }

    return PermutationTestResult(
        observed_result=(
            observed_result
        ),
        null_statistics=(
            null_statistics
        ),
        observed_statistics=(
            observed_statistics
        ),
        empirical_p_values=(
            empirical_p_values
        ),
        summary=summary,
        permutation_mode=mode,
    )


def permutation_comparison_table(
    result: PermutationTestResult,
) -> pd.DataFrame:
    """
    Create publication-friendly observed-vs-null summary.
    """

    rows = []

    for statistic in (
        result.empirical_p_values.index
    ):

        null_values = (
            result.null_statistics[
                statistic
            ]
        )

        rows.append(
            {
                "statistic":
                    statistic,

                "observed":
                    float(
                        result.observed_statistics[
                            statistic
                        ]
                    ),

                "null_mean":
                    float(
                        null_values.mean()
                    ),

                "null_sd":
                    float(
                        null_values.std()
                    ),

                "null_95th_percentile":
                    float(
                        null_values.quantile(
                            0.95
                        )
                    ),

                "empirical_p_value":
                    float(
                        result.empirical_p_values[
                            statistic
                        ]
                    ),
            }
        )

    return (
        pd.DataFrame(
            rows
        )
        .set_index(
            "statistic"
        )
    )


def main() -> None:
    """
    Run a DEVELOPMENT-SCALE HERMES permutation experiment.

    We intentionally use only 20 permutations and 5 repeated cross-fit
    runs here so the pipeline can be validated without fitting thousands
    of models.

    The final scientific analysis will use substantially more permutations
    after the implementation is tested.
    """

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    dataset = (
        build_treatment_effect_dataset()
    )

    result = run_permutation_test(
        X=dataset.X,
        treatment=dataset.T,
        outcome=dataset.Y,
        permutation_mode=(
            "feature_permutation"
        ),
        n_permutations=20,
        n_repeats=5,
        n_splits=5,
        C=0.1,
        base_random_state=2026,
        observed_base_random_state=42,
    )

    print()

    print(
        "=== HERMES 2.0 "
        "PERMUTATION / NULL VALIDATION ==="
    )

    print()

    for key, value in (
        result.summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Observed heterogeneity statistics:"
    )

    print(
        result.observed_statistics
        .to_string()
    )

    print()

    print(
        "Observed vs null comparison:"
    )

    print(
        permutation_comparison_table(
            result
        )
        .to_string()
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is a development-scale "
        "permutation analysis."
    )

    print(
        "Do not interpret Monte Carlo "
        "p-values from only 20 permutations "
        "as final inferential results."
    )


if __name__ == "__main__":
    main()