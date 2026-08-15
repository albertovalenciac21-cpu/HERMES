"""
HERMES 2.0
Patient-Level Treatment-Effect Uncertainty
==========================================

Purpose
-------
Quantify uncertainty around individualized treatment-effect estimates produced
by repeated cross-fitting.

The repeated-cross-fit pipeline generates multiple ITE estimates for each
patient across independent repeated cross-fitting runs. This module converts
those estimates into patient-level uncertainty summaries.

For each patient we estimate:

    - mean ITE
    - median ITE
    - standard deviation
    - empirical lower/upper interval
    - interval width
    - fraction of estimates > 0
    - fraction of estimates < 0
    - sign stability
    - signal-to-uncertainty ratio
    - qualitative evidence state

The qualitative evidence states are research-engineering categories only:

    likely_benefit
    likely_harm
    indeterminate

They are NOT clinical treatment recommendations or validated clinical cutoffs.

The module also supports validation against known true ITE values in synthetic
positive-control experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TreatmentEffectUncertaintyResult:
    """
    Patient-level uncertainty summaries for repeated HERMES ITE estimates.
    """

    patient_table: pd.DataFrame
    summary: dict[str, Any]


def _validate_alpha(
    alpha: float,
) -> float:
    """
    Validate empirical interval alpha.
    """

    alpha = float(
        alpha
    )

    if not (
        0.0
        < alpha
        < 1.0
    ):
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    return alpha


def _extract_ite_matrix(
    repeated_result: Any,
) -> pd.DataFrame:
    """
    Extract the patient × repeat ITE matrix from RepeatedCrossFitResult.

    Current validated HERMES structure:

        repeated_result.ite_by_repeat

    Expected shape:

        n_patients × n_repeats
    """

    if not hasattr(
        repeated_result,
        "ite_by_repeat",
    ):
        raise AttributeError(
            "RepeatedCrossFitResult does not contain 'ite_by_repeat'."
        )

    ite_matrix = (
        repeated_result
        .ite_by_repeat
    )

    if not isinstance(
        ite_matrix,
        pd.DataFrame,
    ):
        raise TypeError(
            "RepeatedCrossFitResult.ite_by_repeat must be a pandas DataFrame."
        )

    if ite_matrix.empty:
        raise ValueError(
            "RepeatedCrossFitResult.ite_by_repeat cannot be empty."
        )

    if ite_matrix.shape[1] < 2:
        raise ValueError(
            "At least two repeated ITE estimates are required "
            "for uncertainty estimation."
        )

    matrix = ite_matrix.copy()

    matrix = matrix.astype(
        float
    )

    if not np.isfinite(
        matrix.to_numpy()
    ).all():
        raise ValueError(
            "RepeatedCrossFitResult.ite_by_repeat contains non-finite values."
        )

    return matrix


def build_uncertainty_table(
    ite_matrix: pd.DataFrame,
    *,
    alpha: float = 0.05,
    minimum_sign_stability: float = 0.90,
    minimum_signal_uncertainty_ratio: float = 1.0,
) -> pd.DataFrame:
    """
    Convert repeated patient-level ITE estimates into uncertainty summaries.

    The empirical interval is formed from quantiles across repeated
    cross-fitting estimates.

    IMPORTANT
    ---------
    These are empirical resampling intervals, not formal causal confidence
    intervals.
    """

    alpha = _validate_alpha(
        alpha
    )

    if ite_matrix.empty:
        raise ValueError(
            "ite_matrix cannot be empty."
        )

    if ite_matrix.shape[1] < 2:
        raise ValueError(
            "ite_matrix must contain at least two repeats."
        )

    if not (
        0.5
        <= minimum_sign_stability
        <= 1.0
    ):
        raise ValueError(
            "minimum_sign_stability must be between 0.5 and 1.0."
        )

    if (
        minimum_signal_uncertainty_ratio
        < 0
    ):
        raise ValueError(
            "minimum_signal_uncertainty_ratio must be non-negative."
        )

    values = ite_matrix.astype(
        float
    )

    if not np.isfinite(
        values.to_numpy()
    ).all():
        raise ValueError(
            "ite_matrix contains non-finite values."
        )

    lower_q = (
        alpha
        / 2.0
    )

    upper_q = (
        1.0
        - alpha / 2.0
    )

    table = pd.DataFrame(
        index=values.index
    )

    table[
        "mean_ite"
    ] = values.mean(
        axis=1
    )

    table[
        "median_ite"
    ] = values.median(
        axis=1
    )

    table[
        "ite_std"
    ] = values.std(
        axis=1,
        ddof=1,
    )

    table[
        "ite_lower"
    ] = values.quantile(
        lower_q,
        axis=1,
    )

    table[
        "ite_upper"
    ] = values.quantile(
        upper_q,
        axis=1,
    )

    table[
        "interval_width"
    ] = (
        table[
            "ite_upper"
        ]
        - table[
            "ite_lower"
        ]
    )

    table[
        "fraction_positive"
    ] = (
        values
        > 0.0
    ).mean(
        axis=1
    )

    table[
        "fraction_negative"
    ] = (
        values
        < 0.0
    ).mean(
        axis=1
    )

    table[
        "fraction_zero"
    ] = (
        values
        == 0.0
    ).mean(
        axis=1
    )

    table[
        "sign_stability"
    ] = (
        table[
            [
                "fraction_positive",
                "fraction_negative",
            ]
        ]
        .max(
            axis=1
        )
    )

    safe_std = (
        table[
            "ite_std"
        ]
        .replace(
            0.0,
            np.nan,
        )
    )

    table[
        "signal_uncertainty_ratio"
    ] = (
        table[
            "mean_ite"
        ]
        .abs()
        / safe_std
    )

    # A perfectly identical repeated estimate has zero empirical variance.
    # If the mean is also zero, it carries no treatment-effect signal.
    # If the mean is nonzero, it is maximally stable under this repeated-fit
    # definition.
    zero_std = (
        table[
            "ite_std"
        ]
        == 0.0
    )

    nonzero_mean = (
        table[
            "mean_ite"
        ]
        != 0.0
    )

    table.loc[
        zero_std
        & nonzero_mean,
        "signal_uncertainty_ratio",
    ] = np.inf

    table.loc[
        zero_std
        & ~nonzero_mean,
        "signal_uncertainty_ratio",
    ] = 0.0

    table[
        "interval_excludes_zero"
    ] = (
        (
            table[
                "ite_lower"
            ]
            > 0.0
        )
        |
        (
            table[
                "ite_upper"
            ]
            < 0.0
        )
    )

    stable = (
        table[
            "sign_stability"
        ]
        >= minimum_sign_stability
    )

    adequate_signal = (
        table[
            "signal_uncertainty_ratio"
        ]
        >= minimum_signal_uncertainty_ratio
    )

    positive = (
        table[
            "mean_ite"
        ]
        > 0.0
    )

    negative = (
        table[
            "mean_ite"
        ]
        < 0.0
    )

    table[
        "evidence_state"
    ] = "indeterminate"

    table.loc[
        stable
        & adequate_signal
        & positive,
        "evidence_state",
    ] = "likely_benefit"

    table.loc[
        stable
        & adequate_signal
        & negative,
        "evidence_state",
    ] = "likely_harm"

    return table


def summarize_uncertainty_table(
    patient_table: pd.DataFrame,
) -> dict[str, Any]:
    """
    Cohort-level summary of treatment-effect uncertainty.
    """

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

    missing = (
        required_columns
        - set(
            patient_table.columns
        )
    )

    if missing:
        raise ValueError(
            "patient_table is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    counts = (
        patient_table[
            "evidence_state"
        ]
        .value_counts()
        .to_dict()
    )

    total = int(
        len(
            patient_table
        )
    )

    summary: dict[
        str,
        Any,
    ] = {
        "patients":
            total,

        "mean_ite":
            float(
                patient_table[
                    "mean_ite"
                ].mean()
            ),

        "median_ite":
            float(
                patient_table[
                    "mean_ite"
                ].median()
            ),

        "mean_ite_std":
            float(
                patient_table[
                    "ite_std"
                ].mean()
            ),

        "median_ite_std":
            float(
                patient_table[
                    "ite_std"
                ].median()
            ),

        "mean_interval_width":
            float(
                patient_table[
                    "interval_width"
                ].mean()
            ),

        "median_interval_width":
            float(
                patient_table[
                    "interval_width"
                ].median()
            ),

        "fraction_interval_excludes_zero":
            float(
                patient_table[
                    "interval_excludes_zero"
                ].mean()
            ),

        "fraction_sign_stability_ge_90pct":
            float(
                (
                    patient_table[
                        "sign_stability"
                    ]
                    >= 0.90
                ).mean()
            ),

        "likely_benefit":
            int(
                counts.get(
                    "likely_benefit",
                    0,
                )
            ),

        "likely_harm":
            int(
                counts.get(
                    "likely_harm",
                    0,
                )
            ),

        "indeterminate":
            int(
                counts.get(
                    "indeterminate",
                    0,
                )
            ),
    }

    if total > 0:

        summary[
            "fraction_likely_benefit"
        ] = float(
            summary[
                "likely_benefit"
            ]
            / total
        )

        summary[
            "fraction_likely_harm"
        ] = float(
            summary[
                "likely_harm"
            ]
            / total
        )

        summary[
            "fraction_indeterminate"
        ] = float(
            summary[
                "indeterminate"
            ]
            / total
        )

    return summary


def quantify_treatment_effect_uncertainty(
    repeated_result: Any,
    *,
    alpha: float = 0.05,
    minimum_sign_stability: float = 0.90,
    minimum_signal_uncertainty_ratio: float = 1.0,
) -> TreatmentEffectUncertaintyResult:
    """
    Main uncertainty interface for RepeatedCrossFitResult.
    """

    ite_matrix = _extract_ite_matrix(
        repeated_result
    )

    patient_table = build_uncertainty_table(
        ite_matrix,
        alpha=alpha,
        minimum_sign_stability=(
            minimum_sign_stability
        ),
        minimum_signal_uncertainty_ratio=(
            minimum_signal_uncertainty_ratio
        ),
    )

    summary = summarize_uncertainty_table(
        patient_table
    )

    summary[
        "alpha"
    ] = float(
        alpha
    )

    summary[
        "minimum_sign_stability"
    ] = float(
        minimum_sign_stability
    )

    summary[
        "minimum_signal_uncertainty_ratio"
    ] = float(
        minimum_signal_uncertainty_ratio
    )

    summary[
        "n_repeated_estimates"
    ] = int(
        ite_matrix.shape[1]
    )

    return TreatmentEffectUncertaintyResult(
        patient_table=(
            patient_table
        ),
        summary=(
            summary
        ),
    )


def validate_uncertainty_against_truth(
    patient_table: pd.DataFrame,
    true_ite: pd.Series,
    *,
    stable_threshold: float = 0.90,
) -> dict[str, float]:
    """
    Validate uncertainty summaries against known true ITE values.

    Intended for synthetic experiments where individual causal effects are
    known from the data-generating mechanism.

    IMPORTANT
    ---------
    Empirical repeated-fit quantiles are not guaranteed frequentist confidence
    intervals. Coverage is therefore evaluated diagnostically rather than
    assumed to equal 1 - alpha.
    """

    if not (
        0.5
        <= stable_threshold
        <= 1.0
    ):
        raise ValueError(
            "stable_threshold must be between 0.5 and 1.0."
        )

    if not patient_table.index.equals(
        true_ite.index
    ):
        raise ValueError(
            "patient_table and true_ite must have identical patient order."
        )

    truth = true_ite.astype(
        float
    )

    if not np.isfinite(
        truth.to_numpy()
    ).all():
        raise ValueError(
            "true_ite contains non-finite values."
        )

    lower = (
        patient_table[
            "ite_lower"
        ]
    )

    upper = (
        patient_table[
            "ite_upper"
        ]
    )

    covered = (
        (
            truth
            >= lower
        )
        &
        (
            truth
            <= upper
        )
    )

    error = (
        patient_table[
            "mean_ite"
        ]
        - truth
    )

    absolute_error = (
        error.abs()
    )

    uncertainty = (
        patient_table[
            "ite_std"
        ]
    )

    if (
        uncertainty.std()
        > 0
        and absolute_error.std()
        > 0
    ):

        error_uncertainty_correlation = float(
            np.corrcoef(
                absolute_error.to_numpy(
                    dtype=float
                ),
                uncertainty.to_numpy(
                    dtype=float
                ),
            )[0, 1]
        )

    else:

        error_uncertainty_correlation = float(
            "nan"
        )

    stable = (
        patient_table[
            "sign_stability"
        ]
        >= stable_threshold
    )

    true_sign = np.sign(
        truth
    )

    estimated_sign = np.sign(
        patient_table[
            "mean_ite"
        ]
    )

    overall_sign_accuracy = float(
        (
            true_sign
            == estimated_sign
        ).mean()
    )

    if stable.any():

        stable_sign_accuracy = float(
            (
                true_sign[
                    stable
                ]
                ==
                estimated_sign[
                    stable
                ]
            ).mean()
        )

        stable_mean_absolute_error = float(
            absolute_error[
                stable
            ].mean()
        )

    else:

        stable_sign_accuracy = float(
            "nan"
        )

        stable_mean_absolute_error = float(
            "nan"
        )

    unstable = (
        ~stable
    )

    if unstable.any():

        unstable_sign_accuracy = float(
            (
                true_sign[
                    unstable
                ]
                ==
                estimated_sign[
                    unstable
                ]
            ).mean()
        )

        unstable_mean_absolute_error = float(
            absolute_error[
                unstable
            ].mean()
        )

    else:

        unstable_sign_accuracy = float(
            "nan"
        )

        unstable_mean_absolute_error = float(
            "nan"
        )

    return {
        "empirical_interval_coverage":
            float(
                covered.mean()
            ),

        "mean_absolute_ite_error":
            float(
                absolute_error.mean()
            ),

        "median_absolute_ite_error":
            float(
                absolute_error.median()
            ),

        "root_mean_squared_ite_error":
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            error.to_numpy(
                                dtype=float
                            )
                        )
                    )
                )
            ),

        "error_uncertainty_correlation":
            error_uncertainty_correlation,

        "overall_sign_accuracy":
            overall_sign_accuracy,

        "stable_patient_sign_accuracy":
            stable_sign_accuracy,

        "unstable_patient_sign_accuracy":
            unstable_sign_accuracy,

        "stable_patient_mean_absolute_error":
            stable_mean_absolute_error,

        "unstable_patient_mean_absolute_error":
            unstable_mean_absolute_error,

        "fraction_stable_patients":
            float(
                stable.mean()
            ),
    }


def main() -> None:
    """
    Development demonstration using the synthetic HERMES positive control.
    """

    from backend.app.treatment_effects.positive_control import (
        run_positive_control,
    )

    print(
        "=== HERMES 2.0 TREATMENT-EFFECT UNCERTAINTY ==="
    )

    print()

    positive_control = run_positive_control(
        n_patients=500,
        n_features=20,
        treatment_interaction=1.5,
        n_repeats=20,
        n_splits=5,
        C=0.1,
        data_random_state=2026,
        model_random_state=42,
    )

    print(
        "Repeated ITE matrix: "
        f"{positive_control.hermes_result.ite_by_repeat.shape}"
    )

    uncertainty = (
        quantify_treatment_effect_uncertainty(
            positive_control.hermes_result,
            alpha=0.05,
            minimum_sign_stability=0.90,
            minimum_signal_uncertainty_ratio=1.0,
        )
    )

    truth_validation = (
        validate_uncertainty_against_truth(
            uncertainty.patient_table,
            positive_control.dataset.true_ite,
            stable_threshold=0.90,
        )
    )

    print()

    print(
        "Cohort uncertainty summary:"
    )

    for key, value in (
        uncertainty.summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Synthetic truth validation:"
    )

    for key, value in (
        truth_validation.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Evidence-state counts:"
    )

    print(
        uncertainty
        .patient_table[
            "evidence_state"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Most confident predicted benefit:"
    )

    benefit = (
        uncertainty
        .patient_table[
            uncertainty
            .patient_table[
                "evidence_state"
            ]
            == "likely_benefit"
        ]
        .sort_values(
            [
                "sign_stability",
                "signal_uncertainty_ratio",
                "mean_ite",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
    )

    print(
        benefit
        .head(10)
        .to_string()
    )

    print()

    print(
        "Most uncertain patients:"
    )

    uncertain = (
        uncertainty
        .patient_table
        .sort_values(
            [
                "sign_stability",
                "interval_width",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print(
        uncertain
        .head(10)
        .to_string()
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Repeated-fit empirical intervals quantify model/resampling "
        "stability and should not be interpreted as formal causal "
        "confidence intervals."
    )

    print(
        "Evidence states are research outputs, not clinical "
        "treatment recommendations."
    )


if __name__ == "__main__":
    main()