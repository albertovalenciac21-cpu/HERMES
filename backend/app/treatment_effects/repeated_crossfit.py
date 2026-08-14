"""
HERMES 2.0
Repeated Cross-Fitted Treatment-Effect Estimation
=================================================

Purpose
-------
Quantify the stability of patient-level treatment-effect estimates across
multiple independently randomized cross-fitting partitions.

A single cross-fitting run produces:

    tau_i^(r)
        = P(Y=1 | T=1, X_i; model not trained on patient i)
        - P(Y=1 | T=0, X_i; model not trained on patient i)

Repeated cross-fitting generates:

    tau_i^(1), tau_i^(2), ..., tau_i^(R)

for every patient.

We then summarize each patient's treatment-effect distribution using:

    mean ITE
    median ITE
    standard deviation
    minimum
    maximum
    sign stability
    fraction of repeats with positive ITE

This allows HERMES to distinguish:

    large + stable predicted treatment benefit

from:

    large but unstable treatment-effect estimates.

IMPORTANT
---------
Repeated cross-fitting measures resampling stability.

It does NOT by itself establish:
    - causal validity of heterogeneous treatment effects
    - statistical significance
    - clinical utility
    - external generalizability

Those require additional validation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from backend.app.treatment_effects.crossfit import (
    CrossFitTreatmentEffectResult,
    crossfit_treatment_effect_model,
)


@dataclass
class RepeatedCrossFitResult:
    """
    Results from repeated cross-fitted treatment-effect estimation.
    """

    ite_by_repeat: pd.DataFrame

    observed_probability_by_repeat: pd.DataFrame
    probability_control_by_repeat: pd.DataFrame
    probability_treated_by_repeat: pd.DataFrame

    patient_summary: pd.DataFrame
    repeat_summary: pd.DataFrame

    summary: dict[str, Any]

    random_states: tuple[int, ...]

    @property
    def n_patients(self) -> int:
        return int(
            self.ite_by_repeat.shape[0]
        )

    @property
    def n_repeats(self) -> int:
        return int(
            self.ite_by_repeat.shape[1]
        )


def generate_random_states(
    n_repeats: int,
    *,
    base_random_state: int = 42,
) -> tuple[int, ...]:
    """
    Generate deterministic but distinct random states.

    The same base_random_state and n_repeats always produce
    the same sequence.
    """

    if n_repeats < 2:
        raise ValueError(
            "n_repeats must be at least 2."
        )

    rng = np.random.default_rng(
        base_random_state
    )

    states = rng.choice(
        np.arange(
            1,
            2_147_483_647,
            dtype=np.int64,
        ),
        size=n_repeats,
        replace=False,
    )

    return tuple(
        int(value)
        for value in states
    )


def validate_random_states(
    random_states: Sequence[int],
) -> tuple[int, ...]:
    """
    Validate explicitly supplied repeat seeds.
    """

    states = tuple(
        int(value)
        for value in random_states
    )

    if len(states) < 2:
        raise ValueError(
            "At least two random states are required."
        )

    if len(set(states)) != len(states):
        raise ValueError(
            "Random states must be unique."
        )

    if any(
        value < 0
        for value in states
    ):
        raise ValueError(
            "Random states must be non-negative."
        )

    return states


def _summarize_patient_ite(
    ite_by_repeat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute patient-level treatment-effect stability statistics.
    """

    summary = pd.DataFrame(
        index=ite_by_repeat.index
    )

    summary.index.name = (
        ite_by_repeat.index.name
    )

    summary[
        "mean_ite"
    ] = ite_by_repeat.mean(
        axis=1
    )

    summary[
        "median_ite"
    ] = ite_by_repeat.median(
        axis=1
    )

    summary[
        "ite_std"
    ] = ite_by_repeat.std(
        axis=1,
        ddof=1,
    )

    summary[
        "minimum_ite"
    ] = ite_by_repeat.min(
        axis=1
    )

    summary[
        "maximum_ite"
    ] = ite_by_repeat.max(
        axis=1
    )

    summary[
        "ite_range"
    ] = (
        summary["maximum_ite"]
        - summary["minimum_ite"]
    )

    summary[
        "fraction_positive"
    ] = (
        ite_by_repeat > 0
    ).mean(
        axis=1
    )

    summary[
        "fraction_negative"
    ] = (
        ite_by_repeat < 0
    ).mean(
        axis=1
    )

    summary[
        "fraction_zero"
    ] = (
        ite_by_repeat == 0
    ).mean(
        axis=1
    )

    # ---------------------------------------------------------
    # Sign stability
    #
    # 1.0 means every repeat gives the same non-zero sign.
    # 0.5 means the sign is essentially split.
    # ---------------------------------------------------------

    summary[
        "sign_stability"
    ] = np.maximum(
        summary[
            "fraction_positive"
        ],
        summary[
            "fraction_negative"
        ],
    )

    summary[
        "consensus_direction"
    ] = np.select(
        [
            summary[
                "fraction_positive"
            ] > 0.5,
            summary[
                "fraction_negative"
            ] > 0.5,
        ],
        [
            "benefit",
            "harm",
        ],
        default="uncertain",
    )

    # ---------------------------------------------------------
    # Simple stability index
    #
    # Larger |mean ITE| relative to resampling SD means a more
    # stable treatment-effect estimate.
    #
    # This is descriptive, NOT a statistical test.
    # ---------------------------------------------------------

    denominator = (
        summary["ite_std"]
        .replace(
            0.0,
            np.nan,
        )
    )

    summary[
        "stability_signal_ratio"
    ] = (
        summary[
            "mean_ite"
        ].abs()
        / denominator
    )

    summary[
        "stability_signal_ratio"
    ] = (
        summary[
            "stability_signal_ratio"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(
            np.inf
        )
    )

    return summary


def _build_repeat_summary(
    results: Sequence[
        CrossFitTreatmentEffectResult
    ],
    random_states: Sequence[int],
) -> pd.DataFrame:
    """
    Construct one diagnostic row per repeated cross-fit run.
    """

    records: list[
        dict[str, Any]
    ] = []

    for repeat_number, (
        result,
        random_state,
    ) in enumerate(
        zip(
            results,
            random_states,
        ),
        start=1,
    ):

        records.append(
            {
                "repeat":
                    repeat_number,
                "random_state":
                    int(
                        random_state
                    ),
                "oof_auc":
                    float(
                        result.summary[
                            "crossfitted_observed_auc"
                        ]
                    ),
                "oof_brier":
                    float(
                        result.summary[
                            "crossfitted_observed_brier"
                        ]
                    ),
                "mean_ite":
                    float(
                        result.ite.mean()
                    ),
                "median_ite":
                    float(
                        result.ite.median()
                    ),
                "ite_std":
                    float(
                        result.ite.std()
                    ),
                "minimum_ite":
                    float(
                        result.ite.min()
                    ),
                "maximum_ite":
                    float(
                        result.ite.max()
                    ),
                "fraction_positive":
                    float(
                        (
                            result.ite
                            > 0
                        ).mean()
                    ),
                "fraction_negative":
                    float(
                        (
                            result.ite
                            < 0
                        ).mean()
                    ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .set_index(
            "repeat"
        )
    )


def repeated_crossfit_treatment_effect_model(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    n_repeats: int = 20,
    n_splits: int = 5,
    C: float = 0.1,
    max_iter: int = 10000,
    base_random_state: int = 42,
    random_states: Sequence[int] | None = None,
) -> RepeatedCrossFitResult:
    """
    Run repeated HERMES cross-fitting.

    Parameters
    ----------
    X
        Biological feature matrix.

    treatment
        Binary randomized treatment:
            0 = CT
            1 = CT/A

    outcome
        Binary outcome:
            0 = RD
            1 = pCR

    n_repeats
        Number of independently shuffled cross-fitting runs.

    n_splits
        Number of folds within each repeat.

    C
        L2 logistic-regression inverse regularization strength.

    max_iter
        Maximum logistic-regression iterations.

    base_random_state
        Seed used to deterministically generate repeat seeds.

    random_states
        Optional explicit sequence of repeat-specific seeds.
        If supplied, n_repeats is inferred from this sequence.
    """

    if random_states is None:
        states = (
            generate_random_states(
                n_repeats,
                base_random_state=(
                    base_random_state
                ),
            )
        )

    else:
        states = (
            validate_random_states(
                random_states
            )
        )

        n_repeats = len(
            states
        )

    results: list[
        CrossFitTreatmentEffectResult
    ] = []

    ite_columns: dict[
        str,
        pd.Series,
    ] = {}

    observed_columns: dict[
        str,
        pd.Series,
    ] = {}

    control_columns: dict[
        str,
        pd.Series,
    ] = {}

    treated_columns: dict[
        str,
        pd.Series,
    ] = {}

    # ---------------------------------------------------------
    # Repeated cross-fitting
    # ---------------------------------------------------------

    for repeat_number, (
        random_state
    ) in enumerate(
        states,
        start=1,
    ):

        result = (
            crossfit_treatment_effect_model(
                X=X,
                treatment=treatment,
                outcome=outcome,
                n_splits=n_splits,
                C=C,
                max_iter=max_iter,
                random_state=(
                    random_state
                ),
            )
        )

        results.append(
            result
        )

        column_name = (
            f"repeat_{repeat_number:03d}"
        )

        ite_columns[
            column_name
        ] = result.ite

        observed_columns[
            column_name
        ] = (
            result
            .observed_probability
        )

        control_columns[
            column_name
        ] = (
            result
            .probability_control
        )

        treated_columns[
            column_name
        ] = (
            result
            .probability_treated
        )

    # ---------------------------------------------------------
    # Assemble patient x repeat matrices
    # ---------------------------------------------------------

    ite_by_repeat = pd.DataFrame(
        ite_columns,
        index=X.index,
    )

    observed_probability_by_repeat = (
        pd.DataFrame(
            observed_columns,
            index=X.index,
        )
    )

    probability_control_by_repeat = (
        pd.DataFrame(
            control_columns,
            index=X.index,
        )
    )

    probability_treated_by_repeat = (
        pd.DataFrame(
            treated_columns,
            index=X.index,
        )
    )

    # ---------------------------------------------------------
    # Integrity
    # ---------------------------------------------------------

    matrices = [
        ite_by_repeat,
        observed_probability_by_repeat,
        probability_control_by_repeat,
        probability_treated_by_repeat,
    ]

    for matrix in matrices:

        if matrix.shape != (
            len(X),
            n_repeats,
        ):
            raise RuntimeError(
                "Repeated cross-fitting produced "
                "an unexpected matrix shape."
            )

        if matrix.isna().any().any():
            raise RuntimeError(
                "Repeated cross-fitting produced "
                "missing predictions."
            )

        if not matrix.index.equals(
            X.index
        ):
            raise RuntimeError(
                "Patient order changed during "
                "repeated cross-fitting."
            )

    # ---------------------------------------------------------
    # Patient stability summary
    # ---------------------------------------------------------

    patient_summary = (
        _summarize_patient_ite(
            ite_by_repeat
        )
    )

    # ---------------------------------------------------------
    # Repeat-level diagnostics
    # ---------------------------------------------------------

    repeat_summary = (
        _build_repeat_summary(
            results,
            states,
        )
    )

    # ---------------------------------------------------------
    # Aggregated counterfactual predictions
    # ---------------------------------------------------------

    mean_control_prediction = (
        probability_control_by_repeat
        .mean(
            axis=1
        )
    )

    mean_treated_prediction = (
        probability_treated_by_repeat
        .mean(
            axis=1
        )
    )

    aggregated_ite = (
        mean_treated_prediction
        - mean_control_prediction
    )

    # This should equal patient mean ITE because subtraction
    # commutes with averaging.
    np.testing.assert_allclose(
        aggregated_ite.to_numpy(),
        patient_summary[
            "mean_ite"
        ].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )

    # ---------------------------------------------------------
    # Cohort-level stability metrics
    # ---------------------------------------------------------

    mean_patient_sd = float(
        patient_summary[
            "ite_std"
        ].mean()
    )

    median_patient_sd = float(
        patient_summary[
            "ite_std"
        ].median()
    )

    highly_sign_stable_fraction = float(
        (
            patient_summary[
                "sign_stability"
            ]
            >= 0.90
        ).mean()
    )

    unanimous_sign_fraction = float(
        (
            patient_summary[
                "sign_stability"
            ]
            == 1.0
        ).mean()
    )

    consensus_benefit_fraction = float(
        (
            patient_summary[
                "consensus_direction"
            ]
            == "benefit"
        ).mean()
    )

    consensus_harm_fraction = float(
        (
            patient_summary[
                "consensus_direction"
            ]
            == "harm"
        ).mean()
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary: dict[
        str,
        Any,
    ] = {
        "patients":
            int(
                len(X)
            ),
        "biological_features":
            int(
                X.shape[1]
            ),
        "n_repeats":
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
        "mean_repeat_oof_auc":
            float(
                repeat_summary[
                    "oof_auc"
                ].mean()
            ),
        "std_repeat_oof_auc":
            float(
                repeat_summary[
                    "oof_auc"
                ].std()
            ),
        "mean_repeat_oof_brier":
            float(
                repeat_summary[
                    "oof_brier"
                ].mean()
            ),
        "std_repeat_oof_brier":
            float(
                repeat_summary[
                    "oof_brier"
                ].std()
            ),
        "cohort_mean_ite":
            float(
                patient_summary[
                    "mean_ite"
                ].mean()
            ),
        "cohort_median_ite":
            float(
                patient_summary[
                    "mean_ite"
                ].median()
            ),
        "minimum_patient_mean_ite":
            float(
                patient_summary[
                    "mean_ite"
                ].min()
            ),
        "maximum_patient_mean_ite":
            float(
                patient_summary[
                    "mean_ite"
                ].max()
            ),
        "mean_patient_ite_sd":
            mean_patient_sd,
        "median_patient_ite_sd":
            median_patient_sd,
        "fraction_sign_stability_ge_90pct":
            highly_sign_stable_fraction,
        "fraction_unanimous_sign":
            unanimous_sign_fraction,
        "fraction_consensus_benefit":
            consensus_benefit_fraction,
        "fraction_consensus_harm":
            consensus_harm_fraction,
    }

    return RepeatedCrossFitResult(
        ite_by_repeat=(
            ite_by_repeat
        ),
        observed_probability_by_repeat=(
            observed_probability_by_repeat
        ),
        probability_control_by_repeat=(
            probability_control_by_repeat
        ),
        probability_treated_by_repeat=(
            probability_treated_by_repeat
        ),
        patient_summary=(
            patient_summary
        ),
        repeat_summary=(
            repeat_summary
        ),
        summary=summary,
        random_states=states,
    )


def stable_patient_table(
    result: RepeatedCrossFitResult,
    *,
    sort_by: str = "mean_ite",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Return patient-level repeated-cross-fit stability results.
    """

    if sort_by not in (
        result.patient_summary.columns
    ):
        raise ValueError(
            f"Unknown sort column: {sort_by}"
        )

    return (
        result.patient_summary
        .sort_values(
            sort_by,
            ascending=ascending,
        )
        .copy()
    )


def main() -> None:
    """
    Run repeated HERMES cross-fitting on NeoTRIP.
    """

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    dataset = (
        build_treatment_effect_dataset()
    )

    result = (
        repeated_crossfit_treatment_effect_model(
            X=dataset.X,
            treatment=dataset.T,
            outcome=dataset.Y,
            n_repeats=20,
            n_splits=5,
            C=0.1,
            base_random_state=42,
        )
    )

    print(
        "=== HERMES 2.0 "
        "REPEATED CROSS-FITTED "
        "TREATMENT-EFFECT MODEL ==="
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
        "Repeat-level diagnostics:"
    )

    print(
        result.repeat_summary
        .to_string()
    )

    print()

    print(
        "Patients with highest "
        "mean estimated benefit:"
    )

    print(
        stable_patient_table(
            result,
            sort_by="mean_ite",
            ascending=False,
        )
        .head(10)
        .to_string()
    )

    print()

    print(
        "Patients with most stable "
        "estimated treatment effect:"
    )

    print(
        stable_patient_table(
            result,
            sort_by="ite_std",
            ascending=True,
        )
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()