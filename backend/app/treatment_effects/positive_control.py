"""
HERMES 2.0
Positive-Control Simulation for Treatment-Effect Recovery
==========================================================

Purpose
-------
Test whether HERMES can recover treatment-effect heterogeneity when the
ground truth is known.

The simulation creates a randomized two-arm trial with synthetic biological
features.

One feature is deliberately made predictive of treatment benefit:

    logit P(Y=1 | X, T)
        =
        beta_0
        + beta_T * T
        + beta_X * X_signal
        + beta_TX * T * X_signal
        + additional prognostic effects

The true individualized treatment effect is therefore known:

    tau_i
        =
        P(Y=1 | X_i, T=1)
        -
        P(Y=1 | X_i, T=0)

HERMES is NOT given the true tau values.

It receives only:
    - biological feature matrix X
    - randomized treatment assignment T
    - binary outcome Y

We then compare HERMES repeated-cross-fit ITE estimates against the
known data-generating truth.

A useful treatment-effect learner should show:

    1. positive correlation between estimated and true ITE
    2. correct ranking of high-benefit patients
    3. improved treatment benefit in predicted high-benefit groups
    4. correct treatment-effect direction
    5. stronger recovery as the injected interaction grows
    6. much weaker apparent heterogeneity when beta_TX = 0

IMPORTANT
---------
This is a controlled simulation.

Success here demonstrates that the HERMES implementation can recover
heterogeneous treatment effects under favorable known conditions.

It does NOT establish that heterogeneous treatment effects exist in NeoTRIP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.app.treatment_effects.repeated_crossfit import (
    RepeatedCrossFitResult,
    repeated_crossfit_treatment_effect_model,
)


# =============================================================
# Data containers
# =============================================================


@dataclass
class PositiveControlDataset:
    """
    Synthetic randomized-trial dataset with known treatment effects.
    """

    X: pd.DataFrame

    T: pd.Series
    Y: pd.Series

    probability_control: pd.Series
    probability_treated: pd.Series

    true_ite: pd.Series

    signal_feature: str

    parameters: dict[str, Any]

    @property
    def n_patients(self) -> int:
        return int(
            self.X.shape[0]
        )

    @property
    def n_features(self) -> int:
        return int(
            self.X.shape[1]
        )


@dataclass
class PositiveControlResult:
    """
    HERMES recovery results for a positive-control simulation.
    """

    dataset: PositiveControlDataset

    hermes_result: RepeatedCrossFitResult

    patient_table: pd.DataFrame

    metrics: dict[str, float]

    summary: dict[str, Any]


# =============================================================
# Mathematical helpers
# =============================================================


def sigmoid(
    x: np.ndarray,
) -> np.ndarray:
    """
    Numerically stable logistic function.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    positive = (
        x >= 0
    )

    negative = ~positive

    output = np.empty_like(
        x,
        dtype=float,
    )

    output[
        positive
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -x[
                    positive
                ]
            )
        )
    )

    exp_x = np.exp(
        x[
            negative
        ]
    )

    output[
        negative
    ] = (
        exp_x
        / (
            1.0
            + exp_x
        )
    )

    return output


def pearson_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    """
    Pearson correlation without requiring scipy.
    """

    x_values = np.asarray(
        x,
        dtype=float,
    )

    y_values = np.asarray(
        y,
        dtype=float,
    )

    if len(
        x_values
    ) != len(
        y_values
    ):
        raise ValueError(
            "Correlation vectors must have equal length."
        )

    if (
        np.std(
            x_values
        )
        == 0
        or np.std(
            y_values
        )
        == 0
    ):
        return float(
            "nan"
        )

    return float(
        np.corrcoef(
            x_values,
            y_values,
        )[0, 1]
    )


def rank_values(
    values: pd.Series,
) -> pd.Series:
    """
    Convert values to average ranks.
    """

    return (
        values
        .rank(
            method="average"
        )
        .astype(float)
    )


def spearman_correlation(
    x: pd.Series,
    y: pd.Series,
) -> float:
    """
    Spearman rank correlation.
    """

    return pearson_correlation(
        rank_values(
            x
        ),
        rank_values(
            y
        ),
    )


# =============================================================
# Synthetic trial generation
# =============================================================


def generate_positive_control_dataset(
    *,
    n_patients: int = 500,
    n_features: int = 20,
    treatment_probability: float = 0.5,
    intercept: float = -0.25,
    treatment_main_effect: float = 0.25,
    signal_main_effect: float = 0.45,
    treatment_interaction: float = 1.50,
    prognostic_features: int = 4,
    prognostic_scale: float = 0.20,
    feature_correlation: float = 0.20,
    random_state: int = 2026,
) -> PositiveControlDataset:
    """
    Generate a randomized synthetic treatment-effect dataset.

    Feature_001 is the injected treatment-effect modifier.

    Other features may influence baseline outcome probability but do not
    modify treatment effect.
    """

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if n_patients < 50:
        raise ValueError(
            "n_patients must be at least 50."
        )

    if n_features < 2:
        raise ValueError(
            "n_features must be at least 2."
        )

    if not (
        0.0
        < treatment_probability
        < 1.0
    ):
        raise ValueError(
            "treatment_probability must lie between 0 and 1."
        )

    if prognostic_features < 0:
        raise ValueError(
            "prognostic_features must be non-negative."
        )

    if prognostic_features >= n_features:
        raise ValueError(
            "prognostic_features must be smaller than n_features."
        )

    if not (
        0.0
        <= feature_correlation
        < 1.0
    ):
        raise ValueError(
            "feature_correlation must be in [0, 1)."
        )

    rng = np.random.default_rng(
        random_state
    )

    # ---------------------------------------------------------
    # Correlated biological features
    # ---------------------------------------------------------

    covariance = np.full(
        (
            n_features,
            n_features,
        ),
        feature_correlation,
        dtype=float,
    )

    np.fill_diagonal(
        covariance,
        1.0,
    )

    X_values = rng.multivariate_normal(
        mean=np.zeros(
            n_features,
            dtype=float,
        ),
        cov=covariance,
        size=n_patients,
    )

    feature_names = [
        f"FEATURE_{i:03d}"
        for i in range(
            1,
            n_features + 1,
        )
    ]

    patient_ids = [
        f"SIM_{i:05d}"
        for i in range(
            1,
            n_patients + 1,
        )
    ]

    X = pd.DataFrame(
        X_values,
        index=patient_ids,
        columns=feature_names,
    )

    X.index.name = (
        "Patient_ID"
    )

    signal_feature = (
        feature_names[0]
    )

    signal = (
        X[
            signal_feature
        ]
    )

    # ---------------------------------------------------------
    # Randomized treatment assignment
    # ---------------------------------------------------------

    T_values = rng.binomial(
        n=1,
        p=treatment_probability,
        size=n_patients,
    )

    T = pd.Series(
        T_values,
        index=X.index,
        name="T",
        dtype=int,
    )

    # ---------------------------------------------------------
    # Prognostic biology
    #
    # Features after FEATURE_001 influence baseline response,
    # but do not modify treatment effect.
    # ---------------------------------------------------------

    prognostic_component = np.zeros(
        n_patients,
        dtype=float,
    )

    if prognostic_features > 0:

        coefficients = np.linspace(
            prognostic_scale,
            -prognostic_scale,
            prognostic_features,
        )

        for offset, coefficient in enumerate(
            coefficients,
            start=1,
        ):

            prognostic_component += (
                coefficient
                * X_values[
                    :,
                    offset,
                ]
            )

    # ---------------------------------------------------------
    # Potential outcome under control
    # ---------------------------------------------------------

    linear_control = (
        intercept
        + signal_main_effect
        * signal.to_numpy(
            dtype=float
        )
        + prognostic_component
    )

    # ---------------------------------------------------------
    # Potential outcome under treatment
    #
    # FEATURE_001 modifies treatment effect.
    # ---------------------------------------------------------

    linear_treated = (
        intercept
        + signal_main_effect
        * signal.to_numpy(
            dtype=float
        )
        + prognostic_component
        + treatment_main_effect
        + treatment_interaction
        * signal.to_numpy(
            dtype=float
        )
    )

    probability_control_values = sigmoid(
        linear_control
    )

    probability_treated_values = sigmoid(
        linear_treated
    )

    true_ite_values = (
        probability_treated_values
        - probability_control_values
    )

    # ---------------------------------------------------------
    # Observed randomized potential outcome
    # ---------------------------------------------------------

    observed_probability = np.where(
        T_values == 1,
        probability_treated_values,
        probability_control_values,
    )

    Y_values = rng.binomial(
        n=1,
        p=observed_probability,
        size=n_patients,
    )

    Y = pd.Series(
        Y_values,
        index=X.index,
        name="Y",
        dtype=int,
    )

    probability_control = pd.Series(
        probability_control_values,
        index=X.index,
        name="true_probability_control",
    )

    probability_treated = pd.Series(
        probability_treated_values,
        index=X.index,
        name="true_probability_treated",
    )

    true_ite = pd.Series(
        true_ite_values,
        index=X.index,
        name="true_ite",
    )

    parameters: dict[
        str,
        Any,
    ] = {
        "n_patients":
            int(
                n_patients
            ),

        "n_features":
            int(
                n_features
            ),

        "treatment_probability":
            float(
                treatment_probability
            ),

        "intercept":
            float(
                intercept
            ),

        "treatment_main_effect":
            float(
                treatment_main_effect
            ),

        "signal_main_effect":
            float(
                signal_main_effect
            ),

        "treatment_interaction":
            float(
                treatment_interaction
            ),

        "prognostic_features":
            int(
                prognostic_features
            ),

        "prognostic_scale":
            float(
                prognostic_scale
            ),

        "feature_correlation":
            float(
                feature_correlation
            ),

        "random_state":
            int(
                random_state
            ),
    }

    return PositiveControlDataset(
        X=X,
        T=T,
        Y=Y,
        probability_control=(
            probability_control
        ),
        probability_treated=(
            probability_treated
        ),
        true_ite=(
            true_ite
        ),
        signal_feature=(
            signal_feature
        ),
        parameters=(
            parameters
        ),
    )


# =============================================================
# Recovery metrics
# =============================================================


def build_positive_control_patient_table(
    dataset: PositiveControlDataset,
    hermes_result: RepeatedCrossFitResult,
) -> pd.DataFrame:
    """
    Build patient-level ground-truth versus HERMES comparison.
    """

    estimated_ite = (
        hermes_result
        .patient_summary[
            "mean_ite"
        ]
    )

    if not estimated_ite.index.equals(
        dataset.X.index
    ):
        raise RuntimeError(
            "HERMES patient order does not match "
            "the positive-control dataset."
        )

    table = pd.DataFrame(
        index=dataset.X.index
    )

    table[
        "signal_feature_value"
    ] = (
        dataset.X[
            dataset.signal_feature
        ]
    )

    table[
        "T"
    ] = dataset.T

    table[
        "Y"
    ] = dataset.Y

    table[
        "true_probability_control"
    ] = (
        dataset
        .probability_control
    )

    table[
        "true_probability_treated"
    ] = (
        dataset
        .probability_treated
    )

    table[
        "true_ite"
    ] = (
        dataset.true_ite
    )

    table[
        "estimated_ite"
    ] = estimated_ite

    table[
        "estimated_ite_std"
    ] = (
        hermes_result
        .patient_summary[
            "ite_std"
        ]
    )

    table[
        "estimated_sign_stability"
    ] = (
        hermes_result
        .patient_summary[
            "sign_stability"
        ]
    )

    table[
        "true_benefit"
    ] = (
        table[
            "true_ite"
        ]
        > 0
    )

    table[
        "estimated_benefit"
    ] = (
        table[
            "estimated_ite"
        ]
        > 0
    )

    return table


def calculate_positive_control_metrics(
    patient_table: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate recovery metrics comparing HERMES ITE estimates with truth.
    """

    true_ite = (
        patient_table[
            "true_ite"
        ]
    )

    estimated_ite = (
        patient_table[
            "estimated_ite"
        ]
    )

    # ---------------------------------------------------------
    # Correlation / ranking recovery
    # ---------------------------------------------------------

    ite_pearson = pearson_correlation(
        true_ite,
        estimated_ite,
    )

    ite_spearman = spearman_correlation(
        true_ite,
        estimated_ite,
    )

    signal_pearson = pearson_correlation(
        patient_table[
            "signal_feature_value"
        ],
        estimated_ite,
    )

    # ---------------------------------------------------------
    # Sign recovery
    # ---------------------------------------------------------

    nonzero_truth = (
        true_ite != 0
    )

    if nonzero_truth.any():

        sign_accuracy = float(
            (
                np.sign(
                    true_ite[
                        nonzero_truth
                    ]
                )
                ==
                np.sign(
                    estimated_ite[
                        nonzero_truth
                    ]
                )
            ).mean()
        )

    else:

        sign_accuracy = float(
            "nan"
        )

    # ---------------------------------------------------------
    # Top-quartile ranking
    # ---------------------------------------------------------

    estimated_top_threshold = (
        estimated_ite.quantile(
            0.75
        )
    )

    estimated_bottom_threshold = (
        estimated_ite.quantile(
            0.25
        )
    )

    estimated_top = (
        estimated_ite
        >= estimated_top_threshold
    )

    estimated_bottom = (
        estimated_ite
        <= estimated_bottom_threshold
    )

    true_mean_ite_top_predicted = float(
        true_ite[
            estimated_top
        ].mean()
    )

    true_mean_ite_bottom_predicted = float(
        true_ite[
            estimated_bottom
        ].mean()
    )

    true_ite_separation = float(
        true_mean_ite_top_predicted
        - true_mean_ite_bottom_predicted
    )

    # ---------------------------------------------------------
    # True top-quartile overlap
    # ---------------------------------------------------------

    true_top_threshold = (
        true_ite.quantile(
            0.75
        )
    )

    true_top = (
        true_ite
        >= true_top_threshold
    )

    top_quartile_overlap = float(
        (
            estimated_top
            & true_top
        ).sum()
        / max(
            1,
            int(
                true_top.sum()
            ),
        )
    )

    # ---------------------------------------------------------
    # Absolute estimation error
    # ---------------------------------------------------------

    error = (
        estimated_ite
        - true_ite
    )

    mean_absolute_error = float(
        error.abs().mean()
    )

    root_mean_squared_error = float(
        np.sqrt(
            np.mean(
                np.square(
                    error.to_numpy(
                        dtype=float
                    )
                )
            )
        )
    )

    # ---------------------------------------------------------
    # Heterogeneity comparison
    # ---------------------------------------------------------

    true_ite_sd = float(
        true_ite.std()
    )

    estimated_ite_sd = float(
        estimated_ite.std()
    )

    # ---------------------------------------------------------
    # Treatment-effect scale
    # ---------------------------------------------------------

    mean_true_ite = float(
        true_ite.mean()
    )

    mean_estimated_ite = float(
        estimated_ite.mean()
    )

    return {
        "ite_pearson_correlation":
            ite_pearson,

        "ite_spearman_correlation":
            ite_spearman,

        "signal_feature_estimated_ite_correlation":
            signal_pearson,

        "ite_sign_accuracy":
            sign_accuracy,

        "top_quartile_overlap":
            top_quartile_overlap,

        "true_mean_ite_top_predicted_quartile":
            true_mean_ite_top_predicted,

        "true_mean_ite_bottom_predicted_quartile":
            true_mean_ite_bottom_predicted,

        "true_ite_top_bottom_separation":
            true_ite_separation,

        "mean_absolute_ite_error":
            mean_absolute_error,

        "root_mean_squared_ite_error":
            root_mean_squared_error,

        "true_ite_sd":
            true_ite_sd,

        "estimated_ite_sd":
            estimated_ite_sd,

        "mean_true_ite":
            mean_true_ite,

        "mean_estimated_ite":
            mean_estimated_ite,
    }


# =============================================================
# HERMES positive-control experiment
# =============================================================


def run_positive_control(
    *,
    n_patients: int = 500,
    n_features: int = 20,
    treatment_interaction: float = 1.50,
    treatment_main_effect: float = 0.25,
    signal_main_effect: float = 0.45,
    n_repeats: int = 10,
    n_splits: int = 5,
    C: float = 0.1,
    data_random_state: int = 2026,
    model_random_state: int = 42,
) -> PositiveControlResult:
    """
    Generate a known HTE signal and test HERMES recovery.
    """

    dataset = (
        generate_positive_control_dataset(
            n_patients=n_patients,
            n_features=n_features,
            treatment_interaction=(
                treatment_interaction
            ),
            treatment_main_effect=(
                treatment_main_effect
            ),
            signal_main_effect=(
                signal_main_effect
            ),
            random_state=(
                data_random_state
            ),
        )
    )

    hermes_result = (
        repeated_crossfit_treatment_effect_model(
            X=dataset.X,
            treatment=dataset.T,
            outcome=dataset.Y,
            n_repeats=n_repeats,
            n_splits=n_splits,
            C=C,
            base_random_state=(
                model_random_state
            ),
        )
    )

    patient_table = (
        build_positive_control_patient_table(
            dataset,
            hermes_result,
        )
    )

    metrics = (
        calculate_positive_control_metrics(
            patient_table
        )
    )

    summary: dict[
        str,
        Any,
    ] = {
        "patients":
            dataset.n_patients,

        "features":
            dataset.n_features,

        "signal_feature":
            dataset.signal_feature,

        "injected_treatment_interaction":
            float(
                treatment_interaction
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

        "observed_treatment_fraction":
            float(
                dataset.T.mean()
            ),

        "observed_outcome_fraction":
            float(
                dataset.Y.mean()
            ),

        **metrics,
    }

    return PositiveControlResult(
        dataset=dataset,
        hermes_result=hermes_result,
        patient_table=patient_table,
        metrics=metrics,
        summary=summary,
    )


# =============================================================
# Interaction-strength recovery curve
# =============================================================


def run_interaction_strength_experiment(
    interaction_strengths: tuple[
        float,
        ...,
    ] = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ),
    *,
    n_patients: int = 500,
    n_features: int = 20,
    n_repeats: int = 5,
    n_splits: int = 5,
    C: float = 0.1,
    data_random_state: int = 2026,
    model_random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate HERMES recovery as the true interaction strength increases.

    The same biological feature matrix/randomized-trial seed is used for
    each interaction strength so changes are attributable primarily to
    the injected HTE strength rather than completely different cohorts.
    """

    records: list[
        dict[str, float]
    ] = []

    for interaction_strength in (
        interaction_strengths
    ):

        result = run_positive_control(
            n_patients=n_patients,
            n_features=n_features,
            treatment_interaction=(
                interaction_strength
            ),
            n_repeats=n_repeats,
            n_splits=n_splits,
            C=C,
            data_random_state=(
                data_random_state
            ),
            model_random_state=(
                model_random_state
            ),
        )

        record: dict[
            str,
            float,
        ] = {
            "interaction_strength":
                float(
                    interaction_strength
                ),

            "true_ite_sd":
                float(
                    result.metrics[
                        "true_ite_sd"
                    ]
                ),

            "estimated_ite_sd":
                float(
                    result.metrics[
                        "estimated_ite_sd"
                    ]
                ),

            "ite_pearson_correlation":
                float(
                    result.metrics[
                        "ite_pearson_correlation"
                    ]
                ),

            "ite_spearman_correlation":
                float(
                    result.metrics[
                        "ite_spearman_correlation"
                    ]
                ),

            "ite_sign_accuracy":
                float(
                    result.metrics[
                        "ite_sign_accuracy"
                    ]
                ),

            "top_quartile_overlap":
                float(
                    result.metrics[
                        "top_quartile_overlap"
                    ]
                ),

            "true_ite_top_bottom_separation":
                float(
                    result.metrics[
                        "true_ite_top_bottom_separation"
                    ]
                ),

            "mean_absolute_ite_error":
                float(
                    result.metrics[
                        "mean_absolute_ite_error"
                    ]
                ),
        }

        records.append(
            record
        )

        print(
            "Interaction "
            f"{interaction_strength:.2f} "
            "complete"
        )

    return (
        pd.DataFrame(
            records
        )
        .set_index(
            "interaction_strength"
        )
    )


# =============================================================
# CLI
# =============================================================


def main() -> None:
    """
    Run a development-scale positive-control experiment.
    """

    print(
        "=== HERMES 2.0 "
        "POSITIVE-CONTROL TREATMENT-EFFECT RECOVERY ==="
    )

    print()

    result = run_positive_control(
        n_patients=500,
        n_features=20,
        treatment_interaction=1.50,
        n_repeats=10,
        n_splits=5,
        C=0.1,
        data_random_state=2026,
        model_random_state=42,
    )

    for key, value in (
        result.summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Highest true-benefit patients:"
    )

    print(
        result.patient_table
        .sort_values(
            "true_ite",
            ascending=False,
        )
        .head(10)[
            [
                "signal_feature_value",
                "true_ite",
                "estimated_ite",
                "estimated_ite_std",
                "estimated_sign_stability",
            ]
        ]
        .to_string()
    )

    print()

    print(
        "Highest HERMES-predicted benefit:"
    )

    print(
        result.patient_table
        .sort_values(
            "estimated_ite",
            ascending=False,
        )
        .head(10)[
            [
                "signal_feature_value",
                "true_ite",
                "estimated_ite",
                "estimated_ite_std",
                "estimated_sign_stability",
            ]
        ]
        .to_string()
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "Positive correlation and correct patient ranking "
        "indicate recovery of the injected treatment-effect signal."
    )

    print(
        "This simulation validates capability only; it does not "
        "establish treatment-effect heterogeneity in NeoTRIP."
    )


if __name__ == "__main__":
    main()