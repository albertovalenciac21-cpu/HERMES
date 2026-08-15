"""
HERMES 2.0
Biological Treatment-Effect Modifier Discovery
===============================================

Purpose
-------
Identify biological pathways whose baseline activity may modify the
incremental treatment effect of adding atezolizumab to chemotherapy in
NeoTRIP.

For each biological pathway, HERMES fits the interaction model:

    logit[P(Y = 1)] =
        beta_0
        + beta_T * T
        + beta_X * X
        + beta_TX * (T * X)

where:

    Y = pCR outcome
    T = randomized treatment indicator
        0 -> chemotherapy
        1 -> chemotherapy + atezolizumab
    X = standardized baseline biological pathway score

The parameter of primary interest is:

    beta_TX

which represents treatment-effect modification on the log-odds scale.

This module deliberately distinguishes:

    beta_X   -> prognostic association
    beta_TX  -> predictive/treatment-interaction association

Multiple testing across the Hallmark pathways is controlled with the
Benjamini-Hochberg false-discovery-rate procedure.

IMPORTANT
---------
This is an exploratory treatment-effect modifier analysis.

A statistically interesting interaction is not automatically a validated
predictive biomarker. Confirmation requires robustness analyses, repeated
cross-fitting, biological validation, and ideally external validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm


# =============================================================
# Result containers
# =============================================================


@dataclass
class InteractionModelResult:
    """
    Result from one pathway × treatment interaction model.
    """

    feature: str

    n: int
    converged: bool

    feature_mean: float
    feature_sd: float

    intercept: float
    treatment_coefficient: float
    prognostic_coefficient: float
    interaction_coefficient: float

    treatment_standard_error: float
    prognostic_standard_error: float
    interaction_standard_error: float

    treatment_z: float
    prognostic_z: float
    interaction_z: float

    treatment_p_value: float
    prognostic_p_value: float
    interaction_p_value: float

    interaction_odds_ratio: float
    interaction_or_ci_lower: float
    interaction_or_ci_upper: float

    feature_q25_z: float
    feature_q75_z: float

    pcr_probability_control_q25: float
    pcr_probability_treated_q25: float
    risk_difference_q25: float

    pcr_probability_control_q75: float
    pcr_probability_treated_q75: float
    risk_difference_q75: float

    risk_difference_contrast: float


@dataclass
class ModifierDiscoveryResult:
    """
    Full HERMES biological modifier-discovery result.
    """

    modifier_table: pd.DataFrame
    summary: dict[str, Any]


# =============================================================
# Validation
# =============================================================


def _validate_binary_series(
    values: pd.Series,
    name: str,
) -> pd.Series:
    """
    Validate a binary 0/1 pandas Series.
    """

    if not isinstance(
        values,
        pd.Series,
    ):
        raise TypeError(
            f"{name} must be a pandas Series."
        )

    if values.isna().any():
        raise ValueError(
            f"{name} contains missing values."
        )

    unique = set(
        values.astype(int).unique()
    )

    if not unique.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"{name} must contain only 0 and 1."
        )

    if len(unique) < 2:
        raise ValueError(
            f"{name} must contain both binary classes."
        )

    return values.astype(
        int
    )


def _validate_feature_matrix(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate biological feature matrix.
    """

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "X cannot be empty."
        )

    numeric = X.astype(
        float
    )

    if not np.isfinite(
        numeric.to_numpy()
    ).all():
        raise ValueError(
            "X contains non-finite values."
        )

    if numeric.columns.duplicated().any():
        raise ValueError(
            "X contains duplicate feature names."
        )

    return numeric


# =============================================================
# Benjamini-Hochberg FDR
# =============================================================


def benjamini_hochberg(
    p_values: pd.Series,
) -> pd.Series:
    """
    Benjamini-Hochberg adjusted p-values.

    Returned values are monotonic and bounded to [0, 1].
    """

    if not isinstance(
        p_values,
        pd.Series,
    ):
        p_values = pd.Series(
            p_values
        )

    values = p_values.astype(
        float
    )

    if values.isna().any():
        raise ValueError(
            "p_values contains missing values."
        )

    if (
        (values < 0.0)
        | (values > 1.0)
    ).any():
        raise ValueError(
            "p_values must be between 0 and 1."
        )

    n = len(
        values
    )

    if n == 0:
        return values.copy()

    order = np.argsort(
        values.to_numpy()
    )

    sorted_p = (
        values.to_numpy()[
            order
        ]
    )

    ranks = np.arange(
        1,
        n + 1,
        dtype=float,
    )

    adjusted = (
        sorted_p
        * n
        / ranks
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )

    restored = np.empty(
        n,
        dtype=float,
    )

    restored[
        order
    ] = adjusted

    return pd.Series(
        restored,
        index=values.index,
        name="fdr",
    )


# =============================================================
# Logistic interaction model
# =============================================================


def _negative_log_likelihood(
    beta: np.ndarray,
    design: np.ndarray,
    outcome: np.ndarray,
) -> float:
    """
    Numerically stable logistic negative log-likelihood.
    """

    linear = (
        design
        @ beta
    )

    return float(
        np.sum(
            np.logaddexp(
                0.0,
                linear,
            )
            - outcome * linear
        )
    )


def _negative_log_likelihood_gradient(
    beta: np.ndarray,
    design: np.ndarray,
    outcome: np.ndarray,
) -> np.ndarray:
    """
    Gradient of logistic negative log-likelihood.
    """

    probabilities = expit(
        design
        @ beta
    )

    return (
        design.T
        @ (
            probabilities
            - outcome
        )
    )


def _fit_logistic_model(
    design: np.ndarray,
    outcome: np.ndarray,
    *,
    max_iter: int = 10000,
) -> tuple[
    np.ndarray,
    np.ndarray,
    bool,
]:
    """
    Fit an unpenalized logistic regression model robustly.

    Strategy
    --------
    1. Fit with BFGS.
    2. Refit with L-BFGS-B as a numerical cross-check/fallback.
    3. Evaluate every finite candidate using both:
       - optimizer success
       - gradient infinity norm
    4. Prefer candidates satisfying numerical convergence.
    5. Among converged candidates, select the solution with the
       lowest negative log-likelihood.

    This prevents an otherwise valid maximum-likelihood estimate
    from being labeled non-converged solely because of optimizer
    precision-loss bookkeeping.
    """

    if design.ndim != 2:
        raise ValueError(
            "design must be two-dimensional."
        )

    if outcome.ndim != 1:
        raise ValueError(
            "outcome must be one-dimensional."
        )

    if design.shape[0] != len(
        outcome
    ):
        raise ValueError(
            "design and outcome dimensions do not match."
        )

    if not np.isfinite(
        design
    ).all():
        raise ValueError(
            "design contains non-finite values."
        )

    if not np.isfinite(
        outcome
    ).all():
        raise ValueError(
            "outcome contains non-finite values."
        )

    initial = np.zeros(
        design.shape[1],
        dtype=float,
    )

    gradient_tolerance = (
        1e-5
    )

    # ---------------------------------------------------------
    # Primary optimizer: BFGS
    # ---------------------------------------------------------

    fit_bfgs = minimize(
        _negative_log_likelihood,
        initial,
        args=(
            design,
            outcome,
        ),
        jac=(
            _negative_log_likelihood_gradient
        ),
        method="BFGS",
        options={
            "maxiter":
                int(
                    max_iter
                ),

            "gtol":
                1e-8,
        },
    )

    # ---------------------------------------------------------
    # Numerical cross-check/fallback: L-BFGS-B
    # ---------------------------------------------------------

    fallback_start = (
        fit_bfgs.x
        if np.isfinite(
            fit_bfgs.x
        ).all()
        else initial
    )

    fit_lbfgs = minimize(
        _negative_log_likelihood,
        fallback_start,
        args=(
            design,
            outcome,
        ),
        jac=(
            _negative_log_likelihood_gradient
        ),
        method="L-BFGS-B",
        options={
            "maxiter":
                int(
                    max_iter
                ),

            "ftol":
                1e-12,

            "gtol":
                1e-8,

            "maxls":
                50,
        },
    )

    candidates = [
        fit_bfgs,
        fit_lbfgs,
    ]

    # ---------------------------------------------------------
    # Evaluate candidate solutions
    # ---------------------------------------------------------

    evaluated_candidates: list[
        dict[str, Any]
    ] = []

    for candidate in candidates:

        if not (
            np.isfinite(
                candidate.fun
            )
            and np.isfinite(
                candidate.x
            ).all()
        ):
            continue

        beta_candidate = (
            candidate.x.astype(
                float
            )
        )

        gradient = (
            _negative_log_likelihood_gradient(
                beta_candidate,
                design,
                outcome,
            )
        )

        if np.isfinite(
            gradient
        ).all():

            gradient_inf_norm = float(
                np.max(
                    np.abs(
                        gradient
                    )
                )
            )

        else:

            gradient_inf_norm = (
                np.inf
            )

        numerically_converged = bool(
            candidate.success
            or (
                gradient_inf_norm
                <= gradient_tolerance
            )
        )

        evaluated_candidates.append(
            {
                "fit":
                    candidate,

                "beta":
                    beta_candidate,

                "objective":
                    float(
                        candidate.fun
                    ),

                "gradient_inf_norm":
                    gradient_inf_norm,

                "converged":
                    numerically_converged,
            }
        )

    if not evaluated_candidates:
        raise RuntimeError(
            "Logistic interaction model failed to produce "
            "a finite optimization solution."
        )

    # ---------------------------------------------------------
    # Prefer candidates satisfying convergence criteria
    # ---------------------------------------------------------

    converged_candidates = [
        candidate
        for candidate
        in evaluated_candidates
        if candidate[
            "converged"
        ]
    ]

    if converged_candidates:

        selected = min(
            converged_candidates,
            key=lambda candidate: (
                candidate[
                    "objective"
                ]
            ),
        )

        converged = True

    else:

        selected = min(
            evaluated_candidates,
            key=lambda candidate: (
                candidate[
                    "objective"
                ]
            ),
        )

        converged = False

    beta = selected[
        "beta"
    ]

    # ---------------------------------------------------------
    # Observed-information covariance matrix
    # ---------------------------------------------------------

    probabilities = expit(
        design
        @ beta
    )

    weights = (
        probabilities
        * (
            1.0
            - probabilities
        )
    )

    information = (
        design.T
        @ (
            design
            * weights[:, None]
        )
    )

    covariance = np.linalg.pinv(
        information,
        rcond=1e-12,
    )

    if not np.isfinite(
        covariance
    ).all():
        raise RuntimeError(
            "Logistic interaction covariance matrix "
            "contains non-finite values."
        )

    return (
        beta,
        covariance,
        converged,
    )


# =============================================================
# Single-pathway analysis
# =============================================================


def fit_pathway_interaction(
    feature: pd.Series,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    feature_name: str | None = None,
    max_iter: int = 10000,
) -> InteractionModelResult:
    """
    Fit:

        outcome ~ treatment + pathway + treatment:pathway

    Pathway scores are standardized before modeling.
    """

    treatment = _validate_binary_series(
        treatment,
        "treatment",
    )

    outcome = _validate_binary_series(
        outcome,
        "outcome",
    )

    if not isinstance(
        feature,
        pd.Series,
    ):
        raise TypeError(
            "feature must be a pandas Series."
        )

    if not (
        feature.index.equals(
            treatment.index
        )
        and feature.index.equals(
            outcome.index
        )
    ):
        raise ValueError(
            "feature, treatment, and outcome must have identical patient order."
        )

    x = feature.astype(
        float
    )

    if not np.isfinite(
        x.to_numpy()
    ).all():
        raise ValueError(
            "feature contains non-finite values."
        )

    name = (
        feature_name
        if feature_name is not None
        else str(
            feature.name
        )
    )

    mean = float(
        x.mean()
    )

    sd = float(
        x.std(
            ddof=0
        )
    )

    if (
        not np.isfinite(
            sd
        )
        or sd <= 0.0
    ):
        raise ValueError(
            f"Feature '{name}' has zero or invalid variance."
        )

    z = (
        x
        - mean
    ) / sd

    t = treatment.to_numpy(
        dtype=float
    )

    y = outcome.to_numpy(
        dtype=float
    )

    z_array = z.to_numpy(
        dtype=float
    )

    interaction = (
        t
        * z_array
    )

    design = np.column_stack(
        [
            np.ones(
                len(
                    x
                ),
                dtype=float,
            ),
            t,
            z_array,
            interaction,
        ]
    )

    beta, covariance, converged = (
        _fit_logistic_model(
            design,
            y,
            max_iter=max_iter,
        )
    )

    standard_errors = np.sqrt(
        np.clip(
            np.diag(
                covariance
            ),
            a_min=0.0,
            a_max=None,
        )
    )

    z_statistics = np.divide(
        beta,
        standard_errors,
        out=np.full_like(
            beta,
            np.nan,
            dtype=float,
        ),
        where=(
            standard_errors
            > 0.0
        ),
    )

    p_values = (
        2.0
        * norm.sf(
            np.abs(
                z_statistics
            )
        )
    )

    interaction_coefficient = float(
        beta[
            3
        ]
    )

    interaction_se = float(
        standard_errors[
            3
        ]
    )

    interaction_or = float(
        np.exp(
            interaction_coefficient
        )
    )

    interaction_ci_lower = float(
        np.exp(
            interaction_coefficient
            - 1.96
            * interaction_se
        )
    )

    interaction_ci_upper = float(
        np.exp(
            interaction_coefficient
            + 1.96
            * interaction_se
        )
    )

    q25_z = float(
        z.quantile(
            0.25
        )
    )

    q75_z = float(
        z.quantile(
            0.75
        )
    )

    def predicted_probability(
        *,
        treatment_value: float,
        feature_value: float,
    ) -> float:
        """
        Predict pCR probability at a specified treatment and pathway value.
        """

        row = np.array(
            [
                1.0,
                treatment_value,
                feature_value,
                (
                    treatment_value
                    * feature_value
                ),
            ],
            dtype=float,
        )

        return float(
            expit(
                row
                @ beta
            )
        )

    control_q25 = (
        predicted_probability(
            treatment_value=0.0,
            feature_value=q25_z,
        )
    )

    treated_q25 = (
        predicted_probability(
            treatment_value=1.0,
            feature_value=q25_z,
        )
    )

    control_q75 = (
        predicted_probability(
            treatment_value=0.0,
            feature_value=q75_z,
        )
    )

    treated_q75 = (
        predicted_probability(
            treatment_value=1.0,
            feature_value=q75_z,
        )
    )

    rd_q25 = float(
        treated_q25
        - control_q25
    )

    rd_q75 = float(
        treated_q75
        - control_q75
    )

    return InteractionModelResult(
        feature=(
            name
        ),

        n=int(
            len(
                x
            )
        ),

        converged=(
            converged
        ),

        feature_mean=(
            mean
        ),

        feature_sd=(
            sd
        ),

        intercept=float(
            beta[
                0
            ]
        ),

        treatment_coefficient=float(
            beta[
                1
            ]
        ),

        prognostic_coefficient=float(
            beta[
                2
            ]
        ),

        interaction_coefficient=(
            interaction_coefficient
        ),

        treatment_standard_error=float(
            standard_errors[
                1
            ]
        ),

        prognostic_standard_error=float(
            standard_errors[
                2
            ]
        ),

        interaction_standard_error=(
            interaction_se
        ),

        treatment_z=float(
            z_statistics[
                1
            ]
        ),

        prognostic_z=float(
            z_statistics[
                2
            ]
        ),

        interaction_z=float(
            z_statistics[
                3
            ]
        ),

        treatment_p_value=float(
            p_values[
                1
            ]
        ),

        prognostic_p_value=float(
            p_values[
                2
            ]
        ),

        interaction_p_value=float(
            p_values[
                3
            ]
        ),

        interaction_odds_ratio=(
            interaction_or
        ),

        interaction_or_ci_lower=(
            interaction_ci_lower
        ),

        interaction_or_ci_upper=(
            interaction_ci_upper
        ),

        feature_q25_z=(
            q25_z
        ),

        feature_q75_z=(
            q75_z
        ),

        pcr_probability_control_q25=(
            control_q25
        ),

        pcr_probability_treated_q25=(
            treated_q25
        ),

        risk_difference_q25=(
            rd_q25
        ),

        pcr_probability_control_q75=(
            control_q75
        ),

        pcr_probability_treated_q75=(
            treated_q75
        ),

        risk_difference_q75=(
            rd_q75
        ),

        risk_difference_contrast=float(
            rd_q75
            - rd_q25
        ),
    )


# =============================================================
# Full modifier discovery
# =============================================================


def discover_treatment_modifiers(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    fdr_threshold: float = 0.10,
    max_iter: int = 10000,
) -> ModifierDiscoveryResult:
    """
    Analyze every biological pathway independently for treatment interaction.
    """

    X = _validate_feature_matrix(
        X
    )

    treatment = _validate_binary_series(
        treatment,
        "treatment",
    )

    outcome = _validate_binary_series(
        outcome,
        "outcome",
    )

    if not (
        X.index.equals(
            treatment.index
        )
        and X.index.equals(
            outcome.index
        )
    ):
        raise ValueError(
            "X, treatment, and outcome must have identical patient order."
        )

    if not (
        0.0
        < fdr_threshold
        < 1.0
    ):
        raise ValueError(
            "fdr_threshold must be between 0 and 1."
        )

    records: list[
        dict[str, Any]
    ] = []

    failed_features: list[
        str
    ] = []

    for feature_name in X.columns:

        try:

            result = (
                fit_pathway_interaction(
                    X[
                        feature_name
                    ],
                    treatment,
                    outcome,
                    feature_name=(
                        str(
                            feature_name
                        )
                    ),
                    max_iter=max_iter,
                )
            )

            records.append(
                result.__dict__
            )

        except Exception:

            failed_features.append(
                str(
                    feature_name
                )
            )

    if len(
        records
    ) == 0:
        raise RuntimeError(
            "No pathway interaction models could be fitted."
        )

    table = pd.DataFrame(
        records
    )

    table[
        "interaction_fdr"
    ] = benjamini_hochberg(
        table[
            "interaction_p_value"
        ]
    )

    table[
        "prognostic_fdr"
    ] = benjamini_hochberg(
        table[
            "prognostic_p_value"
        ]
    )

    table[
        "interaction_direction"
    ] = np.where(
        table[
            "interaction_coefficient"
        ]
        > 0.0,
        (
            "greater_benefit_with_higher_pathway"
        ),
        (
            "greater_benefit_with_lower_pathway"
        ),
    )

    table[
        "nominal_interaction"
    ] = (
        table[
            "interaction_p_value"
        ]
        < 0.05
    )

    table[
        "fdr_significant_interaction"
    ] = (
        table[
            "interaction_fdr"
        ]
        < fdr_threshold
    )

    table[
        "nominal_prognostic"
    ] = (
        table[
            "prognostic_p_value"
        ]
        < 0.05
    )

    table[
        "absolute_interaction_coefficient"
    ] = (
        table[
            "interaction_coefficient"
        ]
        .abs()
    )

    table[
        "absolute_risk_difference_contrast"
    ] = (
        table[
            "risk_difference_contrast"
        ]
        .abs()
    )

    table = (
        table
        .sort_values(
            [
                "interaction_fdr",
                "interaction_p_value",
                "absolute_interaction_coefficient",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    table[
        "interaction_rank"
    ] = np.arange(
        1,
        len(
            table
        )
        + 1,
        dtype=int,
    )

    summary: dict[
        str,
        Any,
    ] = {
        "patients":
            int(
                len(
                    X
                )
            ),

        "features_requested":
            int(
                X.shape[
                    1
                ]
            ),

        "features_analyzed":
            int(
                len(
                    table
                )
            ),

        "features_failed":
            int(
                len(
                    failed_features
                )
            ),

        "failed_feature_names":
            tuple(
                failed_features
            ),

        "treatment_control_n":
            int(
                (
                    treatment
                    == 0
                ).sum()
            ),

        "treatment_active_n":
            int(
                (
                    treatment
                    == 1
                ).sum()
            ),

        "outcome_negative_n":
            int(
                (
                    outcome
                    == 0
                ).sum()
            ),

        "outcome_positive_n":
            int(
                (
                    outcome
                    == 1
                ).sum()
            ),

        "nominal_interaction_count":
            int(
                table[
                    "nominal_interaction"
                ].sum()
            ),

        "fdr_interaction_count":
            int(
                table[
                    "fdr_significant_interaction"
                ].sum()
            ),

        "nominal_prognostic_count":
            int(
                table[
                    "nominal_prognostic"
                ].sum()
            ),

        "fdr_threshold":
            float(
                fdr_threshold
            ),

        "all_models_converged":
            bool(
                table[
                    "converged"
                ].all()
            ),
    }

    return ModifierDiscoveryResult(
        modifier_table=(
            table
        ),
        summary=(
            summary
        ),
    )


# =============================================================
# NeoTRIP interface
# =============================================================


def run_neotrip_modifier_discovery(
    *,
    fdr_threshold: float = 0.10,
    max_iter: int = 10000,
) -> ModifierDiscoveryResult:
    """
    Run pathway-level modifier discovery on the real NeoTRIP HERMES dataset.
    """

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    dataset = (
        build_treatment_effect_dataset()
    )

    return discover_treatment_modifiers(
        dataset.X,
        dataset.T,
        dataset.Y,
        fdr_threshold=(
            fdr_threshold
        ),
        max_iter=(
            max_iter
        ),
    )


# =============================================================
# CLI
# =============================================================


def main() -> None:
    """
    Run exploratory biological treatment-effect modifier discovery in NeoTRIP.
    """

    print(
        "=== HERMES 2.0 BIOLOGICAL "
        "TREATMENT-EFFECT MODIFIER DISCOVERY ==="
    )

    print()

    result = (
        run_neotrip_modifier_discovery(
            fdr_threshold=0.10,
            max_iter=10000,
        )
    )

    print(
        "=== COHORT SUMMARY ==="
    )

    for key, value in (
        result.summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    display_columns = [
        "interaction_rank",
        "feature",
        "interaction_coefficient",
        "interaction_standard_error",
        "interaction_odds_ratio",
        "interaction_or_ci_lower",
        "interaction_or_ci_upper",
        "interaction_p_value",
        "interaction_fdr",
        "interaction_direction",
        "risk_difference_q25",
        "risk_difference_q75",
        "risk_difference_contrast",
        "prognostic_coefficient",
        "prognostic_p_value",
        "prognostic_fdr",
    ]

    print(
        "=== TOP CANDIDATE TREATMENT MODIFIERS ==="
    )

    print(
        result
        .modifier_table[
            display_columns
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()

    nominal = (
        result
        .modifier_table[
            result
            .modifier_table[
                "nominal_interaction"
            ]
        ]
    )

    print(
        "=== NOMINAL INTERACTIONS (P < 0.05) ==="
    )

    if nominal.empty:

        print(
            "None."
        )

    else:

        print(
            nominal[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    print()

    significant = (
        result
        .modifier_table[
            result
            .modifier_table[
                "fdr_significant_interaction"
            ]
        ]
    )

    print(
        "=== FDR-SIGNIFICANT INTERACTIONS ==="
    )

    if significant.empty:

        print(
            "None."
        )

    else:

        print(
            significant[
                display_columns
            ]
            .to_string(
                index=False
            )
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Interaction coefficients estimate pathway-dependent "
        "treatment-effect modification on the log-odds scale."
    )

    print(
        "Positive interaction coefficients imply greater predicted "
        "incremental treatment benefit as pathway activity increases."
    )

    print(
        "Negative interaction coefficients imply greater predicted "
        "incremental treatment benefit as pathway activity decreases."
    )

    print(
        "These analyses are exploratory and do not establish a "
        "validated predictive biomarker."
    )


if __name__ == "__main__":
    main()