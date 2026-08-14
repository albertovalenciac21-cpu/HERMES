"""
HERMES 2.0
Baseline Individualized Treatment-Effect Model
================================================

Purpose
-------
Estimate heterogeneous benefit from adding atezolizumab to chemotherapy
using biologically interpretable Hallmark pathway features.

Model
-----
The baseline model is a regularized logistic interaction model:

    logit P(Y = 1)
        = beta_0
        + beta_T * T
        + beta_X * X
        + gamma * (T x X)

where:

    X = baseline biological state
    T = randomized treatment
        0 = CT
        1 = CT/A
    Y = pathologic complete response
        0 = RD
        1 = pCR

For every patient the fitted model generates:

    P(pCR | CT)
    P(pCR | CT/A)

and:

    estimated ITE
        = P(pCR | CT/A)
        - P(pCR | CT)

IMPORTANT
---------
This is an in-sample engineering baseline.

The resulting treatment-effect estimates must NOT yet be interpreted as
validated individualized treatment effects or clinical biomarkers.

Subsequent HERMES stages will introduce:

    - cross-fitting
    - nested cross-validation
    - fold-specific scaling/preprocessing
    - tuning of regularization
    - permutation/null controls
    - uncertainty estimation
    - alternative treatment-effect learners
    - external validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class BaselineTreatmentEffectResult:
    """
    Results produced by the HERMES baseline treatment-effect model.
    """

    model: LogisticRegression
    scaler: StandardScaler

    feature_names: list[str]
    design_feature_names: list[str]

    treatment: pd.Series
    outcome: pd.Series

    observed_probability: pd.Series
    probability_control: pd.Series
    probability_treated: pd.Series

    ite: pd.Series

    treatment_coefficient: float

    biological_coefficients: pd.Series
    interaction_coefficients: pd.Series

    summary: dict


def _validate_binary_series(
    series: pd.Series,
    name: str,
) -> None:
    """
    Validate a patient-aligned binary vector.
    """

    if not isinstance(series, pd.Series):
        raise TypeError(
            f"{name} must be a pandas Series."
        )

    if series.isna().any():
        raise ValueError(
            f"{name} contains missing values."
        )

    unique_values = set(
        series.astype(int).unique().tolist()
    )

    if unique_values != {0, 1}:
        raise ValueError(
            f"{name} must contain exactly 0 and 1. "
            f"Found: {sorted(unique_values)}"
        )


def validate_model_inputs(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
) -> None:
    """
    Validate HERMES treatment-effect modeling inputs.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "X cannot be empty."
        )

    if X.shape[1] == 0:
        raise ValueError(
            "X must contain biological features."
        )

    if X.index.duplicated().any():
        raise ValueError(
            "Duplicate patient IDs detected in X."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            "Duplicate biological features detected."
        )

    if X.isna().any().any():
        raise ValueError(
            "Missing values detected in X."
        )

    values = X.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite biological feature "
            "values detected."
        )

    _validate_binary_series(
        treatment,
        "treatment",
    )

    _validate_binary_series(
        outcome,
        "outcome",
    )

    if not X.index.equals(
        treatment.index
    ):
        raise ValueError(
            "Treatment vector is not aligned "
            "with X."
        )

    if not X.index.equals(
        outcome.index
    ):
        raise ValueError(
            "Outcome vector is not aligned "
            "with X."
        )


def build_interaction_design(
    X_scaled: np.ndarray,
    treatment: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Construct the treatment-interaction design matrix.

    Layout:

        treatment
        biological main effects
        treatment x biological interactions

    For 50 Hallmark pathways:

        1 + 50 + 50 = 101 predictors
    """

    X_scaled = np.asarray(
        X_scaled,
        dtype=float,
    )

    treatment = np.asarray(
        treatment,
        dtype=float,
    ).reshape(-1, 1)

    if (
        X_scaled.shape[0]
        != treatment.shape[0]
    ):
        raise ValueError(
            "X and treatment contain "
            "different patient counts."
        )

    interactions = (
        X_scaled
        * treatment
    )

    design = np.column_stack(
        [
            treatment,
            X_scaled,
            interactions,
        ]
    )

    feature_names = [
        str(name)
        for name in feature_names
    ]

    interaction_names = [
        f"T_x_{name}"
        for name in feature_names
    ]

    design_names = (
        ["T"]
        + feature_names
        + interaction_names
    )

    return (
        design,
        design_names,
    )


def fit_baseline_treatment_effect_model(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    C: float = 0.1,
    max_iter: int = 10000,
) -> BaselineTreatmentEffectResult:
    """
    Fit the first HERMES 2.0 treatment-effect baseline.

    Parameters
    ----------
    X
        Biological feature matrix.

    treatment
        Binary randomized treatment:

            0 = CT
            1 = CT/A

    outcome
        Binary clinical outcome:

            0 = RD
            1 = pCR

    C
        Inverse L2 regularization strength.

        Smaller values imply stronger regularization.

    max_iter
        Maximum logistic-regression iterations.
    """

    validate_model_inputs(
        X,
        treatment,
        outcome,
    )

    treatment = (
        treatment
        .astype(int)
        .copy()
        .rename("T")
    )

    outcome = (
        outcome
        .astype(int)
        .copy()
        .rename("Y")
    )

    feature_names = [
        str(column)
        for column in X.columns
    ]

    # ---------------------------------------------------------
    # Standardize biological pathways
    #
    # NOTE:
    # This baseline uses the complete dataset for scaling.
    # Cross-fitted HERMES will fit scaling exclusively inside
    # training folds.
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = (
        scaler.fit_transform(
            X.to_numpy(
                dtype=float
            )
        )
    )

    # ---------------------------------------------------------
    # Observed-treatment design
    # ---------------------------------------------------------

    observed_design, design_names = (
        build_interaction_design(
            X_scaled,
            treatment.to_numpy(),
            feature_names,
        )
    )

    # ---------------------------------------------------------
    # Regularized logistic model
    # ---------------------------------------------------------

    model = LogisticRegression(
        penalty="l2",
        C=float(C),
        solver="lbfgs",
        max_iter=int(max_iter),
    )

    model.fit(
        observed_design,
        outcome.to_numpy(),
    )

    # ---------------------------------------------------------
    # Observed-treatment predictions
    # ---------------------------------------------------------

    observed_probability = pd.Series(
        model.predict_proba(
            observed_design
        )[:, 1],
        index=X.index,
        name="observed_pcr_probability",
    )

    # ---------------------------------------------------------
    # Counterfactual CT
    #
    # Every patient is assigned T = 0.
    # ---------------------------------------------------------

    control_treatment = np.zeros(
        X.shape[0],
        dtype=int,
    )

    control_design, _ = (
        build_interaction_design(
            X_scaled,
            control_treatment,
            feature_names,
        )
    )

    probability_control = pd.Series(
        model.predict_proba(
            control_design
        )[:, 1],
        index=X.index,
        name="pcr_probability_CT",
    )

    # ---------------------------------------------------------
    # Counterfactual CT/A
    #
    # Every patient is assigned T = 1.
    # ---------------------------------------------------------

    treated_treatment = np.ones(
        X.shape[0],
        dtype=int,
    )

    treated_design, _ = (
        build_interaction_design(
            X_scaled,
            treated_treatment,
            feature_names,
        )
    )

    probability_treated = pd.Series(
        model.predict_proba(
            treated_design
        )[:, 1],
        index=X.index,
        name="pcr_probability_CT_A",
    )

    # ---------------------------------------------------------
    # Individualized treatment effect
    # ---------------------------------------------------------

    ite = (
        probability_treated
        - probability_control
    ).rename(
        "estimated_ite"
    )

    # ---------------------------------------------------------
    # Coefficients
    # ---------------------------------------------------------

    coefficients = pd.Series(
        model.coef_[0],
        index=design_names,
        dtype=float,
    )

    treatment_coefficient = float(
        coefficients["T"]
    )

    biological_coefficients = (
        coefficients.loc[
            feature_names
        ]
        .copy()
        .rename(
            "biological_main_effect"
        )
    )

    interaction_names = [
        f"T_x_{name}"
        for name in feature_names
    ]

    interaction_coefficients = (
        coefficients.loc[
            interaction_names
        ]
        .copy()
    )

    interaction_coefficients.index = (
        feature_names
    )

    interaction_coefficients.name = (
        "treatment_interaction"
    )

    # ---------------------------------------------------------
    # Basic model audit
    # ---------------------------------------------------------

    treatment_counts = (
        treatment
        .value_counts()
        .sort_index()
        .to_dict()
    )

    outcome_counts = (
        outcome
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary = {
        "patients": int(
            X.shape[0]
        ),
        "biological_features": int(
            X.shape[1]
        ),
        "design_features": int(
            observed_design.shape[1]
        ),
        "control_patients": int(
            treatment_counts[0]
        ),
        "treated_patients": int(
            treatment_counts[1]
        ),
        "residual_disease": int(
            outcome_counts[0]
        ),
        "pcr": int(
            outcome_counts[1]
        ),
        "observed_pcr_rate": float(
            outcome.mean()
        ),
        "mean_predicted_CT_pcr": float(
            probability_control.mean()
        ),
        "mean_predicted_CT_A_pcr": float(
            probability_treated.mean()
        ),
        "mean_estimated_ite": float(
            ite.mean()
        ),
        "median_estimated_ite": float(
            ite.median()
        ),
        "minimum_estimated_ite": float(
            ite.min()
        ),
        "maximum_estimated_ite": float(
            ite.max()
        ),
        "standard_deviation_ite": float(
            ite.std()
        ),
        "fraction_positive_ite": float(
            (ite > 0).mean()
        ),
        "fraction_negative_ite": float(
            (ite < 0).mean()
        ),
        "regularization_C": float(C),
    }

    return BaselineTreatmentEffectResult(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        design_feature_names=(
            design_names
        ),
        treatment=treatment,
        outcome=outcome,
        observed_probability=(
            observed_probability
        ),
        probability_control=(
            probability_control
        ),
        probability_treated=(
            probability_treated
        ),
        ite=ite,
        treatment_coefficient=(
            treatment_coefficient
        ),
        biological_coefficients=(
            biological_coefficients
        ),
        interaction_coefficients=(
            interaction_coefficients
        ),
        summary=summary,
    )


def treatment_effect_table(
    result: BaselineTreatmentEffectResult,
) -> pd.DataFrame:
    """
    Return patient-level counterfactual estimates.
    """

    return pd.DataFrame(
        {
            "T": result.treatment,
            "Y": result.outcome,
            "observed_pcr_probability":
                result.observed_probability,
            "pcr_probability_CT":
                result.probability_control,
            "pcr_probability_CT_A":
                result.probability_treated,
            "estimated_ite":
                result.ite,
        }
    )


def interaction_table(
    result: BaselineTreatmentEffectResult,
) -> pd.DataFrame:
    """
    Return biological main effects and treatment interactions.

    Positive treatment interaction:
        Higher pathway state is associated with greater modeled
        benefit from CT/A relative to CT.

    Negative treatment interaction:
        Higher pathway state is associated with lower modeled
        benefit from CT/A relative to CT.

    These are exploratory model coefficients and are NOT yet
    validated treatment-predictive biomarkers.
    """

    table = pd.DataFrame(
        {
            "biological_main_effect":
                result.biological_coefficients,
            "treatment_interaction":
                result.interaction_coefficients,
        }
    )

    table[
        "absolute_treatment_interaction"
    ] = (
        table[
            "treatment_interaction"
        ]
        .abs()
    )

    return table.sort_values(
        "absolute_treatment_interaction",
        ascending=False,
    )


def main() -> None:
    """
    Run the engineering baseline on the current NeoTRIP dataset.
    """

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    dataset = (
        build_treatment_effect_dataset()
    )

    result = (
        fit_baseline_treatment_effect_model(
            X=dataset.X,
            treatment=dataset.T,
            outcome=dataset.Y,
            C=0.1,
        )
    )

    print(
        "=== HERMES 2.0 "
        "BASELINE TREATMENT-EFFECT MODEL ==="
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
        "Treatment coefficient:"
    )

    print(
        result.treatment_coefficient
    )

    print()

    print(
        "Top candidate treatment-effect modifiers:"
    )

    print(
        interaction_table(
            result
        )
        .head(10)
        .to_string()
    )

    print()

    print(
        "First patient-level "
        "counterfactual estimates:"
    )

    print(
        treatment_effect_table(
            result
        )
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()