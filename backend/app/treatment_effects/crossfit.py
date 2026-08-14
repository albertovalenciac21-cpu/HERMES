"""
HERMES 2.0
Cross-Fitted Treatment-Effect Estimator
=======================================

Purpose
-------
Generate patient-level treatment-effect estimates using models that were
never trained on the patient being predicted.

For patient i in held-out fold k:

    mu0_i = P(Y=1 | T=0, X_i; model trained on D_-k)
    mu1_i = P(Y=1 | T=1, X_i; model trained on D_-k)

and:

    tau_i = mu1_i - mu0_i

This is a major upgrade over the initial in-sample HERMES baseline.

Cross-fitting procedure
-----------------------
1. Stratify patients using joint treatment x outcome status.
2. Divide the cohort into K folds.
3. For each fold:
       - fit scaler on training patients only
       - fit the regularized treatment-interaction model on training only
       - predict held-out patients only
4. Assemble one out-of-fold estimate for every patient.

The biological feature matrix is assumed to have already been constructed.
Later HERMES versions will move additional data-dependent preprocessing
inside the folds as well.

IMPORTANT
---------
Cross-fitting reduces training-set optimism but does not by itself prove
causal heterogeneity or clinical utility.

Further validation will include:
    - nested tuning of regularization
    - repeated cross-fitting
    - permutation/null controls
    - calibration
    - treatment-effect ranking metrics
    - uncertainty estimates
    - alternative learners
    - external validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from backend.app.treatment_effects.baseline_model import (
    build_interaction_design,
    validate_model_inputs,
)


@dataclass
class CrossFitTreatmentEffectResult:
    """
    Results from HERMES cross-fitted treatment-effect estimation.
    """

    fold: pd.Series

    treatment: pd.Series
    outcome: pd.Series

    observed_probability: pd.Series
    probability_control: pd.Series
    probability_treated: pd.Series

    ite: pd.Series

    fold_summary: pd.DataFrame
    summary: dict[str, Any]

    @property
    def n_patients(self) -> int:
        return int(len(self.ite))


def build_joint_strata(
    treatment: pd.Series,
    outcome: pd.Series,
) -> pd.Series:
    """
    Create joint treatment x outcome stratification labels.

    Groups:

        T0_Y0 = CT + RD
        T0_Y1 = CT + pCR
        T1_Y0 = CT/A + RD
        T1_Y1 = CT/A + pCR
    """

    if not treatment.index.equals(
        outcome.index
    ):
        raise ValueError(
            "Treatment and outcome indices "
            "must be aligned."
        )

    strata = (
        "T"
        + treatment.astype(int).astype(str)
        + "_Y"
        + outcome.astype(int).astype(str)
    )

    strata.name = "joint_stratum"

    return strata


def validate_crossfit_configuration(
    treatment: pd.Series,
    outcome: pd.Series,
    n_splits: int,
) -> None:
    """
    Ensure requested cross-fitting is feasible.
    """

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    strata = build_joint_strata(
        treatment,
        outcome,
    )

    counts = strata.value_counts()

    if counts.min() < n_splits:
        raise ValueError(
            "At least one treatment x outcome "
            "stratum contains fewer observations "
            f"than n_splits={n_splits}. "
            f"Counts: {counts.to_dict()}"
        )


def _fit_fold_model(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    Y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    C: float,
    max_iter: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Fit one training fold and predict one held-out fold.

    Returns
    -------
    observed_probability
        Probability under each held-out patient's observed treatment.

    probability_control
        Counterfactual probability if all held-out patients receive CT.

    probability_treated
        Counterfactual probability if all held-out patients receive CT/A.
    """

    # ---------------------------------------------------------
    # Fold-specific scaling
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train.to_numpy(dtype=float)
    )

    X_test_scaled = scaler.transform(
        X_test.to_numpy(dtype=float)
    )

    feature_names = [
        str(column)
        for column in X_train.columns
    ]

    # ---------------------------------------------------------
    # Training design
    # ---------------------------------------------------------

    train_design, _ = (
        build_interaction_design(
            X_train_scaled,
            T_train.to_numpy(dtype=int),
            feature_names,
        )
    )

    # ---------------------------------------------------------
    # Regularized model
    # ---------------------------------------------------------

    model = LogisticRegression(
        C=float(C),
        solver="lbfgs",
        max_iter=int(max_iter),
    )

    model.fit(
        train_design,
        Y_train.to_numpy(dtype=int),
    )

    # ---------------------------------------------------------
    # Held-out observed-treatment design
    # ---------------------------------------------------------

    observed_design, _ = (
        build_interaction_design(
            X_test_scaled,
            T_test := np.asarray(
                T_train.iloc[:0],
            ),
            feature_names,
        )
    )

    # The placeholder above is immediately replaced below.
    # Keeping prediction construction explicit prevents any
    # accidental use of training-treatment values.

    del observed_design
    del T_test

    return (
        model,
        scaler,
        X_test_scaled,
    )


def crossfit_treatment_effect_model(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    n_splits: int = 5,
    C: float = 0.1,
    max_iter: int = 10000,
    random_state: int = 42,
) -> CrossFitTreatmentEffectResult:
    """
    Generate cross-fitted HERMES treatment-effect estimates.

    Every patient is predicted exactly once by a model that did not
    include that patient during fitting.
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

    validate_crossfit_configuration(
        treatment,
        outcome,
        n_splits,
    )

    joint_strata = build_joint_strata(
        treatment,
        outcome,
    )

    # ---------------------------------------------------------
    # Output containers
    # ---------------------------------------------------------

    fold_assignment = pd.Series(
        index=X.index,
        dtype="int64",
        name="fold",
    )

    observed_probability = pd.Series(
        index=X.index,
        dtype=float,
        name="observed_pcr_probability",
    )

    probability_control = pd.Series(
        index=X.index,
        dtype=float,
        name="pcr_probability_CT",
    )

    probability_treated = pd.Series(
        index=X.index,
        dtype=float,
        name="pcr_probability_CT_A",
    )

    # ---------------------------------------------------------
    # Stratified folds
    # ---------------------------------------------------------

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_records: list[dict[str, Any]] = []

    feature_names = [
        str(column)
        for column in X.columns
    ]

    for fold_number, (
        train_positions,
        test_positions,
    ) in enumerate(
        splitter.split(
            X,
            joint_strata,
        ),
        start=1,
    ):

        train_ids = X.index[
            train_positions
        ]

        test_ids = X.index[
            test_positions
        ]

        X_train = X.loc[
            train_ids
        ]

        X_test = X.loc[
            test_ids
        ]

        T_train = treatment.loc[
            train_ids
        ]

        T_test = treatment.loc[
            test_ids
        ]

        Y_train = outcome.loc[
            train_ids
        ]

        Y_test = outcome.loc[
            test_ids
        ]

        # -----------------------------------------------------
        # Fit scaler on training patients only
        # -----------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train.to_numpy(
                    dtype=float
                )
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test.to_numpy(
                    dtype=float
                )
            )
        )

        # -----------------------------------------------------
        # Training interaction design
        # -----------------------------------------------------

        train_design, _ = (
            build_interaction_design(
                X_train_scaled,
                T_train.to_numpy(
                    dtype=int
                ),
                feature_names,
            )
        )

        # -----------------------------------------------------
        # Fit fold model
        # -----------------------------------------------------

        model = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=int(max_iter),
        )

        model.fit(
            train_design,
            Y_train.to_numpy(
                dtype=int
            ),
        )

        # -----------------------------------------------------
        # Observed-treatment predictions
        # -----------------------------------------------------

        observed_design, _ = (
            build_interaction_design(
                X_test_scaled,
                T_test.to_numpy(
                    dtype=int
                ),
                feature_names,
            )
        )

        observed_pred = (
            model.predict_proba(
                observed_design
            )[:, 1]
        )

        # -----------------------------------------------------
        # Counterfactual CT predictions
        # -----------------------------------------------------

        control_design, _ = (
            build_interaction_design(
                X_test_scaled,
                np.zeros(
                    len(test_ids),
                    dtype=int,
                ),
                feature_names,
            )
        )

        control_pred = (
            model.predict_proba(
                control_design
            )[:, 1]
        )

        # -----------------------------------------------------
        # Counterfactual CT/A predictions
        # -----------------------------------------------------

        treated_design, _ = (
            build_interaction_design(
                X_test_scaled,
                np.ones(
                    len(test_ids),
                    dtype=int,
                ),
                feature_names,
            )
        )

        treated_pred = (
            model.predict_proba(
                treated_design
            )[:, 1]
        )

        # -----------------------------------------------------
        # Store held-out predictions
        # -----------------------------------------------------

        fold_assignment.loc[
            test_ids
        ] = fold_number

        observed_probability.loc[
            test_ids
        ] = observed_pred

        probability_control.loc[
            test_ids
        ] = control_pred

        probability_treated.loc[
            test_ids
        ] = treated_pred

        # -----------------------------------------------------
        # Fold diagnostics
        # -----------------------------------------------------

        try:
            fold_auc = float(
                roc_auc_score(
                    Y_test,
                    observed_pred,
                )
            )
        except ValueError:
            fold_auc = np.nan

        fold_brier = float(
            brier_score_loss(
                Y_test,
                observed_pred,
            )
        )

        fold_records.append(
            {
                "fold": fold_number,
                "train_n": int(
                    len(train_ids)
                ),
                "test_n": int(
                    len(test_ids)
                ),
                "train_CT": int(
                    (T_train == 0).sum()
                ),
                "train_CT_A": int(
                    (T_train == 1).sum()
                ),
                "test_CT": int(
                    (T_test == 0).sum()
                ),
                "test_CT_A": int(
                    (T_test == 1).sum()
                ),
                "test_RD": int(
                    (Y_test == 0).sum()
                ),
                "test_pCR": int(
                    (Y_test == 1).sum()
                ),
                "observed_auc": (
                    fold_auc
                ),
                "observed_brier": (
                    fold_brier
                ),
                "mean_control_prediction": float(
                    np.mean(
                        control_pred
                    )
                ),
                "mean_treated_prediction": float(
                    np.mean(
                        treated_pred
                    )
                ),
                "mean_ite": float(
                    np.mean(
                        treated_pred
                        - control_pred
                    )
                ),
            }
        )

    # ---------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------

    prediction_objects = [
        fold_assignment,
        observed_probability,
        probability_control,
        probability_treated,
    ]

    for obj in prediction_objects:
        if obj.isna().any():
            raise RuntimeError(
                "Cross-fitting failed to generate "
                "exactly one prediction for every patient."
            )

    if not (
        fold_assignment
        .between(
            1,
            n_splits,
        )
        .all()
    ):
        raise RuntimeError(
            "Invalid fold assignment detected."
        )

    # ---------------------------------------------------------
    # Cross-fitted ITE
    # ---------------------------------------------------------

    ite = (
        probability_treated
        - probability_control
    ).rename(
        "crossfitted_ite"
    )

    # ---------------------------------------------------------
    # Overall predictive diagnostics
    # ---------------------------------------------------------

    overall_auc = float(
        roc_auc_score(
            outcome,
            observed_probability,
        )
    )

    overall_brier = float(
        brier_score_loss(
            outcome,
            observed_probability,
        )
    )

    # ---------------------------------------------------------
    # Observed randomized arm effect
    # ---------------------------------------------------------

    observed_control_rate = float(
        outcome[
            treatment == 0
        ].mean()
    )

    observed_treated_rate = float(
        outcome[
            treatment == 1
        ].mean()
    )

    observed_risk_difference = (
        observed_treated_rate
        - observed_control_rate
    )

    fold_summary = (
        pd.DataFrame(
            fold_records
        )
        .set_index("fold")
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary: dict[str, Any] = {
        "patients": int(
            X.shape[0]
        ),
        "biological_features": int(
            X.shape[1]
        ),
        "n_splits": int(
            n_splits
        ),
        "regularization_C": float(
            C
        ),
        "observed_CT_pcr_rate": (
            observed_control_rate
        ),
        "observed_CT_A_pcr_rate": (
            observed_treated_rate
        ),
        "observed_risk_difference": float(
            observed_risk_difference
        ),
        "crossfitted_observed_auc": (
            overall_auc
        ),
        "crossfitted_observed_brier": (
            overall_brier
        ),
        "mean_predicted_CT_pcr": float(
            probability_control.mean()
        ),
        "mean_predicted_CT_A_pcr": float(
            probability_treated.mean()
        ),
        "mean_crossfitted_ite": float(
            ite.mean()
        ),
        "median_crossfitted_ite": float(
            ite.median()
        ),
        "minimum_crossfitted_ite": float(
            ite.min()
        ),
        "maximum_crossfitted_ite": float(
            ite.max()
        ),
        "standard_deviation_crossfitted_ite": float(
            ite.std()
        ),
        "fraction_positive_crossfitted_ite": float(
            (ite > 0).mean()
        ),
        "fraction_negative_crossfitted_ite": float(
            (ite < 0).mean()
        ),
    }

    return CrossFitTreatmentEffectResult(
        fold=fold_assignment.astype(
            int
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
        fold_summary=fold_summary,
        summary=summary,
    )


def crossfit_patient_table(
    result: CrossFitTreatmentEffectResult,
) -> pd.DataFrame:
    """
    Return one out-of-fold treatment-effect row per patient.
    """

    return pd.DataFrame(
        {
            "fold": result.fold,
            "T": result.treatment,
            "Y": result.outcome,
            "observed_pcr_probability":
                result.observed_probability,
            "pcr_probability_CT":
                result.probability_control,
            "pcr_probability_CT_A":
                result.probability_treated,
            "crossfitted_ite":
                result.ite,
        }
    )


def main() -> None:
    """
    Run five-fold HERMES cross-fitting on NeoTRIP.
    """

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    dataset = (
        build_treatment_effect_dataset()
    )

    result = (
        crossfit_treatment_effect_model(
            X=dataset.X,
            treatment=dataset.T,
            outcome=dataset.Y,
            n_splits=5,
            C=0.1,
            random_state=42,
        )
    )

    print(
        "=== HERMES 2.0 "
        "CROSS-FITTED TREATMENT-EFFECT MODEL ==="
    )

    print()

    for key, value in (
        result.summary.items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print("Fold diagnostics:")

    print(
        result.fold_summary
        .to_string()
    )

    print()

    print(
        "First cross-fitted "
        "patient estimates:"
    )

    print(
        crossfit_patient_table(
            result
        )
        .head(10)
        .to_string()
    )


if __name__ == "__main__":
    main()