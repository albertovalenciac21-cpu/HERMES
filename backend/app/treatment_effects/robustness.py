"""
HERMES 2.0
Treatment-Effect Robustness and Sensitivity Analysis
=====================================================

Purpose
-------
Evaluate whether HERMES individualized treatment-effect estimates remain
stable under reasonable perturbations of the analysis pipeline.

The robustness framework evaluates:

1. Regularization sensitivity
2. Cross-fitting fold sensitivity
3. Patient-level treatment-effect ranking stability
4. Patient-level treatment-effect sign stability
5. Cohort-level treatment-effect stability
6. Prediction-performance stability
7. Biological modifier directional stability under stratified subsampling

This module deliberately separates:

    repeated-fit uncertainty
        -> variability across repeated cross-fitting partitions

from:

    robustness / sensitivity
        -> variability across reasonable modeling assumptions

and from:

    external validation
        -> performance in an independent cohort

These are related but distinct scientific questions.

IMPORTANT
---------
Robustness categories generated here are research-engineering summaries.

They do NOT establish:

    - causal identification of heterogeneous treatment effects
    - clinical treatment recommendations
    - validated predictive biomarkers
    - external generalizability

The goal is to determine whether HERMES conclusions are fragile or stable
with respect to reasonable analytic perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from backend.app.treatment_effects.modifier_discovery import (
    discover_treatment_modifiers,
)
from backend.app.treatment_effects.repeated_crossfit import (
    generate_random_states,
    repeated_crossfit_treatment_effect_model,
)


@dataclass
class TreatmentEffectRobustnessResult:
    scenario_summary: pd.DataFrame
    patient_ite_by_scenario: pd.DataFrame
    patient_robustness: pd.DataFrame
    pairwise_scenario_comparison: pd.DataFrame
    modifier_robustness: pd.DataFrame
    summary: dict[str, Any]


def _validate_feature_matrix(
    X: pd.DataFrame,
) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")

    if X.empty:
        raise ValueError("X cannot be empty.")

    values = X.astype(float)

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("X contains non-finite values.")

    if values.columns.duplicated().any():
        raise ValueError("X contains duplicate feature names.")

    return values


def _validate_binary_series(
    values: pd.Series,
    name: str,
) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{name} must be a pandas Series.")

    if values.isna().any():
        raise ValueError(f"{name} contains missing values.")

    numeric = values.astype(int)
    unique = set(numeric.unique())

    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0 and 1.")

    if len(unique) != 2:
        raise ValueError(f"{name} must contain both binary classes.")

    return numeric


def _validate_positive_float_sequence(
    values: Sequence[float],
    name: str,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)

    if len(result) == 0:
        raise ValueError(f"{name} cannot be empty.")

    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values.")

    for value in result:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must contain positive finite values."
            )

    return result


def _validate_split_sequence(
    values: Sequence[int],
) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)

    if len(result) == 0:
        raise ValueError("n_splits_values cannot be empty.")

    if len(set(result)) != len(result):
        raise ValueError(
            "n_splits_values must contain unique values."
        )

    if any(value < 2 for value in result):
        raise ValueError(
            "Every cross-fitting fold count must be at least 2."
        )

    return result


def _validate_alignment(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
) -> None:
    if not (
        X.index.equals(treatment.index)
        and X.index.equals(outcome.index)
    ):
        raise ValueError(
            "X, treatment, and outcome must have identical patient order."
        )


def _scenario_name(
    C: float,
    n_splits: int,
) -> str:
    c_string = (
        f"{C:g}"
        .replace(".", "p")
        .replace("-", "m")
    )

    return f"C_{c_string}__splits_{n_splits}"


def _safe_spearman(
    a: pd.Series,
    b: pd.Series,
) -> float:
    a_values = a.to_numpy(dtype=float)
    b_values = b.to_numpy(dtype=float)

    if (
        np.std(a_values) == 0.0
        or np.std(b_values) == 0.0
    ):
        return float("nan")

    value = spearmanr(
        a_values,
        b_values,
    ).statistic

    return float(value)


def _top_fraction_membership(
    values: pd.Series,
    fraction: float,
) -> set[Any]:
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            "fraction must be between 0 and 1."
        )

    n = len(values)

    count = max(
        1,
        int(np.ceil(n * fraction)),
    )

    return set(
        values
        .sort_values(ascending=False)
        .head(count)
        .index
    )


def _set_overlap(
    first: set[Any],
    second: set[Any],
) -> float:
    if len(first) == 0:
        return float("nan")

    return float(
        len(first & second)
        / len(first)
    )


def compare_scenarios_pairwise(
    patient_ite_by_scenario: pd.DataFrame,
    *,
    top_fraction: float = 0.25,
) -> pd.DataFrame:
    if not isinstance(
        patient_ite_by_scenario,
        pd.DataFrame,
    ):
        raise TypeError(
            "patient_ite_by_scenario must be a pandas DataFrame."
        )

    if patient_ite_by_scenario.shape[1] < 2:
        raise ValueError(
            "At least two sensitivity scenarios are required."
        )

    if patient_ite_by_scenario.isna().any().any():
        raise ValueError(
            "patient_ite_by_scenario contains missing values."
        )

    records: list[dict[str, Any]] = []

    for scenario_a, scenario_b in combinations(
        patient_ite_by_scenario.columns,
        2,
    ):
        first = patient_ite_by_scenario[scenario_a]
        second = patient_ite_by_scenario[scenario_b]

        correlation = _safe_spearman(
            first,
            second,
        )

        first_top = _top_fraction_membership(
            first,
            top_fraction,
        )

        second_top = _top_fraction_membership(
            second,
            top_fraction,
        )

        top_overlap = _set_overlap(
            first_top,
            second_top,
        )

        first_sign = np.sign(
            first.to_numpy(dtype=float)
        )

        second_sign = np.sign(
            second.to_numpy(dtype=float)
        )

        sign_agreement = float(
            np.mean(
                first_sign
                == second_sign
            )
        )

        absolute_difference = (
            first
            - second
        ).abs()

        records.append(
            {
                "scenario_a": str(scenario_a),
                "scenario_b": str(scenario_b),
                "spearman_ite": correlation,
                "top_fraction": float(top_fraction),
                "top_patient_overlap": top_overlap,
                "sign_agreement": sign_agreement,
                "mean_absolute_ite_difference": float(
                    absolute_difference.mean()
                ),
                "maximum_absolute_ite_difference": float(
                    absolute_difference.max()
                ),
            }
        )

    return pd.DataFrame(records)


def build_patient_robustness_table(
    patient_ite_by_scenario: pd.DataFrame,
) -> pd.DataFrame:
    if patient_ite_by_scenario.empty:
        raise ValueError(
            "patient_ite_by_scenario cannot be empty."
        )

    if patient_ite_by_scenario.shape[1] < 2:
        raise ValueError(
            "At least two scenarios are required."
        )

    matrix = patient_ite_by_scenario.astype(float)

    if not np.isfinite(
        matrix.to_numpy()
    ).all():
        raise ValueError(
            "patient_ite_by_scenario contains non-finite values."
        )

    table = pd.DataFrame(
        index=matrix.index
    )

    table.index.name = matrix.index.name

    table["mean_ite"] = matrix.mean(axis=1)
    table["median_ite"] = matrix.median(axis=1)

    table["ite_sensitivity_sd"] = matrix.std(
        axis=1,
        ddof=1,
    )

    table["minimum_ite"] = matrix.min(axis=1)
    table["maximum_ite"] = matrix.max(axis=1)

    table["ite_sensitivity_range"] = (
        table["maximum_ite"]
        - table["minimum_ite"]
    )

    table["fraction_positive"] = (
        matrix > 0.0
    ).mean(axis=1)

    table["fraction_negative"] = (
        matrix < 0.0
    ).mean(axis=1)

    table["fraction_zero"] = (
        matrix == 0.0
    ).mean(axis=1)

    table["sign_stability"] = (
        table[
            [
                "fraction_positive",
                "fraction_negative",
            ]
        ]
        .max(axis=1)
    )

    table["consensus_direction"] = np.select(
        [
            table["fraction_positive"] > 0.5,
            table["fraction_negative"] > 0.5,
        ],
        [
            "benefit",
            "harm",
        ],
        default="uncertain",
    )

    ranks = matrix.rank(
        axis=0,
        method="average",
        ascending=False,
    )

    table["mean_benefit_rank"] = ranks.mean(axis=1)
    table["median_benefit_rank"] = ranks.median(axis=1)

    table["benefit_rank_sd"] = ranks.std(
        axis=1,
        ddof=1,
    )

    table["best_benefit_rank"] = ranks.min(axis=1)
    table["worst_benefit_rank"] = ranks.max(axis=1)

    table["benefit_rank_range"] = (
        table["worst_benefit_rank"]
        - table["best_benefit_rank"]
    )

    n_patients = len(table)

    if n_patients > 1:
        table["normalized_rank_sd"] = (
            table["benefit_rank_sd"]
            / (n_patients - 1)
        )
    else:
        table["normalized_rank_sd"] = 0.0

    safe_sd = (
        table["ite_sensitivity_sd"]
        .replace(
            0.0,
            np.nan,
        )
    )

    table["sensitivity_signal_ratio"] = (
        table["mean_ite"].abs()
        / safe_sd
    )

    zero_sd = (
        table["ite_sensitivity_sd"]
        == 0.0
    )

    nonzero_mean = (
        table["mean_ite"]
        != 0.0
    )

    table.loc[
        zero_sd & nonzero_mean,
        "sensitivity_signal_ratio",
    ] = np.inf

    table.loc[
        zero_sd & ~nonzero_mean,
        "sensitivity_signal_ratio",
    ] = 0.0

    robust_sign = (
        table["sign_stability"]
        >= 0.90
    )

    moderate_sign = (
        table["sign_stability"]
        >= 0.75
    )

    low_rank_variability = (
        table["normalized_rank_sd"]
        <= 0.10
    )

    moderate_rank_variability = (
        table["normalized_rank_sd"]
        <= 0.20
    )

    adequate_signal = (
        table["sensitivity_signal_ratio"]
        >= 1.0
    )

    table["robustness_state"] = "unstable"

    table.loc[
        moderate_sign
        & moderate_rank_variability,
        "robustness_state",
    ] = "moderate"

    table.loc[
        robust_sign
        & low_rank_variability
        & adequate_signal,
        "robustness_state",
    ] = "robust"

    return table


def run_model_sensitivity_grid(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    C_values: Sequence[float] = (
        0.03,
        0.10,
        0.30,
    ),
    n_splits_values: Sequence[int] = (
        4,
        5,
        6,
    ),
    n_repeats: int = 5,
    max_iter: int = 10000,
    base_random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    X = _validate_feature_matrix(X)

    treatment = _validate_binary_series(
        treatment,
        "treatment",
    )

    outcome = _validate_binary_series(
        outcome,
        "outcome",
    )

    _validate_alignment(
        X,
        treatment,
        outcome,
    )

    C_values = _validate_positive_float_sequence(
        C_values,
        "C_values",
    )

    n_splits_values = _validate_split_sequence(
        n_splits_values
    )

    if n_repeats < 2:
        raise ValueError(
            "n_repeats must be at least 2."
        )

    shared_random_states = generate_random_states(
        n_repeats,
        base_random_state=base_random_state,
    )

    scenario_records: list[
        dict[str, Any]
    ] = []

    patient_columns: dict[
        str,
        pd.Series,
    ] = {}

    for C in C_values:
        for n_splits in n_splits_values:
            name = _scenario_name(
                C,
                n_splits,
            )

            result = (
                repeated_crossfit_treatment_effect_model(
                    X=X,
                    treatment=treatment,
                    outcome=outcome,
                    n_repeats=n_repeats,
                    n_splits=n_splits,
                    C=C,
                    max_iter=max_iter,
                    random_states=shared_random_states,
                )
            )

            patient_ite = (
                result
                .patient_summary[
                    "mean_ite"
                ]
                .copy()
            )

            patient_columns[
                name
            ] = patient_ite

            scenario_records.append(
                {
                    "scenario": name,
                    "regularization_C": float(C),
                    "n_splits": int(n_splits),
                    "n_repeats": int(n_repeats),
                    "cohort_mean_ite": float(
                        result.summary[
                            "cohort_mean_ite"
                        ]
                    ),
                    "cohort_median_ite": float(
                        result.summary[
                            "cohort_median_ite"
                        ]
                    ),
                    "minimum_patient_mean_ite": float(
                        result.summary[
                            "minimum_patient_mean_ite"
                        ]
                    ),
                    "maximum_patient_mean_ite": float(
                        result.summary[
                            "maximum_patient_mean_ite"
                        ]
                    ),
                    "mean_patient_ite_sd": float(
                        result.summary[
                            "mean_patient_ite_sd"
                        ]
                    ),
                    "fraction_sign_stability_ge_90pct": float(
                        result.summary[
                            "fraction_sign_stability_ge_90pct"
                        ]
                    ),
                    "fraction_unanimous_sign": float(
                        result.summary[
                            "fraction_unanimous_sign"
                        ]
                    ),
                    "fraction_consensus_benefit": float(
                        result.summary[
                            "fraction_consensus_benefit"
                        ]
                    ),
                    "fraction_consensus_harm": float(
                        result.summary[
                            "fraction_consensus_harm"
                        ]
                    ),
                    "mean_repeat_oof_auc": float(
                        result.summary[
                            "mean_repeat_oof_auc"
                        ]
                    ),
                    "std_repeat_oof_auc": float(
                        result.summary[
                            "std_repeat_oof_auc"
                        ]
                    ),
                    "mean_repeat_oof_brier": float(
                        result.summary[
                            "mean_repeat_oof_brier"
                        ]
                    ),
                    "std_repeat_oof_brier": float(
                        result.summary[
                            "std_repeat_oof_brier"
                        ]
                    ),
                }
            )

    scenario_summary = (
        pd.DataFrame(
            scenario_records
        )
        .set_index(
            "scenario"
        )
    )

    patient_ite_by_scenario = pd.DataFrame(
        patient_columns,
        index=X.index,
    )

    if patient_ite_by_scenario.shape != (
        len(X),
        len(C_values)
        * len(n_splits_values),
    ):
        raise RuntimeError(
            "Sensitivity grid produced an unexpected patient × scenario matrix."
        )

    if patient_ite_by_scenario.isna().any().any():
        raise RuntimeError(
            "Sensitivity grid produced missing patient-level ITE estimates."
        )

    return (
        scenario_summary,
        patient_ite_by_scenario,
    )


def stratified_subsample_index(
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    fraction: float = 0.80,
    random_state: int = 42,
) -> pd.Index:
    treatment = _validate_binary_series(
        treatment,
        "treatment",
    )

    outcome = _validate_binary_series(
        outcome,
        "outcome",
    )

    if not treatment.index.equals(
        outcome.index
    ):
        raise ValueError(
            "treatment and outcome must have identical patient order."
        )

    if not 0.50 <= fraction < 1.0:
        raise ValueError(
            "fraction must be between 0.50 and 1.0."
        )

    rng = np.random.default_rng(
        random_state
    )

    strata = pd.DataFrame(
        {
            "T": treatment,
            "Y": outcome,
        },
        index=treatment.index,
    )

    selected: list[Any] = []

    for _, group in strata.groupby(
        [
            "T",
            "Y",
        ],
        sort=True,
    ):
        group_index = group.index.to_numpy()

        target_n = int(
            np.floor(
                len(group_index)
                * fraction
            )
        )

        target_n = max(
            2,
            target_n,
        )

        target_n = min(
            target_n,
            len(group_index),
        )

        sampled = rng.choice(
            group_index,
            size=target_n,
            replace=False,
        )

        selected.extend(
            sampled.tolist()
        )

    selected_set = set(selected)

    ordered = [
        patient
        for patient in treatment.index
        if patient in selected_set
    ]

    return pd.Index(
        ordered,
        name=treatment.index.name,
    )


def run_modifier_robustness(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    n_perturbations: int = 5,
    subsample_fraction: float = 0.80,
    fdr_threshold: float = 0.10,
    max_iter: int = 10000,
    base_random_state: int = 2026,
) -> pd.DataFrame:
    X = _validate_feature_matrix(X)

    treatment = _validate_binary_series(
        treatment,
        "treatment",
    )

    outcome = _validate_binary_series(
        outcome,
        "outcome",
    )

    _validate_alignment(
        X,
        treatment,
        outcome,
    )

    if n_perturbations < 2:
        raise ValueError(
            "n_perturbations must be at least 2."
        )

    full_result = discover_treatment_modifiers(
        X,
        treatment,
        outcome,
        fdr_threshold=fdr_threshold,
        max_iter=max_iter,
    )

    reference = (
        full_result
        .modifier_table
        .set_index(
            "feature"
        )
    )

    rng = np.random.default_rng(
        base_random_state
    )

    perturbation_seeds = generate_random_states(
        n_perturbations,
        base_random_state=base_random_state,
    )

    coefficient_columns: dict[
        str,
        pd.Series,
    ] = {}

    rank_columns: dict[
        str,
        pd.Series,
    ] = {}

    pvalue_columns: dict[
        str,
        pd.Series,
    ] = {}

    convergence_columns: dict[
        str,
        pd.Series,
    ] = {}

    for perturbation_number, seed in enumerate(
        perturbation_seeds,
        start=1,
    ):
        index = stratified_subsample_index(
            treatment,
            outcome,
            fraction=subsample_fraction,
            random_state=int(seed),
        )

        perturbation = discover_treatment_modifiers(
            X.loc[index],
            treatment.loc[index],
            outcome.loc[index],
            fdr_threshold=fdr_threshold,
            max_iter=max_iter,
        )

        table = (
            perturbation
            .modifier_table
            .set_index(
                "feature"
            )
            .reindex(
                reference.index
            )
        )

        column_name = (
            f"perturbation_"
            f"{perturbation_number:03d}"
        )

        coefficient_columns[
            column_name
        ] = table[
            "interaction_coefficient"
        ]

        rank_columns[
            column_name
        ] = table[
            "interaction_rank"
        ]

        pvalue_columns[
            column_name
        ] = table[
            "interaction_p_value"
        ]

        convergence_columns[
            column_name
        ] = table[
            "converged"
        ].astype(float)

    coefficients = pd.DataFrame(
        coefficient_columns,
        index=reference.index,
    )

    ranks = pd.DataFrame(
        rank_columns,
        index=reference.index,
    )

    pvalues = pd.DataFrame(
        pvalue_columns,
        index=reference.index,
    )

    convergence = pd.DataFrame(
        convergence_columns,
        index=reference.index,
    )

    result = pd.DataFrame(
        index=reference.index
    )

    result[
        "full_interaction_coefficient"
    ] = reference[
        "interaction_coefficient"
    ]

    result[
        "full_interaction_p_value"
    ] = reference[
        "interaction_p_value"
    ]

    result[
        "full_interaction_fdr"
    ] = reference[
        "interaction_fdr"
    ]

    result[
        "full_interaction_rank"
    ] = reference[
        "interaction_rank"
    ]

    result[
        "mean_interaction_coefficient"
    ] = coefficients.mean(
        axis=1
    )

    result[
        "median_interaction_coefficient"
    ] = coefficients.median(
        axis=1
    )

    result[
        "interaction_coefficient_sd"
    ] = coefficients.std(
        axis=1,
        ddof=1,
    )

    result[
        "minimum_interaction_coefficient"
    ] = coefficients.min(
        axis=1
    )

    result[
        "maximum_interaction_coefficient"
    ] = coefficients.max(
        axis=1
    )

    result[
        "fraction_positive_interaction"
    ] = (
        coefficients > 0.0
    ).mean(axis=1)

    result[
        "fraction_negative_interaction"
    ] = (
        coefficients < 0.0
    ).mean(axis=1)

    result[
        "interaction_sign_stability"
    ] = (
        result[
            [
                "fraction_positive_interaction",
                "fraction_negative_interaction",
            ]
        ]
        .max(axis=1)
    )

    result[
        "mean_interaction_rank"
    ] = ranks.mean(
        axis=1
    )

    result[
        "interaction_rank_sd"
    ] = ranks.std(
        axis=1,
        ddof=1,
    )

    result[
        "median_interaction_p_value"
    ] = pvalues.median(
        axis=1
    )

    result[
        "fraction_nominal_interaction"
    ] = (
        pvalues < 0.05
    ).mean(axis=1)

    result[
        "convergence_fraction"
    ] = convergence.mean(
        axis=1
    )

    result[
        "robust_interaction_direction"
    ] = np.where(
        result[
            "interaction_sign_stability"
        ]
        >= 0.80,
        True,
        False,
    )

    result[
        "consensus_interaction_direction"
    ] = np.select(
        [
            result[
                "fraction_positive_interaction"
            ]
            > 0.5,

            result[
                "fraction_negative_interaction"
            ]
            > 0.5,
        ],
        [
            "greater_benefit_with_higher_pathway",
            "greater_benefit_with_lower_pathway",
        ],
        default="unstable_direction",
    )

    result = (
        result
        .sort_values(
            [
                "interaction_sign_stability",
                "mean_interaction_rank",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    result.index.name = "feature"

    return result


def run_treatment_effect_robustness(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    C_values: Sequence[float] = (
        0.03,
        0.10,
        0.30,
    ),
    n_splits_values: Sequence[int] = (
        4,
        5,
        6,
    ),
    n_repeats: int = 5,
    max_iter: int = 10000,
    base_random_state: int = 42,
    top_fraction: float = 0.25,
    n_modifier_perturbations: int = 5,
    modifier_subsample_fraction: float = 0.80,
    modifier_base_random_state: int = 2026,
    fdr_threshold: float = 0.10,
) -> TreatmentEffectRobustnessResult:
    X = _validate_feature_matrix(X)

    treatment = _validate_binary_series(
        treatment,
        "treatment",
    )

    outcome = _validate_binary_series(
        outcome,
        "outcome",
    )

    _validate_alignment(
        X,
        treatment,
        outcome,
    )

    scenario_summary, patient_matrix = (
        run_model_sensitivity_grid(
            X,
            treatment,
            outcome,
            C_values=C_values,
            n_splits_values=n_splits_values,
            n_repeats=n_repeats,
            max_iter=max_iter,
            base_random_state=base_random_state,
        )
    )

    patient_robustness = (
        build_patient_robustness_table(
            patient_matrix
        )
    )

    pairwise = compare_scenarios_pairwise(
        patient_matrix,
        top_fraction=top_fraction,
    )

    modifier_robustness = run_modifier_robustness(
        X,
        treatment,
        outcome,
        n_perturbations=n_modifier_perturbations,
        subsample_fraction=modifier_subsample_fraction,
        fdr_threshold=fdr_threshold,
        max_iter=max_iter,
        base_random_state=modifier_base_random_state,
    )

    robustness_counts = (
        patient_robustness[
            "robustness_state"
        ]
        .value_counts()
        .to_dict()
    )

    patient_count = int(
        len(
            patient_robustness
        )
    )

    finite_spearman = (
        pairwise[
            "spearman_ite"
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if len(finite_spearman) > 0:
        mean_pairwise_spearman = float(
            finite_spearman.mean()
        )

        minimum_pairwise_spearman = float(
            finite_spearman.min()
        )
    else:
        mean_pairwise_spearman = float("nan")
        minimum_pairwise_spearman = float("nan")

    summary: dict[
        str,
        Any,
    ] = {
        "patients":
            patient_count,

        "biological_features":
            int(
                X.shape[1]
            ),

        "n_model_scenarios":
            int(
                patient_matrix.shape[1]
            ),

        "n_repeats_per_scenario":
            int(
                n_repeats
            ),

        "C_values":
            tuple(
                float(value)
                for value in C_values
            ),

        "n_splits_values":
            tuple(
                int(value)
                for value in n_splits_values
            ),

        "mean_pairwise_ite_spearman":
            mean_pairwise_spearman,

        "minimum_pairwise_ite_spearman":
            minimum_pairwise_spearman,

        "mean_top_patient_overlap":
            float(
                pairwise[
                    "top_patient_overlap"
                ].mean()
            ),

        "minimum_top_patient_overlap":
            float(
                pairwise[
                    "top_patient_overlap"
                ].min()
            ),

        "mean_pairwise_sign_agreement":
            float(
                pairwise[
                    "sign_agreement"
                ].mean()
            ),

        "fraction_patient_sign_stability_ge_90pct":
            float(
                (
                    patient_robustness[
                        "sign_stability"
                    ]
                    >= 0.90
                ).mean()
            ),

        "fraction_patient_unanimous_sign":
            float(
                (
                    patient_robustness[
                        "sign_stability"
                    ]
                    == 1.0
                ).mean()
            ),

        "robust_patients":
            int(
                robustness_counts.get(
                    "robust",
                    0,
                )
            ),

        "moderate_patients":
            int(
                robustness_counts.get(
                    "moderate",
                    0,
                )
            ),

        "unstable_patients":
            int(
                robustness_counts.get(
                    "unstable",
                    0,
                )
            ),

        "fraction_robust_patients":
            float(
                robustness_counts.get(
                    "robust",
                    0,
                )
                / patient_count
            ),

        "mean_scenario_cohort_ite":
            float(
                scenario_summary[
                    "cohort_mean_ite"
                ].mean()
            ),

        "sd_scenario_cohort_ite":
            float(
                scenario_summary[
                    "cohort_mean_ite"
                ].std(
                    ddof=1
                )
            ),

        "minimum_scenario_cohort_ite":
            float(
                scenario_summary[
                    "cohort_mean_ite"
                ].min()
            ),

        "maximum_scenario_cohort_ite":
            float(
                scenario_summary[
                    "cohort_mean_ite"
                ].max()
            ),

        "mean_scenario_oof_auc":
            float(
                scenario_summary[
                    "mean_repeat_oof_auc"
                ].mean()
            ),

        "minimum_scenario_oof_auc":
            float(
                scenario_summary[
                    "mean_repeat_oof_auc"
                ].min()
            ),

        "maximum_scenario_oof_auc":
            float(
                scenario_summary[
                    "mean_repeat_oof_auc"
                ].max()
            ),

        "mean_scenario_oof_brier":
            float(
                scenario_summary[
                    "mean_repeat_oof_brier"
                ].mean()
            ),

        "n_modifier_perturbations":
            int(
                n_modifier_perturbations
            ),

        "modifier_subsample_fraction":
            float(
                modifier_subsample_fraction
            ),

        "fraction_modifiers_direction_stable_ge_80pct":
            float(
                (
                    modifier_robustness[
                        "interaction_sign_stability"
                    ]
                    >= 0.80
                ).mean()
            ),

        "fraction_modifiers_unanimous_direction":
            float(
                (
                    modifier_robustness[
                        "interaction_sign_stability"
                    ]
                    == 1.0
                ).mean()
            ),

        "all_modifier_perturbations_converged":
            bool(
                (
                    modifier_robustness[
                        "convergence_fraction"
                    ]
                    == 1.0
                ).all()
            ),
    }

    return TreatmentEffectRobustnessResult(
        scenario_summary=scenario_summary,
        patient_ite_by_scenario=patient_matrix,
        patient_robustness=patient_robustness,
        pairwise_scenario_comparison=pairwise,
        modifier_robustness=modifier_robustness,
        summary=summary,
    )


def run_neotrip_robustness(
    *,
    C_values: Sequence[float] = (
        0.03,
        0.10,
        0.30,
    ),
    n_splits_values: Sequence[int] = (
        4,
        5,
        6,
    ),
    n_repeats: int = 5,
    max_iter: int = 10000,
    base_random_state: int = 42,
    top_fraction: float = 0.25,
    n_modifier_perturbations: int = 5,
    modifier_subsample_fraction: float = 0.80,
    modifier_base_random_state: int = 2026,
    fdr_threshold: float = 0.10,
) -> TreatmentEffectRobustnessResult:
    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    dataset = build_treatment_effect_dataset()

    return run_treatment_effect_robustness(
        dataset.X,
        dataset.T,
        dataset.Y,
        C_values=C_values,
        n_splits_values=n_splits_values,
        n_repeats=n_repeats,
        max_iter=max_iter,
        base_random_state=base_random_state,
        top_fraction=top_fraction,
        n_modifier_perturbations=n_modifier_perturbations,
        modifier_subsample_fraction=modifier_subsample_fraction,
        modifier_base_random_state=modifier_base_random_state,
        fdr_threshold=fdr_threshold,
    )


def main() -> None:
    print(
        "=== HERMES 2.0 "
        "TREATMENT-EFFECT ROBUSTNESS "
        "AND SENSITIVITY ANALYSIS ==="
    )

    print()

    result = run_neotrip_robustness(
        C_values=(
            0.03,
            0.10,
            0.30,
        ),
        n_splits_values=(
            4,
            5,
            6,
        ),
        n_repeats=5,
        max_iter=10000,
        base_random_state=42,
        top_fraction=0.25,
        n_modifier_perturbations=5,
        modifier_subsample_fraction=0.80,
        modifier_base_random_state=2026,
        fdr_threshold=0.10,
    )

    print(
        "=== OVERALL ROBUSTNESS SUMMARY ==="
    )

    for key, value in result.summary.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== MODEL SENSITIVITY SCENARIOS ==="
    )

    print(
        result
        .scenario_summary
        .to_string()
    )

    print()

    print(
        "=== PAIRWISE SCENARIO COMPARISONS ==="
    )

    print(
        result
        .pairwise_scenario_comparison
        .to_string(
            index=False
        )
    )

    print()

    print(
        "=== MOST ROBUST PREDICTED BENEFIT ==="
    )

    robust_benefit = (
        result
        .patient_robustness[
            (
                result
                .patient_robustness[
                    "robustness_state"
                ]
                == "robust"
            )
            &
            (
                result
                .patient_robustness[
                    "consensus_direction"
                ]
                == "benefit"
            )
        ]
        .sort_values(
            [
                "mean_ite",
                "sign_stability",
                "benefit_rank_sd",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
    )

    print(
        robust_benefit
        .head(15)
        .to_string()
    )

    print()

    print(
        "=== MOST ANALYTICALLY UNSTABLE PATIENTS ==="
    )

    unstable = (
        result
        .patient_robustness
        .sort_values(
            [
                "sign_stability",
                "normalized_rank_sd",
                "ite_sensitivity_range",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
    )

    print(
        unstable
        .head(15)
        .to_string()
    )

    print()

    print(
        "=== MOST DIRECTIONALLY STABLE "
        "PATHWAY INTERACTIONS ==="
    )

    modifier_columns = [
        "full_interaction_coefficient",
        "full_interaction_p_value",
        "full_interaction_fdr",
        "mean_interaction_coefficient",
        "interaction_coefficient_sd",
        "interaction_sign_stability",
        "mean_interaction_rank",
        "fraction_nominal_interaction",
        "convergence_fraction",
        "consensus_interaction_direction",
    ]

    print(
        result
        .modifier_robustness[
            modifier_columns
        ]
        .head(20)
        .to_string()
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This analysis measures sensitivity to reasonable "
        "modeling and patient-composition perturbations."
    )

    print(
        "High robustness means the HERMES estimate is less "
        "dependent on the exact analytic specification."
    )

    print(
        "It does not establish causal validity, clinical utility, "
        "a validated predictive biomarker, or external generalizability."
    )


if __name__ == "__main__":
    main()