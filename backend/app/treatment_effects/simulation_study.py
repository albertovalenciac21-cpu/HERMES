"""
HERMES 2.0
Multi-Seed Simulation Study for Treatment-Effect Recovery
=========================================================

Purpose
-------
Evaluate whether HERMES treatment-effect recovery generalizes across
independently generated randomized synthetic trials.

The positive-control module demonstrates recovery in an individual simulated
dataset. This module repeats that experiment across multiple independent data
generating seeds and interaction strengths.

This allows us to quantify:

    1. mean recovery performance
    2. between-simulation variability
    3. Monte Carlo uncertainty
    4. probability of successful recovery
    5. degradation of recovery at weak interaction strengths
    6. improvement as treatment-effect heterogeneity becomes stronger

IMPORTANT
---------
This remains simulation validation.

Successful performance demonstrates that the HERMES implementation can
recover known heterogeneous treatment effects under the simulated
data-generating process.

It does NOT establish treatment-effect heterogeneity in NeoTRIP or any
clinical dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from backend.app.treatment_effects.positive_control import (
    run_positive_control,
)


# =============================================================
# Result container
# =============================================================


@dataclass
class SimulationStudyResult:
    """
    Results from a multi-seed HERMES positive-control simulation study.
    """

    simulation_table: pd.DataFrame

    interaction_summary: pd.DataFrame

    overall_summary: dict[str, Any]


# =============================================================
# Constants
# =============================================================


RECOVERY_METRICS: tuple[str, ...] = (
    "ite_pearson_correlation",
    "ite_spearman_correlation",
    "ite_sign_accuracy",
    "top_quartile_overlap",
    "true_ite_top_bottom_separation",
    "mean_absolute_ite_error",
    "root_mean_squared_ite_error",
    "true_ite_sd",
    "estimated_ite_sd",
)


# =============================================================
# Helper functions
# =============================================================


def _validate_interaction_strengths(
    interaction_strengths: Sequence[float],
) -> tuple[float, ...]:
    """
    Validate and normalize interaction strengths.
    """

    strengths = tuple(
        float(value)
        for value in interaction_strengths
    )

    if len(strengths) == 0:
        raise ValueError(
            "interaction_strengths must contain at least one value."
        )

    if not all(
        np.isfinite(value)
        for value in strengths
    ):
        raise ValueError(
            "interaction_strengths must contain only finite values."
        )

    return strengths


def _validate_random_states(
    random_states: Sequence[int],
) -> tuple[int, ...]:
    """
    Validate independent simulation seeds.
    """

    states = tuple(
        int(value)
        for value in random_states
    )

    if len(states) == 0:
        raise ValueError(
            "random_states must contain at least one seed."
        )

    if len(set(states)) != len(states):
        raise ValueError(
            "random_states must be unique."
        )

    return states


def generate_simulation_random_states(
    *,
    n_simulations: int,
    base_random_state: int = 2026,
) -> tuple[int, ...]:
    """
    Generate deterministic independent random states.

    SeedSequence is used so the simulation seeds are reproducible while
    remaining independent across synthetic trials.
    """

    if n_simulations < 1:
        raise ValueError(
            "n_simulations must be at least 1."
        )

    seed_sequence = np.random.SeedSequence(
        int(base_random_state)
    )

    children = seed_sequence.spawn(
        n_simulations
    )

    random_states = tuple(
        int(
            child.generate_state(
                1,
                dtype=np.uint32,
            )[0]
        )
        for child in children
    )

    return random_states


def _safe_mean(
    values: pd.Series,
) -> float:
    """
    NaN-safe mean.
    """

    array = values.to_numpy(
        dtype=float
    )

    if np.all(
        np.isnan(array)
    ):
        return float("nan")

    return float(
        np.nanmean(array)
    )


def _safe_std(
    values: pd.Series,
) -> float:
    """
    NaN-safe sample standard deviation.
    """

    array = values.to_numpy(
        dtype=float
    )

    valid = array[
        ~np.isnan(array)
    ]

    if len(valid) < 2:
        return float("nan")

    return float(
        np.std(
            valid,
            ddof=1,
        )
    )


def _safe_quantile(
    values: pd.Series,
    quantile: float,
) -> float:
    """
    NaN-safe quantile.
    """

    array = values.to_numpy(
        dtype=float
    )

    valid = array[
        ~np.isnan(array)
    ]

    if len(valid) == 0:
        return float("nan")

    return float(
        np.quantile(
            valid,
            quantile,
        )
    )


def _monte_carlo_standard_error(
    values: pd.Series,
) -> float:
    """
    Monte Carlo standard error of the simulation mean.
    """

    array = values.to_numpy(
        dtype=float
    )

    valid = array[
        ~np.isnan(array)
    ]

    if len(valid) < 2:
        return float("nan")

    return float(
        np.std(
            valid,
            ddof=1,
        )
        / np.sqrt(
            len(valid)
        )
    )


def _mean_confidence_interval(
    values: pd.Series,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """
    Approximate Monte Carlo 95% confidence interval for a simulation mean.
    """

    mean_value = _safe_mean(
        values
    )

    mcse = _monte_carlo_standard_error(
        values
    )

    if (
        np.isnan(mean_value)
        or np.isnan(mcse)
    ):
        return (
            float("nan"),
            float("nan"),
        )

    return (
        float(
            mean_value
            - z_value * mcse
        ),
        float(
            mean_value
            + z_value * mcse
        ),
    )


# =============================================================
# Single simulation
# =============================================================


def run_single_simulation(
    *,
    interaction_strength: float,
    data_random_state: int,
    n_patients: int = 500,
    n_features: int = 20,
    n_repeats: int = 5,
    n_splits: int = 5,
    C: float = 0.1,
    model_random_state: int = 42,
) -> dict[str, Any]:
    """
    Run one independent HERMES positive-control simulation.
    """

    result = run_positive_control(
        n_patients=n_patients,
        n_features=n_features,
        treatment_interaction=float(
            interaction_strength
        ),
        n_repeats=n_repeats,
        n_splits=n_splits,
        C=C,
        data_random_state=int(
            data_random_state
        ),
        model_random_state=int(
            model_random_state
        ),
    )

    record: dict[str, Any] = {
        "interaction_strength":
            float(
                interaction_strength
            ),

        "data_random_state":
            int(
                data_random_state
            ),

        "observed_treatment_fraction":
            float(
                result.summary[
                    "observed_treatment_fraction"
                ]
            ),

        "observed_outcome_fraction":
            float(
                result.summary[
                    "observed_outcome_fraction"
                ]
            ),
    }

    for metric in RECOVERY_METRICS:

        record[
            metric
        ] = float(
            result.metrics[
                metric
            ]
        )

    record[
        "mean_true_ite"
    ] = float(
        result.metrics[
            "mean_true_ite"
        ]
    )

    record[
        "mean_estimated_ite"
    ] = float(
        result.metrics[
            "mean_estimated_ite"
        ]
    )

    return record


# =============================================================
# Multi-seed simulation
# =============================================================


def run_simulation_grid(
    *,
    interaction_strengths: Sequence[float],
    random_states: Sequence[int],
    n_patients: int = 500,
    n_features: int = 20,
    n_repeats: int = 5,
    n_splits: int = 5,
    C: float = 0.1,
    model_random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run all interaction-strength × simulation-seed combinations.
    """

    strengths = _validate_interaction_strengths(
        interaction_strengths
    )

    states = _validate_random_states(
        random_states
    )

    records: list[
        dict[str, Any]
    ] = []

    total_runs = (
        len(strengths)
        * len(states)
    )

    run_number = 0

    for interaction_strength in strengths:

        for simulation_number, random_state in enumerate(
            states,
            start=1,
        ):

            run_number += 1

            if verbose:

                print(
                    f"Run {run_number}/{total_runs}: "
                    f"interaction={interaction_strength:.2f}, "
                    f"simulation={simulation_number}, "
                    f"seed={random_state}"
                )

            record = run_single_simulation(
                interaction_strength=(
                    interaction_strength
                ),
                data_random_state=(
                    random_state
                ),
                n_patients=n_patients,
                n_features=n_features,
                n_repeats=n_repeats,
                n_splits=n_splits,
                C=C,
                model_random_state=(
                    model_random_state
                ),
            )

            record[
                "simulation"
            ] = int(
                simulation_number
            )

            records.append(
                record
            )

    table = pd.DataFrame(
        records
    )

    expected_rows = (
        len(strengths)
        * len(states)
    )

    if len(table) != expected_rows:
        raise RuntimeError(
            "Simulation grid did not produce the expected number of rows."
        )

    return table


# =============================================================
# Aggregation
# =============================================================


def summarize_simulation_metric(
    group: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    """
    Summarize one metric across independent simulations.
    """

    values = group[
        metric
    ].astype(float)

    lower_ci, upper_ci = (
        _mean_confidence_interval(
            values
        )
    )

    return {
        f"{metric}_mean":
            _safe_mean(
                values
            ),

        f"{metric}_sd":
            _safe_std(
                values
            ),

        f"{metric}_mcse":
            _monte_carlo_standard_error(
                values
            ),

        f"{metric}_median":
            _safe_quantile(
                values,
                0.50,
            ),

        f"{metric}_q025":
            _safe_quantile(
                values,
                0.025,
            ),

        f"{metric}_q975":
            _safe_quantile(
                values,
                0.975,
            ),

        f"{metric}_mean_ci_lower":
            lower_ci,

        f"{metric}_mean_ci_upper":
            upper_ci,
    }


def summarize_interaction_strengths(
    simulation_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate HERMES recovery across independent trials at each
    interaction strength.
    """

    required_columns = {
        "interaction_strength",
        "data_random_state",
        *RECOVERY_METRICS,
    }

    missing = (
        required_columns
        - set(
            simulation_table.columns
        )
    )

    if missing:
        raise ValueError(
            "simulation_table is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    records: list[
        dict[str, Any]
    ] = []

    grouped = simulation_table.groupby(
        "interaction_strength",
        sort=True,
    )

    for interaction_strength, group in grouped:

        record: dict[str, Any] = {
            "interaction_strength":
                float(
                    interaction_strength
                ),

            "n_simulations":
                int(
                    len(group)
                ),
        }

        for metric in RECOVERY_METRICS:

            record.update(
                summarize_simulation_metric(
                    group,
                    metric,
                )
            )

        # -----------------------------------------------------
        # Useful recovery probabilities
        # -----------------------------------------------------

        pearson = group[
            "ite_pearson_correlation"
        ].astype(float)

        spearman = group[
            "ite_spearman_correlation"
        ].astype(float)

        sign_accuracy = group[
            "ite_sign_accuracy"
        ].astype(float)

        top_overlap = group[
            "top_quartile_overlap"
        ].astype(float)

        separation = group[
            "true_ite_top_bottom_separation"
        ].astype(float)

        record[
            "fraction_positive_pearson"
        ] = float(
            (
                pearson > 0
            ).mean()
        )

        record[
            "fraction_pearson_ge_0_50"
        ] = float(
            (
                pearson >= 0.50
            ).mean()
        )

        record[
            "fraction_positive_spearman"
        ] = float(
            (
                spearman > 0
            ).mean()
        )

        record[
            "fraction_sign_accuracy_ge_0_75"
        ] = float(
            (
                sign_accuracy >= 0.75
            ).mean()
        )

        record[
            "fraction_top_overlap_ge_0_50"
        ] = float(
            (
                top_overlap >= 0.50
            ).mean()
        )

        record[
            "fraction_positive_true_separation"
        ] = float(
            (
                separation > 0
            ).mean()
        )

        records.append(
            record
        )

    return (
        pd.DataFrame(
            records
        )
        .set_index(
            "interaction_strength"
        )
        .sort_index()
    )


# =============================================================
# Full simulation study
# =============================================================


def run_simulation_study(
    *,
    interaction_strengths: Sequence[float] = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ),
    n_simulations: int = 20,
    n_patients: int = 500,
    n_features: int = 20,
    n_repeats: int = 5,
    n_splits: int = 5,
    C: float = 0.1,
    simulation_base_random_state: int = 2026,
    model_random_state: int = 42,
    verbose: bool = True,
) -> SimulationStudyResult:
    """
    Run the complete multi-seed positive-control simulation study.
    """

    strengths = _validate_interaction_strengths(
        interaction_strengths
    )

    random_states = (
        generate_simulation_random_states(
            n_simulations=n_simulations,
            base_random_state=(
                simulation_base_random_state
            ),
        )
    )

    simulation_table = run_simulation_grid(
        interaction_strengths=strengths,
        random_states=random_states,
        n_patients=n_patients,
        n_features=n_features,
        n_repeats=n_repeats,
        n_splits=n_splits,
        C=C,
        model_random_state=(
            model_random_state
        ),
        verbose=verbose,
    )

    interaction_summary = (
        summarize_interaction_strengths(
            simulation_table
        )
    )

    overall_summary: dict[
        str,
        Any,
    ] = {
        "interaction_strengths":
            strengths,

        "n_interaction_strengths":
            int(
                len(strengths)
            ),

        "n_simulations_per_strength":
            int(
                n_simulations
            ),

        "total_simulation_runs":
            int(
                len(
                    simulation_table
                )
            ),

        "n_patients_per_trial":
            int(
                n_patients
            ),

        "n_features":
            int(
                n_features
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

        "simulation_base_random_state":
            int(
                simulation_base_random_state
            ),

        "model_random_state":
            int(
                model_random_state
            ),
    }

    return SimulationStudyResult(
        simulation_table=(
            simulation_table
        ),
        interaction_summary=(
            interaction_summary
        ),
        overall_summary=(
            overall_summary
        ),
    )


# =============================================================
# CLI
# =============================================================


def main() -> None:
    """
    Run a development-scale multi-seed simulation study.

    The defaults below are intentionally modest so the module can be tested
    quickly. Publication-scale simulation counts should be substantially
    larger.
    """

    print(
        "=== HERMES 2.0 MULTI-SEED "
        "TREATMENT-EFFECT SIMULATION STUDY ==="
    )

    print()

    result = run_simulation_study(
        interaction_strengths=(
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
        ),
        n_simulations=5,
        n_patients=500,
        n_features=20,
        n_repeats=5,
        n_splits=5,
        C=0.1,
        simulation_base_random_state=2026,
        model_random_state=42,
        verbose=True,
    )

    print()

    print(
        "=== STUDY CONFIGURATION ==="
    )

    for key, value in (
        result.overall_summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== INTERACTION-STRENGTH SUMMARY ==="
    )

    display_columns = [
        "n_simulations",

        "ite_pearson_correlation_mean",
        "ite_pearson_correlation_sd",

        "ite_spearman_correlation_mean",
        "ite_spearman_correlation_sd",

        "ite_sign_accuracy_mean",

        "top_quartile_overlap_mean",

        "true_ite_top_bottom_separation_mean",

        "mean_absolute_ite_error_mean",

        "fraction_positive_pearson",

        "fraction_pearson_ge_0_50",

        "fraction_sign_accuracy_ge_0_75",

        "fraction_top_overlap_ge_0_50",
    ]

    print(
        result.interaction_summary[
            display_columns
        ].to_string()
    )

    print()

    print(
        "=== INTERPRETATION ==="
    )

    print(
        "Recovery should strengthen as the injected treatment "
        "interaction increases."
    )

    print(
        "At interaction = 0, true treatment-effect heterogeneity "
        "should be minimal and apparent recovery should be weak."
    )

    print(
        "Consistent recovery across independent synthetic trials "
        "supports robustness of the HERMES treatment-effect pipeline."
    )

    print(
        "This remains simulation validation and does not establish "
        "heterogeneous treatment effects in NeoTRIP."
    )


if __name__ == "__main__":
    main()