"""
HERMES 2.0
Treatment-Effect Robustness and Sensitivity Tests
==================================================

Validation suite for the HERMES 2.0 robustness framework.

The tests verify:

1. Synthetic trial generation.
2. Patient-level robustness arithmetic.
3. Pairwise scenario-comparison integrity.
4. Stratified patient perturbation.
5. Model-sensitivity grid construction.
6. Treatment-effect sign and ranking stability.
7. Modifier-robustness structure.
8. Integrated robustness analysis.
9. Deterministic behavior.
10. Lightweight NeoTRIP integration.

These tests validate software behavior and internal consistency.

They do NOT establish:
    - causal treatment-effect heterogeneity
    - clinical treatment recommendations
    - predictive biomarker validation
    - external generalizability
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.special import expit

from backend.app.treatment_effects.robustness import (
    build_patient_robustness_table,
    compare_scenarios_pairwise,
    run_model_sensitivity_grid,
    run_modifier_robustness,
    run_treatment_effect_robustness,
    stratified_subsample_index,
)


# =============================================================
# Test helper
# =============================================================


def check(
    condition: bool,
    message: str,
) -> None:
    """
    Explicit assertion helper with readable terminal output.
    """

    if not bool(
        condition
    ):
        raise AssertionError(
            message
        )

    print(
        f"PASS: {message}"
    )


# =============================================================
# Synthetic randomized trial
# =============================================================


def generate_synthetic_trial(
    *,
    n: int = 240,
    n_features: int = 6,
    interaction_strength: float = 1.25,
    random_state: int = 2026,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Create a randomized synthetic treatment trial.

    FEATURE_00 contains a real treatment interaction.

    Remaining features are noise features.

    A relatively strong injected treatment interaction is used so the
    robustness framework has a non-degenerate signal to analyze.
    """

    if n < 100:
        raise ValueError(
            "n must be at least 100."
        )

    if n_features < 2:
        raise ValueError(
            "n_features must be at least 2."
        )

    rng = np.random.default_rng(
        random_state
    )

    index = pd.Index(
        [
            f"SIM_{i:05d}"
            for i in range(
                n
            )
        ],
        name="Patient_ID",
    )

    X_array = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(
            n,
            n_features,
        ),
    )

    columns = [
        f"FEATURE_{i:02d}"
        for i in range(
            n_features
        )
    ]

    X = pd.DataFrame(
        X_array,
        index=index,
        columns=columns,
    )

    treatment = rng.binomial(
        1,
        0.5,
        size=n,
    )

    true_modifier = (
        X[
            "FEATURE_00"
        ]
        .to_numpy(
            dtype=float
        )
    )

    linear_predictor = (
        -0.30
        + 0.20
        * treatment
        + 0.15
        * true_modifier
        + interaction_strength
        * treatment
        * true_modifier
    )

    probability = expit(
        linear_predictor
    )

    outcome = rng.binomial(
        1,
        probability,
        size=n,
    )

    T = pd.Series(
        treatment,
        index=index,
        name="T",
        dtype=int,
    )

    Y = pd.Series(
        outcome,
        index=index,
        name="Y",
        dtype=int,
    )

    return (
        X,
        T,
        Y,
    )


# =============================================================
# Main test suite
# =============================================================


def main() -> None:

    print(
        "=== HERMES 2.0 ROBUSTNESS "
        "AND SENSITIVITY TESTS ==="
    )

    print()

    # =========================================================
    # 1. Synthetic dataset
    # =========================================================

    X, T, Y = generate_synthetic_trial(
        n=240,
        n_features=6,
        interaction_strength=1.25,
        random_state=2026,
    )

    check(
        X.shape
        == (
            240,
            6,
        ),
        "synthetic feature dimensions",
    )

    check(
        len(
            T
        )
        == 240,
        "synthetic treatment dimensions",
    )

    check(
        len(
            Y
        )
        == 240,
        "synthetic outcome dimensions",
    )

    check(
        set(
            T.unique()
        )
        == {
            0,
            1,
        },
        "synthetic randomized treatment encoding",
    )

    check(
        set(
            Y.unique()
        )
        == {
            0,
            1,
        },
        "synthetic binary outcome encoding",
    )

    check(
        X.index.equals(
            T.index
        )
        and X.index.equals(
            Y.index
        ),
        "synthetic patient alignment",
    )

    # =========================================================
    # 2. Patient robustness arithmetic
    # =========================================================

    toy_matrix = pd.DataFrame(
        {
            "scenario_1": [
                0.20,
                -0.20,
                0.10,
                -0.05,
            ],

            "scenario_2": [
                0.22,
                -0.18,
                0.12,
                0.02,
            ],

            "scenario_3": [
                0.18,
                -0.23,
                0.09,
                -0.01,
            ],
        },
        index=pd.Index(
            [
                "P1",
                "P2",
                "P3",
                "P4",
            ],
            name="Patient_ID",
        ),
    )

    patient_table = (
        build_patient_robustness_table(
            toy_matrix
        )
    )

    check(
        patient_table.shape[
            0
        ]
        == 4,
        "patient robustness preserves patient count",
    )

    required_patient_columns = {
        "mean_ite",
        "median_ite",
        "ite_sensitivity_sd",
        "minimum_ite",
        "maximum_ite",
        "ite_sensitivity_range",
        "fraction_positive",
        "fraction_negative",
        "fraction_zero",
        "sign_stability",
        "consensus_direction",
        "mean_benefit_rank",
        "median_benefit_rank",
        "benefit_rank_sd",
        "best_benefit_rank",
        "worst_benefit_rank",
        "benefit_rank_range",
        "normalized_rank_sd",
        "sensitivity_signal_ratio",
        "robustness_state",
    }

    check(
        required_patient_columns.issubset(
            patient_table.columns
        ),
        "patient robustness table schema",
    )

    check(
        np.isclose(
            patient_table.loc[
                "P1",
                "mean_ite",
            ],
            0.20,
        ),
        "patient mean ITE arithmetic",
    )

    check(
        np.isclose(
            patient_table.loc[
                "P1",
                "sign_stability",
            ],
            1.0,
        ),
        "unanimous positive sign stability",
    )

    check(
        np.isclose(
            patient_table.loc[
                "P2",
                "sign_stability",
            ],
            1.0,
        ),
        "unanimous negative sign stability",
    )

    check(
        patient_table.loc[
            "P1",
            "consensus_direction",
        ]
        == "benefit",
        "benefit consensus direction",
    )

    check(
        patient_table.loc[
            "P2",
            "consensus_direction",
        ]
        == "harm",
        "harm consensus direction",
    )

    check(
        (
            patient_table[
                "sign_stability"
            ]
            >= 0.0
        ).all()
        and (
            patient_table[
                "sign_stability"
            ]
            <= 1.0
        ).all(),
        "patient sign-stability bounds",
    )

    check(
        set(
            patient_table[
                "robustness_state"
            ]
            .unique()
        ).issubset(
            {
                "robust",
                "moderate",
                "unstable",
            }
        ),
        "valid patient robustness states",
    )

    # =========================================================
    # 3. Pairwise scenario comparisons
    # =========================================================

    pairwise = (
        compare_scenarios_pairwise(
            toy_matrix,
            top_fraction=0.50,
        )
    )

    check(
        len(
            pairwise
        )
        == 3,
        "pairwise scenario count",
    )

    required_pairwise_columns = {
        "scenario_a",
        "scenario_b",
        "spearman_ite",
        "top_fraction",
        "top_patient_overlap",
        "sign_agreement",
        "mean_absolute_ite_difference",
        "maximum_absolute_ite_difference",
    }

    check(
        required_pairwise_columns.issubset(
            pairwise.columns
        ),
        "pairwise comparison schema",
    )

    check(
        (
            pairwise[
                "top_patient_overlap"
            ]
            >= 0.0
        ).all()
        and (
            pairwise[
                "top_patient_overlap"
            ]
            <= 1.0
        ).all(),
        "top-patient overlap bounds",
    )

    check(
        (
            pairwise[
                "sign_agreement"
            ]
            >= 0.0
        ).all()
        and (
            pairwise[
                "sign_agreement"
            ]
            <= 1.0
        ).all(),
        "pairwise sign-agreement bounds",
    )

    check(
        (
            pairwise[
                "mean_absolute_ite_difference"
            ]
            >= 0.0
        ).all(),
        "non-negative pairwise ITE differences",
    )

    # =========================================================
    # 4. Stratified patient perturbation
    # =========================================================

    subsample = (
        stratified_subsample_index(
            T,
            Y,
            fraction=0.80,
            random_state=123,
        )
    )

    check(
        len(
            subsample
        )
        < len(
            X
        ),
        "subsample smaller than full cohort",
    )

    check(
        len(
            subsample
        )
        > 0,
        "non-empty stratified subsample",
    )

    check(
        set(
            subsample
        ).issubset(
            set(
                X.index
            )
        ),
        "subsample contains valid patients",
    )

    subsample_strata = pd.crosstab(
        T.loc[
            subsample
        ],
        Y.loc[
            subsample
        ],
    )

    check(
        subsample_strata.shape
        == (
            2,
            2,
        ),
        "all treatment-outcome strata preserved",
    )

    subsample_repeat = (
        stratified_subsample_index(
            T,
            Y,
            fraction=0.80,
            random_state=123,
        )
    )

    check(
        subsample.equals(
            subsample_repeat
        ),
        "deterministic stratified subsampling",
    )

    # =========================================================
    # 5. Model sensitivity grid
    # =========================================================

    scenario_summary, patient_matrix = (
        run_model_sensitivity_grid(
            X,
            T,
            Y,
            C_values=(
                0.05,
                0.20,
            ),
            n_splits_values=(
                4,
                5,
            ),
            n_repeats=2,
            max_iter=5000,
            base_random_state=42,
        )
    )

    check(
        scenario_summary.shape[
            0
        ]
        == 4,
        "model sensitivity scenario count",
    )

    check(
        patient_matrix.shape
        == (
            240,
            4,
        ),
        "patient × scenario sensitivity matrix dimensions",
    )

    check(
        not patient_matrix.isna().any().any(),
        "no missing sensitivity-grid ITE estimates",
    )

    check(
        np.isfinite(
            patient_matrix.to_numpy()
        ).all(),
        "finite sensitivity-grid ITE estimates",
    )

    check(
        scenario_summary[
            "regularization_C"
        ]
        .nunique()
        == 2,
        "regularization sensitivity represented",
    )

    check(
        scenario_summary[
            "n_splits"
        ]
        .nunique()
        == 2,
        "fold-count sensitivity represented",
    )

    check(
        (
            scenario_summary[
                "mean_repeat_oof_auc"
            ]
            >= 0.0
        ).all()
        and (
            scenario_summary[
                "mean_repeat_oof_auc"
            ]
            <= 1.0
        ).all(),
        "sensitivity-grid OOF AUC bounds",
    )

    check(
        (
            scenario_summary[
                "mean_repeat_oof_brier"
            ]
            >= 0.0
        ).all(),
        "sensitivity-grid Brier bounds",
    )

    # =========================================================
    # 6. Sensitivity-derived patient robustness
    # =========================================================

    sensitivity_patients = (
        build_patient_robustness_table(
            patient_matrix
        )
    )

    check(
        sensitivity_patients.shape[
            0
        ]
        == 240,
        "complete sensitivity patient coverage",
    )

    check(
        (
            sensitivity_patients[
                "fraction_positive"
            ]
            >= 0.0
        ).all()
        and (
            sensitivity_patients[
                "fraction_positive"
            ]
            <= 1.0
        ).all(),
        "positive treatment-effect fraction bounds",
    )

    check(
        (
            sensitivity_patients[
                "fraction_negative"
            ]
            >= 0.0
        ).all()
        and (
            sensitivity_patients[
                "fraction_negative"
            ]
            <= 1.0
        ).all(),
        "negative treatment-effect fraction bounds",
    )

    check(
        (
            sensitivity_patients[
                "normalized_rank_sd"
            ]
            >= 0.0
        ).all(),
        "non-negative ranking instability",
    )

    # =========================================================
    # 7. Modifier robustness
    # =========================================================

    modifier_robustness = (
        run_modifier_robustness(
            X,
            T,
            Y,
            n_perturbations=2,
            subsample_fraction=0.80,
            fdr_threshold=0.10,
            max_iter=5000,
            base_random_state=2026,
        )
    )

    check(
        modifier_robustness.shape[
            0
        ]
        == 6,
        "modifier robustness feature coverage",
    )

    required_modifier_columns = {
        "full_interaction_coefficient",
        "full_interaction_p_value",
        "full_interaction_fdr",
        "full_interaction_rank",
        "mean_interaction_coefficient",
        "median_interaction_coefficient",
        "interaction_coefficient_sd",
        "minimum_interaction_coefficient",
        "maximum_interaction_coefficient",
        "fraction_positive_interaction",
        "fraction_negative_interaction",
        "interaction_sign_stability",
        "mean_interaction_rank",
        "interaction_rank_sd",
        "median_interaction_p_value",
        "fraction_nominal_interaction",
        "convergence_fraction",
        "robust_interaction_direction",
        "consensus_interaction_direction",
    }

    check(
        required_modifier_columns.issubset(
            modifier_robustness.columns
        ),
        "modifier robustness schema",
    )

    check(
        (
            modifier_robustness[
                "interaction_sign_stability"
            ]
            >= 0.0
        ).all()
        and (
            modifier_robustness[
                "interaction_sign_stability"
            ]
            <= 1.0
        ).all(),
        "modifier sign-stability bounds",
    )

    check(
        (
            modifier_robustness[
                "convergence_fraction"
            ]
            >= 0.0
        ).all()
        and (
            modifier_robustness[
                "convergence_fraction"
            ]
            <= 1.0
        ).all(),
        "modifier convergence-fraction bounds",
    )

    true_modifier = (
        modifier_robustness
        .loc[
            "FEATURE_00"
        ]
    )

    check(
        true_modifier[
            "mean_interaction_coefficient"
        ]
        > 0.0,
        "injected modifier direction retained under perturbation",
    )

    check(
        true_modifier[
            "interaction_sign_stability"
        ]
        >= 0.50,
        "injected modifier non-degenerate directional stability",
    )

    # =========================================================
    # 8. Integrated robustness analysis
    # =========================================================

    integrated = (
        run_treatment_effect_robustness(
            X,
            T,
            Y,
            C_values=(
                0.05,
                0.20,
            ),
            n_splits_values=(
                4,
            ),
            n_repeats=2,
            max_iter=5000,
            base_random_state=42,
            top_fraction=0.25,
            n_modifier_perturbations=2,
            modifier_subsample_fraction=0.80,
            modifier_base_random_state=2026,
            fdr_threshold=0.10,
        )
    )

    check(
        integrated.summary[
            "patients"
        ]
        == 240,
        "integrated robustness patient count",
    )

    check(
        integrated.summary[
            "biological_features"
        ]
        == 6,
        "integrated robustness feature count",
    )

    check(
        integrated.summary[
            "n_model_scenarios"
        ]
        == 2,
        "integrated robustness scenario count",
    )

    check(
        integrated.patient_ite_by_scenario.shape
        == (
            240,
            2,
        ),
        "integrated patient × scenario matrix",
    )

    check(
        integrated.patient_robustness.shape[
            0
        ]
        == 240,
        "integrated patient robustness coverage",
    )

    check(
        integrated.modifier_robustness.shape[
            0
        ]
        == 6,
        "integrated modifier robustness coverage",
    )

    check(
        (
            integrated.summary[
                "robust_patients"
            ]
            + integrated.summary[
                "moderate_patients"
            ]
            + integrated.summary[
                "unstable_patients"
            ]
        )
        == 240,
        "patient robustness-state count conservation",
    )

    check(
        0.0
        <= integrated.summary[
            "fraction_robust_patients"
        ]
        <= 1.0,
        "robust-patient fraction bounds",
    )

    check(
        0.0
        <= integrated.summary[
            "mean_pairwise_sign_agreement"
        ]
        <= 1.0,
        "integrated sign-agreement bounds",
    )

    check(
        0.0
        <= integrated.summary[
            "mean_top_patient_overlap"
        ]
        <= 1.0,
        "integrated top-patient overlap bounds",
    )

    check(
        integrated.summary[
            "minimum_scenario_cohort_ite"
        ]
        <= integrated.summary[
            "mean_scenario_cohort_ite"
        ]
        <= integrated.summary[
            "maximum_scenario_cohort_ite"
        ],
        "cohort treatment-effect sensitivity ordering",
    )

    # =========================================================
    # 9. Deterministic integrated robustness
    # =========================================================

    integrated_repeat = (
        run_treatment_effect_robustness(
            X,
            T,
            Y,
            C_values=(
                0.05,
                0.20,
            ),
            n_splits_values=(
                4,
            ),
            n_repeats=2,
            max_iter=5000,
            base_random_state=42,
            top_fraction=0.25,
            n_modifier_perturbations=2,
            modifier_subsample_fraction=0.80,
            modifier_base_random_state=2026,
            fdr_threshold=0.10,
        )
    )

    pd.testing.assert_frame_equal(
        integrated.scenario_summary,
        integrated_repeat.scenario_summary,
        check_exact=True,
    )

    check(
        True,
        "deterministic scenario summary",
    )

    pd.testing.assert_frame_equal(
        integrated.patient_ite_by_scenario,
        integrated_repeat.patient_ite_by_scenario,
        check_exact=True,
    )

    check(
        True,
        "deterministic patient sensitivity matrix",
    )

    pd.testing.assert_frame_equal(
        integrated.patient_robustness,
        integrated_repeat.patient_robustness,
        check_exact=True,
    )

    check(
        True,
        "deterministic patient robustness table",
    )

    pd.testing.assert_frame_equal(
        integrated.pairwise_scenario_comparison,
        integrated_repeat.pairwise_scenario_comparison,
        check_exact=True,
    )

    check(
        True,
        "deterministic pairwise sensitivity comparison",
    )

    pd.testing.assert_frame_equal(
        integrated.modifier_robustness,
        integrated_repeat.modifier_robustness,
        check_exact=True,
    )

    check(
        True,
        "deterministic modifier robustness",
    )

    check(
        integrated.summary
        == integrated_repeat.summary,
        "deterministic integrated robustness summary",
    )

    # =========================================================
    # 10. Lightweight real NeoTRIP integration
    # =========================================================

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    neotrip = (
        build_treatment_effect_dataset()
    )

    check(
        neotrip.X.shape[
            0
        ]
        == 241,
        "NeoTRIP robustness patient count",
    )

    check(
        neotrip.X.shape[
            1
        ]
        == 50,
        "NeoTRIP robustness biological feature count",
    )

    neotrip_scenarios, neotrip_patient_matrix = (
        run_model_sensitivity_grid(
            neotrip.X,
            neotrip.T,
            neotrip.Y,
            C_values=(
                0.05,
                0.20,
            ),
            n_splits_values=(
                5,
            ),
            n_repeats=2,
            max_iter=5000,
            base_random_state=42,
        )
    )

    check(
        neotrip_scenarios.shape[
            0
        ]
        == 2,
        "NeoTRIP lightweight scenario count",
    )

    check(
        neotrip_patient_matrix.shape
        == (
            241,
            2,
        ),
        "NeoTRIP lightweight patient × scenario matrix",
    )

    check(
        np.isfinite(
            neotrip_patient_matrix
            .to_numpy()
        ).all(),
        "finite NeoTRIP robustness estimates",
    )

    neotrip_pairwise = (
        compare_scenarios_pairwise(
            neotrip_patient_matrix,
            top_fraction=0.25,
        )
    )

    check(
        len(
            neotrip_pairwise
        )
        == 1,
        "NeoTRIP pairwise scenario comparison",
    )

    check(
        0.0
        <= float(
            neotrip_pairwise.iloc[
                0
            ][
                "top_patient_overlap"
            ]
        )
        <= 1.0,
        "NeoTRIP top-patient overlap bounds",
    )

    # =========================================================
    # Final report
    # =========================================================

    print()

    print(
        "========================================="
    )

    print(
        "ALL ROBUSTNESS AND SENSITIVITY TESTS PASSED"
    )

    print(
        "========================================="
    )

    print()

    print(
        "Synthetic integrated analysis:"
    )

    print(
        "Patients: "
        f"{integrated.summary['patients']}"
    )

    print(
        "Features: "
        f"{integrated.summary['biological_features']}"
    )

    print(
        "Model scenarios: "
        f"{integrated.summary['n_model_scenarios']}"
    )

    print(
        "Mean pairwise ITE Spearman: "
        f"{integrated.summary['mean_pairwise_ite_spearman']:.4f}"
    )

    print(
        "Mean top-patient overlap: "
        f"{integrated.summary['mean_top_patient_overlap']:.4f}"
    )

    print(
        "Mean sign agreement: "
        f"{integrated.summary['mean_pairwise_sign_agreement']:.4f}"
    )

    print(
        "Robust patients: "
        f"{integrated.summary['robust_patients']}"
    )

    print(
        "Moderate patients: "
        f"{integrated.summary['moderate_patients']}"
    )

    print(
        "Unstable patients: "
        f"{integrated.summary['unstable_patients']}"
    )

    print()

    print(
        "NeoTRIP lightweight robustness:"
    )

    print(
        "Patients: "
        f"{len(neotrip.X)}"
    )

    print(
        "Features: "
        f"{neotrip.X.shape[1]}"
    )

    print(
        "Sensitivity scenarios: "
        f"{neotrip_scenarios.shape[0]}"
    )

    print(
        "Scenario Spearman: "
        f"{float(neotrip_pairwise.iloc[0]['spearman_ite']):.4f}"
    )

    print(
        "Top-quartile overlap: "
        f"{float(neotrip_pairwise.iloc[0]['top_patient_overlap']):.4f}"
    )

    print(
        "Sign agreement: "
        f"{float(neotrip_pairwise.iloc[0]['sign_agreement']):.4f}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "These tests validate the internal behavior and reproducibility "
        "of the HERMES robustness framework."
    )

    print(
        "They do not establish causal validity, clinical utility, "
        "predictive-biomarker validation, or external generalizability."
    )


if __name__ == "__main__":
    main()