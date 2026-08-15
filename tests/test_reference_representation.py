"""
HERMES 2.0
Reference-Fitted Biological Representation Tests
=================================================

Validation suite for frozen-reference biological preprocessing.

The tests verify:

1. Reference representation construction.
2. Reference-only normalization.
3. Frozen pathway aggregation.
4. Preservation of external biological shifts.
5. Prevention of target-cohort preprocessing leakage.
6. Missing-gene handling.
7. Extra-gene handling.
8. Pathway coverage tracking.
9. Deterministic transformation.
10. Holdout transformation behavior.

These tests validate software behavior and preprocessing integrity.

They do NOT establish:
    - external predictive validity
    - causal transportability
    - treatment-effect validity
    - clinical utility
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.reference_representation import (
    aggregate_pathways,
    compare_reference_vs_independent_scaling,
    fit_reference_representation,
    fit_transform_reference,
    transform_with_reference,
)


# =============================================================
# Test helper
# =============================================================


def check(
    condition: bool,
    message: str,
) -> None:

    if not bool(condition):
        raise AssertionError(
            message
        )

    print(
        f"PASS: {message}"
    )


# =============================================================
# Synthetic expression generator
# =============================================================


def generate_expression_data(
    *,
    n_reference: int = 300,
    n_target: int = 100,
    n_genes: int = 40,
    target_shift: float = 1.5,
    random_state: int = 2026,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[str, ...]],
]:

    rng = np.random.default_rng(
        random_state
    )

    genes = [
        f"GENE_{i:03d}"
        for i in range(
            n_genes
        )
    ]

    reference = pd.DataFrame(
        rng.normal(
            loc=5.0,
            scale=2.0,
            size=(
                n_reference,
                n_genes,
            ),
        ),
        index=pd.Index(
            [
                f"REF_{i:05d}"
                for i in range(
                    n_reference
                )
            ],
            name="Patient_ID",
        ),
        columns=genes,
    )

    target = pd.DataFrame(
        rng.normal(
            loc=(
                5.0
                + target_shift
            ),
            scale=2.0,
            size=(
                n_target,
                n_genes,
            ),
        ),
        index=pd.Index(
            [
                f"TARGET_{i:05d}"
                for i in range(
                    n_target
                )
            ],
            name="Patient_ID",
        ),
        columns=genes,
    )

    gene_sets = {
        "PATHWAY_A":
            tuple(
                genes[
                    0:10
                ]
            ),

        "PATHWAY_B":
            tuple(
                genes[
                    10:20
                ]
            ),

        "PATHWAY_C":
            tuple(
                genes[
                    20:30
                ]
            ),

        "PATHWAY_D":
            tuple(
                genes[
                    30:40
                ]
            ),
    }

    return (
        reference,
        target,
        gene_sets,
    )


# =============================================================
# Main tests
# =============================================================


def main() -> None:

    print(
        "=== HERMES 2.0 REFERENCE-FITTED "
        "BIOLOGICAL REPRESENTATION TESTS ==="
    )

    print()

    reference_expression, target_expression, gene_sets = (
        generate_expression_data()
    )

    # =========================================================
    # 1. Basic dimensions
    # =========================================================

    check(
        reference_expression.shape
        == (300, 40),
        "reference expression dimensions",
    )

    check(
        target_expression.shape
        == (100, 40),
        "target expression dimensions",
    )

    check(
        len(
            gene_sets
        )
        == 4,
        "synthetic pathway count",
    )

    # =========================================================
    # 2. Fit representation
    # =========================================================

    representation = fit_reference_representation(
        reference_expression,
        gene_sets,
        minimum_genes_per_pathway=3,
        standardize_pathways=True,
    )

    check(
        representation.summary[
            "reference_patients"
        ]
        == 300,
        "reference patient count",
    )

    check(
        representation.summary[
            "reference_genes"
        ]
        == 40,
        "reference gene count",
    )

    check(
        representation.summary[
            "pathways_requested"
        ]
        == 4,
        "requested pathway count",
    )

    check(
        representation.summary[
            "pathways_retained"
        ]
        == 4,
        "retained pathway count",
    )

    check(
        representation.pathway_names
        == tuple(
            gene_sets.keys()
        ),
        "pathway ordering preservation",
    )

    check(
        representation.gene_mean.index.equals(
            reference_expression.columns
        ),
        "reference gene mean alignment",
    )

    check(
        representation.gene_sd.index.equals(
            reference_expression.columns
        ),
        "reference gene SD alignment",
    )

    check(
        (
            representation.gene_sd
            > 0.0
        ).all(),
        "positive frozen gene SDs",
    )

    # =========================================================
    # 3. Fit-transform reference cohort
    # =========================================================

    fitted_representation, reference_result = (
        fit_transform_reference(
            reference_expression,
            gene_sets,
            minimum_genes_per_pathway=3,
            standardize_pathways=True,
        )
    )

    check(
        reference_result.pathway_scores.shape
        == (300, 4),
        "reference pathway-score dimensions",
    )

    check(
        np.isfinite(
            reference_result
            .pathway_scores
            .to_numpy()
        ).all(),
        "finite reference pathway scores",
    )

    reference_pathway_means = (
        reference_result
        .pathway_scores
        .mean(
            axis=0
        )
    )

    check(
        np.allclose(
            reference_pathway_means.to_numpy(),
            np.zeros(
                4
            ),
            atol=1e-10,
        ),
        "reference pathway means approximately zero",
    )

    reference_pathway_sd = (
        reference_result
        .pathway_scores
        .std(
            axis=0,
            ddof=1,
        )
    )

    check(
        np.allclose(
            reference_pathway_sd.to_numpy(),
            np.ones(
                4
            ),
            atol=1e-10,
        ),
        "reference pathway SD approximately one",
    )

    # =========================================================
    # 4. Frozen transformation of shifted target
    # =========================================================

    target_result = transform_with_reference(
        fitted_representation,
        target_expression,
        missing_gene_policy="error",
    )

    check(
        target_result.pathway_scores.shape
        == (100, 4),
        "target pathway-score dimensions",
    )

    check(
        target_result.summary[
            "missing_required_genes"
        ]
        == 0,
        "complete target gene coverage",
    )

    check(
        target_result.summary[
            "used_reference_gene_scaling"
        ],
        "frozen reference gene scaling used",
    )

    check(
        target_result.summary[
            "used_reference_pathway_scaling"
        ],
        "frozen reference pathway scaling used",
    )

    target_means = (
        target_result
        .pathway_scores
        .mean(
            axis=0
        )
    )

    check(
        (
            target_means.abs()
            > 0.25
        ).all(),
        "external biological shift remains visible",
    )

    check(
        target_means.mean()
        > 0.50,
        "positive injected target shift preserved",
    )

    # =========================================================
    # 5. Leakage diagnostic
    # =========================================================

    scaling_comparison = (
        compare_reference_vs_independent_scaling(
            fitted_representation,
            target_expression,
        )
    )

    check(
        scaling_comparison.shape[
            0
        ]
        == 4,
        "scaling-comparison pathway coverage",
    )

    check(
        (
            scaling_comparison[
                "mean_absolute_difference"
            ]
            > 0.25
        ).all(),
        "independent scaling materially alters pathway coordinates",
    )

    check(
        (
            scaling_comparison[
                "mean_independently_scaled"
            ]
            .abs()
            < 1e-10
        ).all(),
        "independent scaling erases target mean shift",
    )

    check(
        (
            scaling_comparison[
                "mean_reference_scaled"
            ]
            .abs()
            > 0.25
        ).all(),
        "frozen scaling preserves target mean shift",
    )

    check(
        (
            scaling_comparison[
                "spearman_correlation"
            ]
            > 0.90
        ).all(),
        "within-cohort pathway ranking largely preserved",
    )

    # =========================================================
    # 6. Extra gene behavior
    # =========================================================

    target_with_extra = (
        target_expression.copy()
    )

    target_with_extra[
        "EXTRA_GENE"
    ] = np.linspace(
        0.0,
        1.0,
        len(
            target_with_extra
        ),
    )

    extra_result = transform_with_reference(
        fitted_representation,
        target_with_extra,
        allow_extra_genes=True,
        missing_gene_policy="error",
    )

    check(
        "EXTRA_GENE"
        in extra_result.extra_genes,
        "extra target gene detected",
    )

    pd.testing.assert_frame_equal(
        target_result.pathway_scores,
        extra_result.pathway_scores,
        check_exact=True,
    )

    check(
        True,
        "ignored extra gene does not alter representation",
    )

    extra_error = False

    try:
        transform_with_reference(
            fitted_representation,
            target_with_extra,
            allow_extra_genes=False,
            missing_gene_policy="error",
        )

    except ValueError:
        extra_error = True

    check(
        extra_error,
        "unexpected extra gene correctly rejected when required",
    )

    # =========================================================
    # 7. Missing gene behavior
    # =========================================================

    missing_gene = (
        fitted_representation
        .pathway_genes[
            "PATHWAY_A"
        ][0]
    )

    target_missing = (
        target_expression.drop(
            columns=[
                missing_gene
            ]
        )
    )

    missing_error = False

    try:
        transform_with_reference(
            fitted_representation,
            target_missing,
            missing_gene_policy="error",
        )

    except ValueError:
        missing_error = True

    check(
        missing_error,
        "missing required gene correctly rejected",
    )

    imputed_result = (
        transform_with_reference(
            fitted_representation,
            target_missing,
            missing_gene_policy="reference_mean",
        )
    )

    check(
        missing_gene
        in imputed_result.missing_genes,
        "missing gene tracked during reference-mean imputation",
    )

    check(
        imputed_result.summary[
            "missing_required_genes"
        ]
        == 1,
        "missing-gene count preserved",
    )

    check(
        np.isfinite(
            imputed_result
            .pathway_scores
            .to_numpy()
        ).all(),
        "finite pathway scores after reference-mean imputation",
    )

    check(
        imputed_result
        .pathway_coverage
        .loc[
            "PATHWAY_A",
            "observed_fraction",
        ]
        < 1.0,
        "pathway coverage reflects missing gene",
    )

    # =========================================================
    # 8. Pathway filtering
    # =========================================================

    incomplete_gene_sets = {
        "VALID_PATHWAY":
            tuple(
                reference_expression
                .columns[
                    0:5
                ]
            ),

        "TOO_SMALL":
            (
                "GENE_000",
                "NOT_PRESENT_1",
                "NOT_PRESENT_2",
            ),
    }

    filtered_representation = (
        fit_reference_representation(
            reference_expression,
            incomplete_gene_sets,
            minimum_genes_per_pathway=3,
            standardize_pathways=True,
        )
    )

    check(
        filtered_representation.pathway_names
        == (
            "VALID_PATHWAY",
        ),
        "insufficient-coverage pathway excluded",
    )

    check(
        filtered_representation.summary[
            "pathways_retained"
        ]
        == 1,
        "filtered pathway count",
    )

    # =========================================================
    # 9. Aggregate pathway arithmetic
    # =========================================================

    standardized = (
        reference_expression
        - fitted_representation.gene_mean
    ).divide(
        fitted_representation.gene_sd,
        axis=1,
    )

    manual_pathway_a = (
        standardized[
            list(
                fitted_representation
                .pathway_genes[
                    "PATHWAY_A"
                ]
            )
        ]
        .mean(
            axis=1
        )
    )

    aggregated = aggregate_pathways(
        standardized,
        fitted_representation.pathway_genes,
    )

    check(
        np.allclose(
            manual_pathway_a.to_numpy(),
            aggregated[
                "PATHWAY_A"
            ].to_numpy(),
            atol=1e-12,
        ),
        "pathway aggregation arithmetic",
    )

    # =========================================================
    # 10. Determinism
    # =========================================================

    representation_2, reference_result_2 = (
        fit_transform_reference(
            reference_expression,
            gene_sets,
            minimum_genes_per_pathway=3,
            standardize_pathways=True,
        )
    )

    pd.testing.assert_series_equal(
        fitted_representation.gene_mean,
        representation_2.gene_mean,
        check_exact=True,
    )

    check(
        True,
        "deterministic frozen gene means",
    )

    pd.testing.assert_series_equal(
        fitted_representation.gene_sd,
        representation_2.gene_sd,
        check_exact=True,
    )

    check(
        True,
        "deterministic frozen gene SDs",
    )

    pd.testing.assert_frame_equal(
        reference_result.pathway_scores,
        reference_result_2.pathway_scores,
        check_exact=True,
    )

    check(
        True,
        "deterministic reference transformation",
    )

    target_result_2 = transform_with_reference(
        representation_2,
        target_expression,
        missing_gene_policy="error",
    )

    pd.testing.assert_frame_equal(
        target_result.pathway_scores,
        target_result_2.pathway_scores,
        check_exact=True,
    )

    check(
        True,
        "deterministic target transformation",
    )

    # =========================================================
    # 11. Reference / holdout separation
    # =========================================================

    rng = np.random.default_rng(
        42
    )

    patient_order = rng.permutation(
        len(
            reference_expression
        )
    )

    training_positions = (
        patient_order[
            :240
        ]
    )

    holdout_positions = (
        patient_order[
            240:
        ]
    )

    training_expression = (
        reference_expression.iloc[
            training_positions
        ]
        .copy()
    )

    holdout_expression = (
        reference_expression.iloc[
            holdout_positions
        ]
        .copy()
    )

    holdout_representation = (
        fit_reference_representation(
            training_expression,
            gene_sets,
            minimum_genes_per_pathway=3,
            standardize_pathways=True,
        )
    )

    holdout_result = (
        transform_with_reference(
            holdout_representation,
            holdout_expression,
            missing_gene_policy="error",
        )
    )

    check(
        holdout_result.pathway_scores.shape
        == (60, 4),
        "holdout transformation dimensions",
    )

    check(
        set(
            training_expression.index
        ).isdisjoint(
            set(
                holdout_expression.index
            )
        ),
        "training and holdout patients separated",
    )

    check(
        np.isfinite(
            holdout_result
            .pathway_scores
            .to_numpy()
        ).all(),
        "finite holdout pathway representation",
    )

    # The holdout is sampled from the same population, so it
    # should remain relatively close to the training reference.
    check(
        (
            holdout_result
            .pathway_scores
            .mean()
            .abs()
            < 0.50
        ).all(),
        "same-population holdout remains near reference center",
    )

    # =========================================================
    # Final summary
    # =========================================================

    print()

    print(
        "====================================================="
    )

    print(
        "ALL REFERENCE-FITTED REPRESENTATION TESTS PASSED"
    )

    print(
        "====================================================="
    )

    print()

    print(
        "Reference cohort:"
    )

    print(
        f"Patients: {len(reference_expression)}"
    )

    print(
        f"Genes: {reference_expression.shape[1]}"
    )

    print(
        f"Pathways: {len(fitted_representation.pathway_names)}"
    )

    print()

    print(
        "Reference pathway means:"
    )

    print(
        reference_result
        .pathway_scores
        .mean()
        .round(6)
        .to_string()
    )

    print()

    print(
        "Shifted target pathway means under frozen scaling:"
    )

    print(
        target_result
        .pathway_scores
        .mean()
        .round(6)
        .to_string()
    )

    print()

    print(
        "Independent-scaling means:"
    )

    print(
        scaling_comparison[
            "mean_independently_scaled"
        ]
        .round(6)
        .to_string()
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Frozen preprocessing preserves biological differences "
        "between reference and unseen cohorts."
    )

    print(
        "Target-cohort statistics are not used to redefine the "
        "HERMES biological coordinate system."
    )

    print(
        "This is a prerequisite for external validation, "
        "not external validation itself."
    )


if __name__ == "__main__":
    main()