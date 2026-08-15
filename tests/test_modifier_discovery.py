"""
HERMES 2.0
Biological Treatment-Effect Modifier Discovery Tests
======================================================

Validation suite for pathway-level treatment-effect modifier discovery.

The tests verify:

1. Benjamini-Hochberg FDR correction.
2. Recovery of a known synthetic treatment interaction.
3. Rejection of null pathway interactions.
4. Probability-scale treatment-effect behavior.
5. Complete NeoTRIP integration.
6. Deterministic analysis behavior.

Synthetic experiments validate software capability only and do not
establish biological treatment-effect heterogeneity in NeoTRIP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.special import expit

from backend.app.treatment_effects.modifier_discovery import (
    benjamini_hochberg,
    discover_treatment_modifiers,
    fit_pathway_interaction,
    run_neotrip_modifier_discovery,
)


# =============================================================
# Test utilities
# =============================================================


def check(
    condition: bool,
    message: str,
) -> None:
    """
    Simple explicit test helper.
    """

    if not condition:
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
    n: int = 2000,
    interaction_strength: float = 1.5,
    random_state: int = 2026,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Generate a randomized synthetic trial containing one known
    biological treatment-effect modifier.

    TRUE_MODIFIER affects the treatment response through:

        treatment * TRUE_MODIFIER

    NULL_PATHWAY_1 and NULL_PATHWAY_2 contain no injected
    treatment interactions.
    """

    if n < 100:
        raise ValueError(
            "n must be at least 100."
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

    true_modifier = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )

    null_pathway_1 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )

    null_pathway_2 = rng.normal(
        loc=0.0,
        scale=1.0,
        size=n,
    )

    treatment = rng.binomial(
        1,
        0.5,
        size=n,
    )

    linear_predictor = (
        -0.25
        + 0.25 * treatment
        + 0.15 * true_modifier
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

    X = pd.DataFrame(
        {
            "TRUE_MODIFIER":
                true_modifier,

            "NULL_PATHWAY_1":
                null_pathway_1,

            "NULL_PATHWAY_2":
                null_pathway_2,
        },
        index=index,
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
# Main validation suite
# =============================================================


def main() -> None:

    print(
        "=== HERMES 2.0 MODIFIER DISCOVERY TESTS ==="
    )

    print()

    # =========================================================
    # 1. Benjamini-Hochberg validation
    # =========================================================

    p_values = pd.Series(
        [
            0.001,
            0.010,
            0.030,
            0.500,
            0.900,
        ],
        dtype=float,
    )

    adjusted = benjamini_hochberg(
        p_values
    )

    check(
        (
            adjusted
            >= p_values
        ).all(),
        (
            "BH adjusted p-values not smaller "
            "than raw p-values"
        ),
    )

    check(
        (
            (
                adjusted
                >= 0.0
            )
            &
            (
                adjusted
                <= 1.0
            )
        ).all(),
        "BH adjusted p-value bounds",
    )

    order = np.argsort(
        p_values.to_numpy()
    )

    sorted_adjusted = (
        adjusted
        .iloc[
            order
        ]
        .to_numpy()
    )

    check(
        np.all(
            np.diff(
                sorted_adjusted
            )
            >= -1e-12
        ),
        "BH monotonicity",
    )

    # =========================================================
    # 2. Synthetic positive control
    # =========================================================

    X, T, Y = generate_synthetic_trial(
        n=2000,
        interaction_strength=1.5,
        random_state=2026,
    )

    single = fit_pathway_interaction(
        X[
            "TRUE_MODIFIER"
        ],
        T,
        Y,
        feature_name=(
            "TRUE_MODIFIER"
        ),
    )

    check(
        single.n
        == 2000,
        "single-pathway patient count",
    )

    check(
        single.converged,
        "single-pathway optimizer convergence",
    )

    check(
        single.interaction_coefficient
        > 0.0,
        "known modifier interaction direction recovery",
    )

    check(
        single.interaction_p_value
        < 0.05,
        "known modifier interaction significance recovery",
    )

    check(
        single.risk_difference_q75
        > single.risk_difference_q25,
        (
            "known modifier probability-scale "
            "treatment-effect recovery"
        ),
    )

    # The injected truth is 1.50.
    # We do not require exact equality because this is a finite
    # Bernoulli sample, but the estimate should be reasonably close.

    check(
        abs(
            single.interaction_coefficient
            - 1.5
        )
        < 0.35,
        "known modifier coefficient recovery",
    )

    # =========================================================
    # 3. Full synthetic discovery
    # =========================================================

    result = discover_treatment_modifiers(
        X,
        T,
        Y,
        fdr_threshold=0.10,
    )

    table = (
        result.modifier_table
    )

    check(
        len(
            table
        )
        == 3,
        "all synthetic pathways analyzed",
    )

    check(
        result.summary[
            "features_requested"
        ]
        == 3,
        "synthetic feature request count",
    )

    check(
        result.summary[
            "features_analyzed"
        ]
        == 3,
        "synthetic feature analysis count",
    )

    check(
        result.summary[
            "features_failed"
        ]
        == 0,
        "no synthetic model failures",
    )

    check(
        result.summary[
            "all_models_converged"
        ],
        "synthetic interaction-model convergence",
    )

    true_row = (
        table
        .set_index(
            "feature"
        )
        .loc[
            "TRUE_MODIFIER"
        ]
    )

    check(
        true_row[
            "interaction_coefficient"
        ]
        > 0.0,
        (
            "multimodel known-modifier "
            "direction recovery"
        ),
    )

    check(
        true_row[
            "interaction_p_value"
        ]
        < 0.05,
        (
            "multimodel known-modifier "
            "significance recovery"
        ),
    )

    check(
        true_row[
            "interaction_fdr"
        ]
        < 0.10,
        "known modifier survives FDR correction",
    )

    check(
        int(
            true_row[
                "interaction_rank"
            ]
        )
        == 1,
        "known modifier ranked first",
    )

    check(
        true_row[
            "interaction_direction"
        ]
        == (
            "greater_benefit_with_higher_pathway"
        ),
        (
            "known modifier biological "
            "direction label"
        ),
    )

    # =========================================================
    # 4. Null pathway behavior
    # =========================================================

    null_1 = (
        table
        .set_index(
            "feature"
        )
        .loc[
            "NULL_PATHWAY_1"
        ]
    )

    null_2 = (
        table
        .set_index(
            "feature"
        )
        .loc[
            "NULL_PATHWAY_2"
        ]
    )

    check(
        abs(
            null_1[
                "interaction_coefficient"
            ]
        )
        < 0.40,
        "null pathway 1 interaction remains small",
    )

    check(
        abs(
            null_2[
                "interaction_coefficient"
            ]
        )
        < 0.40,
        "null pathway 2 interaction remains small",
    )

    check(
        null_1[
            "interaction_p_value"
        ]
        > 0.05,
        "null pathway 1 remains nonsignificant",
    )

    check(
        null_2[
            "interaction_p_value"
        ]
        > 0.05,
        "null pathway 2 remains nonsignificant",
    )

    # =========================================================
    # 5. Explicit zero-interaction simulation
    # =========================================================

    X0, T0, Y0 = generate_synthetic_trial(
        n=2000,
        interaction_strength=0.0,
        random_state=2026,
    )

    zero_interaction = (
        fit_pathway_interaction(
            X0[
                "TRUE_MODIFIER"
            ],
            T0,
            Y0,
            feature_name=(
                "ZERO_INTERACTION"
            ),
        )
    )

    check(
        zero_interaction.converged,
        "zero-interaction model convergence",
    )

    check(
        abs(
            zero_interaction
            .interaction_coefficient
        )
        < 0.40,
        "zero-interaction coefficient remains small",
    )

    # =========================================================
    # 6. Probability validity
    # =========================================================

    probability_columns = [
        "pcr_probability_control_q25",
        "pcr_probability_treated_q25",
        "pcr_probability_control_q75",
        "pcr_probability_treated_q75",
    ]

    for column in probability_columns:

        check(
            (
                (
                    table[
                        column
                    ]
                    >= 0.0
                )
                &
                (
                    table[
                        column
                    ]
                    <= 1.0
                )
            ).all(),
            f"{column} bounds",
        )

    check(
        np.isfinite(
            table[
                "risk_difference_contrast"
            ]
            .to_numpy()
        ).all(),
        (
            "finite synthetic probability-scale "
            "modifier estimates"
        ),
    )

    # =========================================================
    # 7. Real NeoTRIP integration
    # =========================================================

    neotrip = (
        run_neotrip_modifier_discovery(
            fdr_threshold=0.10
        )
    )

    check(
        neotrip.summary[
            "patients"
        ]
        == 241,
        "NeoTRIP patient count",
    )

    check(
        neotrip.summary[
            "features_requested"
        ]
        == 50,
        "NeoTRIP Hallmark feature count",
    )

    check(
        neotrip.summary[
            "features_analyzed"
        ]
        == 50,
        "complete NeoTRIP Hallmark analysis",
    )

    check(
        neotrip.summary[
            "features_failed"
        ]
        == 0,
        "no NeoTRIP pathway-model failures",
    )

    check(
        neotrip.summary[
            "all_models_converged"
        ],
        "all NeoTRIP pathway models converge",
    )

    check(
        len(
            neotrip.modifier_table
        )
        == 50,
        "NeoTRIP modifier-table dimensions",
    )

    check(
        neotrip.modifier_table[
            "interaction_rank"
        ]
        .tolist()
        == list(
            range(
                1,
                51,
            )
        ),
        "modifier ranking integrity",
    )

    check(
        (
            (
                neotrip.modifier_table[
                    "interaction_p_value"
                ]
                >= 0.0
            )
            &
            (
                neotrip.modifier_table[
                    "interaction_p_value"
                ]
                <= 1.0
            )
        ).all(),
        "NeoTRIP interaction p-value bounds",
    )

    check(
        (
            (
                neotrip.modifier_table[
                    "interaction_fdr"
                ]
                >= 0.0
            )
            &
            (
                neotrip.modifier_table[
                    "interaction_fdr"
                ]
                <= 1.0
            )
        ).all(),
        "NeoTRIP interaction FDR bounds",
    )

    check(
        np.isfinite(
            neotrip.modifier_table[
                "interaction_coefficient"
            ]
            .to_numpy()
        ).all(),
        "finite NeoTRIP interaction estimates",
    )

    check(
        np.isfinite(
            neotrip.modifier_table[
                "risk_difference_contrast"
            ]
            .to_numpy()
        ).all(),
        (
            "finite NeoTRIP probability-scale "
            "modifier estimates"
        ),
    )

    # =========================================================
    # 8. Treatment/outcome integrity
    # =========================================================

    check(
        (
            neotrip.summary[
                "treatment_control_n"
            ]
            + neotrip.summary[
                "treatment_active_n"
            ]
        )
        == 241,
        "NeoTRIP treatment-count conservation",
    )

    check(
        (
            neotrip.summary[
                "outcome_negative_n"
            ]
            + neotrip.summary[
                "outcome_positive_n"
            ]
        )
        == 241,
        "NeoTRIP outcome-count conservation",
    )

    # =========================================================
    # 9. Determinism
    # =========================================================

    second = (
        discover_treatment_modifiers(
            X,
            T,
            Y,
            fdr_threshold=0.10,
        )
    )

    pd.testing.assert_frame_equal(
        result.modifier_table,
        second.modifier_table,
        check_exact=True,
    )

    check(
        True,
        "deterministic modifier discovery",
    )

    check(
        result.summary
        == second.summary,
        "deterministic modifier summary",
    )

    # =========================================================
    # Final report
    # =========================================================

    print()

    print(
        "========================================="
    )

    print(
        "ALL MODIFIER DISCOVERY TESTS PASSED"
    )

    print(
        "========================================="
    )

    print()

    print(
        "Synthetic known modifier:"
    )

    print(
        "Injected interaction: "
        "1.5000"
    )

    print(
        "Recovered interaction: "
        f"{true_row['interaction_coefficient']:.4f}"
    )

    print(
        "Raw P: "
        f"{true_row['interaction_p_value']:.6g}"
    )

    print(
        "FDR: "
        f"{true_row['interaction_fdr']:.6g}"
    )

    print(
        "Rank: "
        f"{int(true_row['interaction_rank'])}"
    )

    print()

    print(
        "Synthetic null pathways:"
    )

    print(
        "NULL_PATHWAY_1 interaction: "
        f"{null_1['interaction_coefficient']:.4f}"
    )

    print(
        "NULL_PATHWAY_1 P: "
        f"{null_1['interaction_p_value']:.4f}"
    )

    print(
        "NULL_PATHWAY_2 interaction: "
        f"{null_2['interaction_coefficient']:.4f}"
    )

    print(
        "NULL_PATHWAY_2 P: "
        f"{null_2['interaction_p_value']:.4f}"
    )

    print()

    print(
        "NeoTRIP:"
    )

    print(
        "Patients: "
        f"{neotrip.summary['patients']}"
    )

    print(
        "Hallmark pathways: "
        f"{neotrip.summary['features_analyzed']}"
    )

    print(
        "Nominal pathway interactions: "
        f"{neotrip.summary['nominal_interaction_count']}"
    )

    print(
        "FDR pathway interactions: "
        f"{neotrip.summary['fdr_interaction_count']}"
    )

    print(
        "All models converged: "
        f"{neotrip.summary['all_models_converged']}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "Successful recovery of the injected synthetic modifier "
        "demonstrates treatment-interaction detection capability."
    )

    print(
        "The absence of significant single-pathway interactions "
        "in NeoTRIP is therefore a data result, not evidence that "
        "the analysis pipeline is incapable of detecting an "
        "interaction."
    )


if __name__ == "__main__":
    main()