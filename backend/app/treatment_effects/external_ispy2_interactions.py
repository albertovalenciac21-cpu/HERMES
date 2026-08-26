"""
HERMES 2.0
Locked I-SPY2 Pathway × Pembrolizumab Interaction Analysis
==========================================================

Purpose
-------
Run the prespecified external treatment-effect-modifier analysis in the frozen
I-SPY2 TNBC cohort after cohort construction and Hallmark pathway scoring have
been completed and audited.

The confirmatory external hypothesis family contains exactly the seven
Hallmark pathways locked before external outcome analysis:

    HALLMARK_COAGULATION
    HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION
    HALLMARK_ANGIOGENESIS
    HALLMARK_MYOGENESIS
    HALLMARK_APICAL_JUNCTION
    HALLMARK_TGF_BETA_SIGNALING
    HALLMARK_KRAS_SIGNALING_UP

For each pathway, HERMES reuses the validated interaction engine:

    logit[P(pCR = 1)] =
        beta_0
        + beta_T * treatment
        + beta_X * pathway
        + beta_TX * treatment * pathway

where:
    treatment = 0 -> paclitaxel control
    treatment = 1 -> paclitaxel + pembrolizumab

Primary external quantity
-------------------------
beta_TX, the treatment × pathway interaction coefficient.

Locked directional hypothesis
-----------------------------
Higher activity of each locked mesenchymal/stromal pathway is expected to be
associated with LOWER incremental pembrolizumab benefit. Therefore:

    expected beta_TX < 0

Multiplicity
------------
Benjamini-Hochberg FDR is applied across EXACTLY the seven locked pathways,
using the threshold stored in LOCKED_EXTERNAL_VALIDATION_PLAN.

Scientific guardrails
---------------------
* No pathway is selected, substituted, removed, or added based on I-SPY2
  outcomes.
* The full seven-pathway family is analyzed and exported regardless of
  statistical significance.
* The existing HERMES modifier-discovery implementation is reused rather than
  creating a second statistical engine.
* This is external research validation, not a clinical treatment rule.
* A concordant direction is not sufficient to establish a predictive
  biomarker.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.treatment_effects.external_ispy2_pathways import (
    ISPY2PathwayRepresentation,
    build_ispy2_pathway_representation,
)
from backend.app.treatment_effects.external_validation_plan import (
    LOCKED_EXTERNAL_VALIDATION_PLAN,
    hash_external_validation_plan,
    validate_external_validation_plan,
)
from backend.app.treatment_effects.modifier_discovery import (
    ModifierDiscoveryResult,
    discover_treatment_modifiers,
)


DEFAULT_OUTPUT_DIR = Path(
    "outputs/hermes2/external_validation/ispy2_interactions"
)


@dataclass
class ISPY2ExternalInteractionResult:
    """Complete locked seven-pathway external interaction result."""

    interaction_table: pd.DataFrame
    engine_result: ModifierDiscoveryResult
    summary: dict[str, Any]
    locked_pathways: tuple[str, ...]
    plan_sha256: str


def _locked_pathways() -> tuple[str, ...]:
    return tuple(
        LOCKED_EXTERNAL_VALIDATION_PLAN.locked_negative_pathway_hypotheses
    )


def _validate_locked_family(
    representation: ISPY2PathwayRepresentation,
) -> tuple[str, ...]:
    """
    Enforce the exact prespecified external hypothesis family.

    The representation may contain all 50 Hallmarks, but this confirmatory
    analysis is restricted to the seven pathways locked before external
    outcome analysis.
    """

    representation.validate()
    validate_external_validation_plan(
        LOCKED_EXTERNAL_VALIDATION_PLAN
    )

    locked = _locked_pathways()

    if len(locked) != 7:
        raise ValueError(
            "The locked external confirmatory family must contain exactly "
            f"7 pathways; found {len(locked)}."
        )

    if len(set(locked)) != len(locked):
        raise ValueError(
            "Locked external pathway hypotheses contain duplicates."
        )

    missing = sorted(
        set(locked) - set(representation.scores.columns)
    )
    if missing:
        raise ValueError(
            "The I-SPY2 pathway representation is missing locked pathways: "
            f"{missing}"
        )

    if tuple(representation.locked_pathways) != locked:
        raise ValueError(
            "The pathway representation lock does not match the frozen "
            "external-validation plan."
        )

    return locked


def run_locked_ispy2_interactions(
    *,
    representation: ISPY2PathwayRepresentation | None = None,
    max_iter: int | None = None,
) -> ISPY2ExternalInteractionResult:
    """
    Run the external interaction analysis across exactly seven locked pathways.
    """

    if representation is None:
        representation = build_ispy2_pathway_representation()

    locked = _validate_locked_family(representation)

    if max_iter is None:
        max_iter = int(
            LOCKED_EXTERNAL_VALIDATION_PLAN.max_iter
        )

    X_locked = representation.scores.loc[
        :,
        list(locked),
    ].copy()

    # Reuse the validated HERMES treatment-modifier engine. Because X_locked
    # contains exactly seven pathways, BH-FDR is computed over exactly the
    # prespecified external hypothesis family.
    engine_result = discover_treatment_modifiers(
        X_locked,
        representation.treatment,
        representation.outcome,
        fdr_threshold=(
            LOCKED_EXTERNAL_VALIDATION_PLAN.hypothesis_fdr_threshold
        ),
        max_iter=max_iter,
    )

    table = engine_result.modifier_table.copy()

    if set(table["feature"]) != set(locked):
        raise RuntimeError(
            "External interaction engine did not return exactly the locked "
            "seven-pathway hypothesis family."
        )

    if len(table) != len(locked):
        raise RuntimeError(
            "External interaction table row count differs from the locked "
            "hypothesis-family size."
        )

    # Directional hypothesis was frozen before external outcome analysis:
    # higher pathway activity -> lower incremental ICI benefit -> beta_TX < 0.
    table["expected_interaction_direction"] = (
        "greater_benefit_with_lower_pathway"
    )

    table["direction_concordant_with_locked_hypothesis"] = (
        table["interaction_coefficient"] < 0.0
    )

    table["interaction_ci_excludes_null"] = (
        (table["interaction_or_ci_upper"] < 1.0)
        | (table["interaction_or_ci_lower"] > 1.0)
    )

    table["locked_external_hypothesis"] = True
    table["posthoc_pathway_selection_used"] = False

    # Preserve the locked hypothesis order as a separate immutable reporting
    # field even though the engine ranks results by statistical evidence.
    locked_order = {
        pathway: index + 1
        for index, pathway in enumerate(locked)
    }
    table["locked_hypothesis_order"] = (
        table["feature"].map(locked_order).astype(int)
    )

    treatment = representation.treatment
    outcome = representation.outcome

    control_rate = float(
        outcome.loc[treatment.eq(0)].mean()
    )
    pembro_rate = float(
        outcome.loc[treatment.eq(1)].mean()
    )

    concordant_n = int(
        table["direction_concordant_with_locked_hypothesis"].sum()
    )

    summary: dict[str, Any] = {
        "cohort": "I-SPY2 GSE194040 TNBC pembrolizumab/control",
        "patients": int(representation.n_patients),
        "control_patients": int(treatment.eq(0).sum()),
        "pembrolizumab_patients": int(treatment.eq(1).sum()),
        "control_pcr_rate": control_rate,
        "pembrolizumab_pcr_rate": pembro_rate,
        "observed_absolute_pcr_difference": float(
            pembro_rate - control_rate
        ),
        "locked_hypotheses_requested": int(len(locked)),
        "locked_hypotheses_analyzed": int(len(table)),
        "all_models_converged": bool(
            engine_result.summary["all_models_converged"]
        ),
        "nominal_interaction_count": int(
            table["nominal_interaction"].sum()
        ),
        "fdr_interaction_count": int(
            table["fdr_significant_interaction"].sum()
        ),
        "direction_concordant_count": concordant_n,
        "direction_concordant_fraction": float(
            concordant_n / len(locked)
        ),
        "fdr_threshold": float(
            LOCKED_EXTERNAL_VALIDATION_PLAN.hypothesis_fdr_threshold
        ),
        "expected_direction": (
            "interaction_coefficient < 0; "
            "greater benefit with lower pathway activity"
        ),
        "multiple_testing_family": (
            "exactly seven pathways locked before external outcome analysis"
        ),
        "posthoc_pathway_selection_used": False,
        "direct_neotrip_raw_scale_transport_used": False,
        "clinical_claims_allowed": False,
        "external_validation_plan_sha256": (
            hash_external_validation_plan(
                LOCKED_EXTERNAL_VALIDATION_PLAN
            )
        ),
    }

    return ISPY2ExternalInteractionResult(
        interaction_table=table,
        engine_result=engine_result,
        summary=summary,
        locked_pathways=locked,
        plan_sha256=summary[
            "external_validation_plan_sha256"
        ],
    )


def export_locked_ispy2_interactions(
    result: ISPY2ExternalInteractionResult,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Export every locked pathway result regardless of significance.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked_path = output_dir / "locked_ispy2_interactions_ranked.csv"
    result.interaction_table.to_csv(
        ranked_path,
        index=False,
    )

    locked_order_table = (
        result.interaction_table
        .sort_values(
            "locked_hypothesis_order",
            ascending=True,
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    locked_order_path = (
        output_dir / "locked_ispy2_interactions_hypothesis_order.csv"
    )
    locked_order_table.to_csv(
        locked_order_path,
        index=False,
    )

    compact_columns = [
        "locked_hypothesis_order",
        "feature",
        "interaction_coefficient",
        "interaction_standard_error",
        "interaction_odds_ratio",
        "interaction_or_ci_lower",
        "interaction_or_ci_upper",
        "interaction_p_value",
        "interaction_fdr",
        "risk_difference_q25",
        "risk_difference_q75",
        "risk_difference_contrast",
        "interaction_direction",
        "expected_interaction_direction",
        "direction_concordant_with_locked_hypothesis",
        "nominal_interaction",
        "fdr_significant_interaction",
        "converged",
    ]

    compact_path = (
        output_dir / "locked_ispy2_interactions_compact.csv"
    )
    locked_order_table[
        compact_columns
    ].to_csv(
        compact_path,
        index=False,
    )

    summary_path = output_dir / "locked_ispy2_interaction_summary.json"
    summary_path.write_text(
        json.dumps(
            result.summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "analysis": "locked_ispy2_pathway_treatment_interactions",
        "cohort": result.summary["cohort"],
        "patients": result.summary["patients"],
        "locked_pathway_count": len(result.locked_pathways),
        "locked_pathways": list(result.locked_pathways),
        "external_validation_plan_sha256": result.plan_sha256,
        "fdr_family": "seven locked pathways only",
        "posthoc_pathway_selection_used": False,
        "all_results_exported_regardless_of_significance": True,
        "direct_neotrip_raw_scale_transport_used": False,
        "clinical_claims_allowed": False,
    }

    manifest_path = output_dir / "locked_ispy2_interaction_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "ranked_results": ranked_path,
        "hypothesis_order_results": locked_order_path,
        "compact_results": compact_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def summarize_locked_ispy2_interactions(
    result: ISPY2ExternalInteractionResult,
) -> None:
    """Print the complete locked external result in a conservative format."""

    s = result.summary

    print("=== HERMES 2.0 LOCKED I-SPY2 INTERACTION ANALYSIS ===")
    print()
    print(f"Patients: {s['patients']}")
    print(
        "Arms: "
        f"control={s['control_patients']}, "
        f"pembrolizumab={s['pembrolizumab_patients']}"
    )
    print(
        "pCR rates: "
        f"control={s['control_pcr_rate']:.4f}, "
        f"pembrolizumab={s['pembrolizumab_pcr_rate']:.4f}"
    )
    print(
        "Observed absolute pCR difference: "
        f"{s['observed_absolute_pcr_difference']:.4f}"
    )
    print()
    print(
        "Locked hypotheses analyzed: "
        f"{s['locked_hypotheses_analyzed']}/"
        f"{s['locked_hypotheses_requested']}"
    )
    print(
        "Direction concordant with locked NeoTRIP hypothesis: "
        f"{s['direction_concordant_count']}/"
        f"{s['locked_hypotheses_analyzed']} "
        f"({s['direction_concordant_fraction']:.3f})"
    )
    print(
        "Interactions: "
        f"nominal={s['nominal_interaction_count']}, "
        f"FDR={s['fdr_interaction_count']}"
    )
    print(
        "All models converged: "
        f"{s['all_models_converged']}"
    )
    print()

    display_columns = [
        "locked_hypothesis_order",
        "feature",
        "interaction_coefficient",
        "interaction_odds_ratio",
        "interaction_or_ci_lower",
        "interaction_or_ci_upper",
        "interaction_p_value",
        "interaction_fdr",
        "risk_difference_contrast",
        "direction_concordant_with_locked_hypothesis",
        "converged",
    ]

    display = (
        result.interaction_table
        .sort_values(
            "locked_hypothesis_order",
            kind="mergesort",
        )[
            display_columns
        ]
    )

    print(display.to_string(index=False))
    print()
    print("IMPORTANT:")
    print(
        "All seven pathways were prespecified before external outcome analysis "
        "and are reported regardless of significance."
    )
    print(
        "Directional concordance alone does not establish a predictive "
        "biomarker or clinical treatment rule."
    )
    print(
        "This is cross-platform, cross-ICI external research replication; "
        "it is not direct transport of NeoTRIP-fitted raw-scale coefficients."
    )


def main() -> None:
    result = run_locked_ispy2_interactions()

    summarize_locked_ispy2_interactions(
        result
    )

    generated = export_locked_ispy2_interactions(
        result
    )

    print()
    print(f"Artifacts written to: {DEFAULT_OUTPUT_DIR}")
    for name, path in generated.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()