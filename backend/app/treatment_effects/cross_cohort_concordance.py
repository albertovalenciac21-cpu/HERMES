"""
HERMES 2.0
NeoTRIP ↔ I-SPY2 Locked Cross-Cohort Concordance Analysis
========================================================

Purpose
-------
Compare treatment × pathway interaction estimates for the EXACT seven
prespecified external-validation pathways between:

    Discovery/reference cohort:
        NeoTRIP
        chemotherapy vs chemotherapy + atezolizumab

    External replication cohort:
        I-SPY2 GSE194040 TNBC
        paclitaxel control vs paclitaxel + pembrolizumab

This module is intentionally descriptive/replicative. It does NOT refit either
cohort, alter either cohort, select pathways, or substitute hypotheses after
viewing I-SPY2 outcomes.

Primary questions
-----------------
1. Do the seven interaction coefficients have the same sign across cohorts?
2. Are pathway interaction-effect rankings correlated across cohorts?
3. Are the I-SPY2 directions concordant with the previously locked negative
   mesenchymal/stromal hypothesis?

Important interpretation
------------------------
NeoTRIP and I-SPY2 differ in ICI agent, chemotherapy regimen, trial design
details, and transcriptomic platform. Pathway scores are standardized within
cohort, making coefficient direction and relative magnitude interpretable,
but this is NOT direct raw-scale coefficient transport and is NOT a formal
claim that the true causal interaction is identical across studies.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from backend.app.treatment_effects.external_validation_plan import (
    LOCKED_EXTERNAL_VALIDATION_PLAN,
    hash_external_validation_plan,
    validate_external_validation_plan,
)


DEFAULT_NEOTRIP_MODIFIER_PATH = Path(
    "outputs/hermes2/primary_neotrip/engine_outputs/modifier_discovery.csv"
)

DEFAULT_ISPY2_INTERACTION_PATH = Path(
    "outputs/hermes2/external_validation/ispy2_interactions/"
    "locked_ispy2_interactions_hypothesis_order.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "outputs/hermes2/external_validation/cross_cohort_concordance"
)


@dataclass
class CrossCohortConcordanceResult:
    comparison_table: pd.DataFrame
    summary: dict[str, Any]
    locked_pathways: tuple[str, ...]


def _locked_pathways() -> tuple[str, ...]:
    return tuple(
        LOCKED_EXTERNAL_VALIDATION_PLAN.locked_negative_pathway_hypotheses
    )


def _exact_sign_concordance_p_value(
    concordant: int,
    total: int,
) -> float:
    """
    Exact one-sided probability of >= observed sign matches under p=0.5.

    This is a descriptive global directional-concordance test. With only seven
    prespecified pathways it has limited power and must not be overinterpreted.
    """

    if total <= 0:
        raise ValueError("total must be positive.")
    if not (0 <= concordant <= total):
        raise ValueError("concordant must lie between 0 and total.")

    numerator = sum(
        comb(total, k)
        for k in range(concordant, total + 1)
    )

    return float(
        numerator / (2 ** total)
    )


def _load_table(
    path: str | Path,
    *,
    label: str,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} artifact not found: {path}"
        )

    table = pd.read_csv(path)

    # HERMES engine artifacts are written with the DataFrame index, producing
    # an unnamed CSV column. It is not scientifically meaningful here.
    unnamed = [
        column
        for column in table.columns
        if str(column).startswith("Unnamed:")
    ]
    if unnamed:
        table = table.drop(columns=unnamed)

    required = {
        "feature",
        "interaction_coefficient",
        "interaction_standard_error",
        "interaction_odds_ratio",
        "interaction_or_ci_lower",
        "interaction_or_ci_upper",
        "interaction_p_value",
    }

    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            f"{label} artifact is missing required columns: {missing}"
        )

    if table["feature"].duplicated().any():
        duplicates = sorted(
            table.loc[
                table["feature"].duplicated(keep=False),
                "feature",
            ].astype(str).unique().tolist()
        )
        raise ValueError(
            f"{label} artifact contains duplicate features: {duplicates}"
        )

    return table.copy()


def _restrict_to_locked_family(
    table: pd.DataFrame,
    *,
    label: str,
    locked: tuple[str, ...],
) -> pd.DataFrame:
    indexed = table.set_index(
        table["feature"].astype(str),
        drop=False,
    )

    missing = sorted(
        set(locked) - set(indexed.index)
    )
    if missing:
        raise ValueError(
            f"{label} artifact is missing locked pathways: {missing}"
        )

    restricted = indexed.loc[
        list(locked)
    ].copy()

    if len(restricted) != len(locked):
        raise RuntimeError(
            f"{label} locked-family restriction changed row count."
        )

    return restricted


def run_cross_cohort_concordance(
    *,
    neotrip_modifier_path: str | Path = DEFAULT_NEOTRIP_MODIFIER_PATH,
    ispy2_interaction_path: str | Path = DEFAULT_ISPY2_INTERACTION_PATH,
) -> CrossCohortConcordanceResult:
    """
    Compare the frozen seven-pathway NeoTRIP and I-SPY2 interaction estimates.
    """

    validate_external_validation_plan(
        LOCKED_EXTERNAL_VALIDATION_PLAN
    )

    locked = _locked_pathways()

    if len(locked) != 7:
        raise ValueError(
            "Cross-cohort confirmatory comparison requires exactly seven "
            f"locked pathways; found {len(locked)}."
        )

    neotrip = _restrict_to_locked_family(
        _load_table(
            neotrip_modifier_path,
            label="NeoTRIP modifier",
        ),
        label="NeoTRIP modifier",
        locked=locked,
    )

    ispy2 = _restrict_to_locked_family(
        _load_table(
            ispy2_interaction_path,
            label="I-SPY2 interaction",
        ),
        label="I-SPY2 interaction",
        locked=locked,
    )

    rows: list[dict[str, Any]] = []

    for order, pathway in enumerate(
        locked,
        start=1,
    ):
        n = neotrip.loc[pathway]
        e = ispy2.loc[pathway]

        beta_n = float(
            n["interaction_coefficient"]
        )
        beta_e = float(
            e["interaction_coefficient"]
        )

        sign_n = int(np.sign(beta_n))
        sign_e = int(np.sign(beta_e))

        rows.append(
            {
                "locked_hypothesis_order": order,
                "feature": pathway,
                "neotrip_interaction_coefficient": beta_n,
                "neotrip_interaction_standard_error": float(
                    n["interaction_standard_error"]
                ),
                "neotrip_interaction_odds_ratio": float(
                    n["interaction_odds_ratio"]
                ),
                "neotrip_interaction_or_ci_lower": float(
                    n["interaction_or_ci_lower"]
                ),
                "neotrip_interaction_or_ci_upper": float(
                    n["interaction_or_ci_upper"]
                ),
                "neotrip_interaction_p_value": float(
                    n["interaction_p_value"]
                ),
                "ispy2_interaction_coefficient": beta_e,
                "ispy2_interaction_standard_error": float(
                    e["interaction_standard_error"]
                ),
                "ispy2_interaction_odds_ratio": float(
                    e["interaction_odds_ratio"]
                ),
                "ispy2_interaction_or_ci_lower": float(
                    e["interaction_or_ci_lower"]
                ),
                "ispy2_interaction_or_ci_upper": float(
                    e["interaction_or_ci_upper"]
                ),
                "ispy2_interaction_p_value": float(
                    e["interaction_p_value"]
                ),
                "neotrip_negative_direction": bool(beta_n < 0.0),
                "ispy2_negative_direction": bool(beta_e < 0.0),
                "cross_cohort_sign_concordant": bool(
                    sign_n == sign_e
                    and sign_n != 0
                ),
                "ispy2_concordant_with_locked_negative_hypothesis": bool(
                    beta_e < 0.0
                ),
                "coefficient_difference_ispy2_minus_neotrip": float(
                    beta_e - beta_n
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    neotrip_beta = comparison[
        "neotrip_interaction_coefficient"
    ].to_numpy(dtype=float)

    ispy2_beta = comparison[
        "ispy2_interaction_coefficient"
    ].to_numpy(dtype=float)

    pearson = pearsonr(
        neotrip_beta,
        ispy2_beta,
    )
    spearman = spearmanr(
        neotrip_beta,
        ispy2_beta,
    )

    sign_matches = int(
        comparison[
            "cross_cohort_sign_concordant"
        ].sum()
    )

    locked_direction_matches = int(
        comparison[
            "ispy2_concordant_with_locked_negative_hypothesis"
        ].sum()
    )

    summary: dict[str, Any] = {
        "analysis": "NeoTRIP_vs_ISPY2_locked_pathway_concordance",
        "locked_pathways": int(len(locked)),
        "cross_cohort_sign_concordant_count": sign_matches,
        "cross_cohort_sign_concordant_fraction": float(
            sign_matches / len(locked)
        ),
        "cross_cohort_sign_concordance_exact_one_sided_p_value": (
            _exact_sign_concordance_p_value(
                sign_matches,
                len(locked),
            )
        ),
        "ispy2_locked_negative_direction_count": (
            locked_direction_matches
        ),
        "ispy2_locked_negative_direction_fraction": float(
            locked_direction_matches / len(locked)
        ),
        "pearson_interaction_coefficient_correlation": float(
            pearson.statistic
        ),
        "pearson_correlation_p_value": float(
            pearson.pvalue
        ),
        "spearman_interaction_coefficient_correlation": float(
            spearman.statistic
        ),
        "spearman_correlation_p_value": float(
            spearman.pvalue
        ),
        "mean_absolute_cross_cohort_coefficient_difference": float(
            comparison[
                "coefficient_difference_ispy2_minus_neotrip"
            ].abs().mean()
        ),
        "external_validation_plan_sha256": (
            hash_external_validation_plan(
                LOCKED_EXTERNAL_VALIDATION_PLAN
            )
        ),
        "posthoc_pathway_selection_used": False,
        "direct_raw_scale_coefficient_transport_claimed": False,
        "clinical_claims_allowed": False,
        "interpretation_guardrail": (
            "Cross-cohort concordance is descriptive biological replication "
            "across different ICI agents/platforms; absence of significance "
            "does not prove absence of heterogeneity, and concordance does "
            "not establish a validated predictive biomarker."
        ),
    }

    return CrossCohortConcordanceResult(
        comparison_table=comparison,
        summary=summary,
        locked_pathways=locked,
    )


def export_cross_cohort_concordance(
    result: CrossCohortConcordanceResult,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    table_path = output_dir / "locked_cross_cohort_concordance.csv"
    result.comparison_table.to_csv(
        table_path,
        index=False,
    )

    summary_path = output_dir / "locked_cross_cohort_concordance_summary.json"
    summary_path.write_text(
        json.dumps(
            result.summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "analysis": result.summary["analysis"],
        "locked_pathway_count": len(result.locked_pathways),
        "locked_pathways": list(result.locked_pathways),
        "posthoc_pathway_selection_used": False,
        "all_locked_pathways_reported": True,
        "direct_raw_scale_coefficient_transport_claimed": False,
        "clinical_claims_allowed": False,
        "external_validation_plan_sha256": (
            result.summary[
                "external_validation_plan_sha256"
            ]
        ),
    }

    manifest_path = output_dir / "locked_cross_cohort_concordance_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "comparison": table_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main() -> None:
    result = run_cross_cohort_concordance()
    s = result.summary

    print("=== HERMES 2.0 NEOTRIP ↔ I-SPY2 CONCORDANCE ===")
    print()
    print(
        "Cross-cohort interaction-sign concordance: "
        f"{s['cross_cohort_sign_concordant_count']}/"
        f"{s['locked_pathways']} "
        f"({s['cross_cohort_sign_concordant_fraction']:.3f})"
    )
    print(
        "Exact one-sided sign-concordance p-value: "
        f"{s['cross_cohort_sign_concordance_exact_one_sided_p_value']:.4f}"
    )
    print(
        "I-SPY2 concordance with locked negative hypothesis: "
        f"{s['ispy2_locked_negative_direction_count']}/"
        f"{s['locked_pathways']} "
        f"({s['ispy2_locked_negative_direction_fraction']:.3f})"
    )
    print()
    print(
        "Pearson beta correlation: "
        f"r={s['pearson_interaction_coefficient_correlation']:.3f}, "
        f"p={s['pearson_correlation_p_value']:.4f}"
    )
    print(
        "Spearman beta correlation: "
        f"rho={s['spearman_interaction_coefficient_correlation']:.3f}, "
        f"p={s['spearman_correlation_p_value']:.4f}"
    )
    print()

    display_columns = [
        "locked_hypothesis_order",
        "feature",
        "neotrip_interaction_coefficient",
        "ispy2_interaction_coefficient",
        "cross_cohort_sign_concordant",
        "ispy2_concordant_with_locked_negative_hypothesis",
    ]

    print(
        result.comparison_table[
            display_columns
        ].to_string(index=False)
    )

    print()
    print("IMPORTANT:")
    print(
        "This analysis compares the same seven pathways selected before "
        "external outcome analysis. No pathways were added or removed."
    )
    print(
        "NeoTRIP and I-SPY2 use different ICI agents and expression platforms; "
        "this is biological/methodological replication, not direct coefficient "
        "transport."
    )

    generated = export_cross_cohort_concordance(
        result
    )

    print()
    print(f"Artifacts written to: {DEFAULT_OUTPUT_DIR}")
    for name, path in generated.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()