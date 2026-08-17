"""
HERMES 2.0
Primary NeoTRIP Scientific Analysis
===================================

Purpose
-------
Run the prespecified primary HERMES 2.0 scientific analysis on the locked
NeoTRIP baseline cohort after the cohort/science audit has passed.

This module deliberately separates:

    1. the frozen HERMES 2.0 computational engine;
    2. the locked NeoTRIP cohort/audit;
    3. the primary scientific-analysis configuration;
    4. publication-oriented exported results.

Primary scientific question
---------------------------
Does pretreatment tumor biological state modify the incremental probability
of pathologic complete response (pCR) associated with adding atezolizumab to
chemotherapy in randomized NeoTRIP patients?

Primary biological representation
---------------------------------
MSigDB Hallmark pathway activity.

Primary heterogeneity-null experiment
-------------------------------------
Feature-profile permutation. This preserves the observed treatment/outcome
relationship and the correlation structure among biological features while
destroying patient-to-biology correspondence.

IMPORTANT
---------
Outputs remain internal research estimates. Repeated-cross-fit intervals are
resampling uncertainty summaries rather than formal causal confidence
intervals. Modifier discovery is exploratory and requires robustness,
biological validation, and ideally external validation before biomarker or
clinical claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.treatment_effects.analysis_pipeline import (
    HermesAnalysisConfig,
    HermesAnalysisResult,
    export_hermes_analysis,
    run_neotrip_hermes_analysis,
)
from backend.app.treatment_effects.feature_builder import (
    TreatmentEffectDataset,
)
from backend.app.treatment_effects.neotrip_science_audit import (
    NeoTRIPScienceAudit,
    export_locked_neotrip_science_audit,
    run_locked_neotrip_science_audit,
)


@dataclass(frozen=True)
class PrimaryNeoTRIPAnalysisPlan:
    """Prespecified HERMES 2.0 primary NeoTRIP scientific-analysis plan."""

    plan_name: str = "hermes2_neotrip_primary_locked_v1"
    engine_tag: str = "hermes-2.0-engine-v1.0"

    n_repeats: int = 100
    n_splits: int = 5
    regularization_C: float = 0.10
    max_iter: int = 10000
    base_random_state: int = 42

    uncertainty_alpha: float = 0.05
    minimum_sign_stability: float = 0.90
    minimum_signal_uncertainty_ratio: float = 1.0

    modifier_fdr_threshold: float = 0.10

    robustness_C_values: tuple[float, ...] = (0.03, 0.10, 0.30)
    robustness_n_splits_values: tuple[int, ...] = (4, 5, 6)
    robustness_n_repeats: int = 10
    robustness_top_fraction: float = 0.25
    robustness_modifier_perturbations: int = 10
    robustness_modifier_subsample_fraction: float = 0.80
    robustness_base_random_state: int = 2026

    permutation_mode: str = "feature_permutation"
    n_permutations: int = 1000
    permutation_n_repeats: int = 10
    permutation_base_random_state: int = 2026

    applicability_borderline_quantile: float = 0.95
    applicability_ood_quantile: float = 0.99

    primary_endpoint: str = "pCR"
    primary_treatment_contrast: str = "CT/A minus CT"
    primary_representation: str = "MSigDB Hallmark pathways"
    primary_estimand: str = (
        "individualized incremental probability of pCR from adding "
        "atezolizumab to chemotherapy"
    )

    analysis_scope: str = "research_internal_validation"
    external_validation_required: bool = True


PRIMARY_NEOTRIP_PLAN = PrimaryNeoTRIPAnalysisPlan()


@dataclass
class PrimaryNeoTRIPAnalysisResult:
    """Complete primary NeoTRIP scientific-analysis bundle."""

    plan: PrimaryNeoTRIPAnalysisPlan
    plan_sha256: str
    audit: NeoTRIPScienceAudit
    dataset: TreatmentEffectDataset
    hermes: HermesAnalysisResult
    science_summary: dict[str, Any]
    top_modifiers: pd.DataFrame


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def hash_analysis_plan(
    plan: PrimaryNeoTRIPAnalysisPlan,
) -> str:
    """Return a deterministic SHA-256 fingerprint of the analysis plan."""

    payload = asdict(plan)
    return sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def plan_to_engine_config(
    plan: PrimaryNeoTRIPAnalysisPlan,
) -> HermesAnalysisConfig:
    """Translate the locked science plan into the validated engine config."""

    return HermesAnalysisConfig(
        n_repeats=plan.n_repeats,
        n_splits=plan.n_splits,
        regularization_C=plan.regularization_C,
        max_iter=plan.max_iter,
        base_random_state=plan.base_random_state,
        uncertainty_alpha=plan.uncertainty_alpha,
        minimum_sign_stability=plan.minimum_sign_stability,
        minimum_signal_uncertainty_ratio=(
            plan.minimum_signal_uncertainty_ratio
        ),
        run_modifier_discovery=True,
        modifier_fdr_threshold=plan.modifier_fdr_threshold,
        run_robustness=True,
        robustness_C_values=plan.robustness_C_values,
        robustness_n_splits_values=plan.robustness_n_splits_values,
        robustness_n_repeats=plan.robustness_n_repeats,
        robustness_top_fraction=plan.robustness_top_fraction,
        robustness_modifier_perturbations=(
            plan.robustness_modifier_perturbations
        ),
        robustness_modifier_subsample_fraction=(
            plan.robustness_modifier_subsample_fraction
        ),
        robustness_base_random_state=plan.robustness_base_random_state,
        run_permutation=True,
        permutation_mode=plan.permutation_mode,
        n_permutations=plan.n_permutations,
        permutation_n_repeats=plan.permutation_n_repeats,
        permutation_base_random_state=plan.permutation_base_random_state,
        run_applicability=True,
        applicability_borderline_quantile=(
            plan.applicability_borderline_quantile
        ),
        applicability_ood_quantile=plan.applicability_ood_quantile,
    )


def _rank_top_modifiers(
    result: HermesAnalysisResult,
    *,
    n: int = 15,
) -> pd.DataFrame:
    if result.modifiers is None:
        return pd.DataFrame()

    table = result.modifiers.modifier_table.copy()

    preferred_sort = [
        column
        for column in (
            "interaction_fdr",
            "interaction_p_value",
            "feature",
        )
        if column in table.columns
    ]

    if preferred_sort:
        table = table.sort_values(
            preferred_sort,
            ascending=True,
            kind="mergesort",
        )

    return table.head(n).copy()


def _build_science_summary(
    *,
    plan: PrimaryNeoTRIPAnalysisPlan,
    audit: NeoTRIPScienceAudit,
    dataset: TreatmentEffectDataset,
    result: HermesAnalysisResult,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "plan_name": plan.plan_name,
        "engine_tag": plan.engine_tag,
        "patients": int(dataset.n_patients),
        "biological_features": int(dataset.n_features),
        "observed_pcr_rate_CT": float(
            audit.cohort_summary["pcr_rate_CT"]
        ),
        "observed_pcr_rate_CT_A": float(
            audit.cohort_summary["pcr_rate_CT_A"]
        ),
        "observed_absolute_pcr_difference_CT_A_minus_CT": float(
            audit.cohort_summary[
                "absolute_pcr_rate_difference_CT_A_minus_CT"
            ]
        ),
        "hermes_cohort_mean_ite": float(
            result.summary["cohort_mean_ite"]
        ),
        "hermes_cohort_median_ite": float(
            result.summary["cohort_median_ite"]
        ),
        "mean_patient_ite_sd": float(
            result.summary["mean_patient_ite_sd"]
        ),
        "nominal_modifier_count": int(
            result.summary.get("nominal_modifier_count", 0)
        ),
        "fdr_modifier_count": int(
            result.summary.get("fdr_modifier_count", 0)
        ),
        "analysis_scope": plan.analysis_scope,
        "external_validation_required": (
            plan.external_validation_required
        ),
    }

    for key, value in result.summary.items():
        if key.startswith("uncertainty_state__"):
            summary[key] = int(value)

    for key, value in result.summary.items():
        if key.startswith("permutation_p__"):
            summary[key] = float(value)

    for key in (
        "fraction_robust_patients",
        "mean_pairwise_ite_spearman",
        "fraction_in_distribution",
        "fraction_out_of_distribution",
    ):
        if key in result.summary:
            summary[key] = float(result.summary[key])

    return summary


def _validate_primary_result(
    audit: NeoTRIPScienceAudit,
    dataset: TreatmentEffectDataset,
    result: HermesAnalysisResult,
) -> None:
    if not all(audit.integrity_checks.values()):
        failed = [
            name
            for name, passed in audit.integrity_checks.items()
            if not passed
        ]
        raise RuntimeError(
            "Locked NeoTRIP science audit failed before primary analysis: "
            + ", ".join(failed)
        )

    dataset.validate()

    if dataset.n_patients != 241:
        raise RuntimeError(
            f"Expected 241 locked NeoTRIP patients; found {dataset.n_patients}."
        )

    if dataset.n_features != 50:
        raise RuntimeError(
            f"Expected 50 Hallmark features; found {dataset.n_features}."
        )

    if not result.patient_table.index.equals(dataset.X.index):
        raise RuntimeError(
            "Primary HERMES patient output changed locked patient ordering."
        )

    if not np.isfinite(
        result.repeated_crossfit.ite_by_repeat.to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            "Primary HERMES ITE matrix contains non-finite values."
        )

    if result.modifiers is None:
        raise RuntimeError("Primary modifier discovery was not run.")
    if result.robustness is None:
        raise RuntimeError("Primary robustness analysis was not run.")
    if result.permutation is None:
        raise RuntimeError("Primary permutation analysis was not run.")
    if result.applicability is None:
        raise RuntimeError("Primary applicability analysis was not run.")


def run_primary_neotrip_analysis(
    *,
    plan: PrimaryNeoTRIPAnalysisPlan = PRIMARY_NEOTRIP_PLAN,
) -> PrimaryNeoTRIPAnalysisResult:
    """Run the locked primary HERMES 2.0 NeoTRIP scientific analysis."""

    audit = run_locked_neotrip_science_audit()

    config = plan_to_engine_config(plan)

    dataset, result = run_neotrip_hermes_analysis(
        min_genes=3,
        min_coverage=0.50,
        config=config,
    )

    _validate_primary_result(
        audit,
        dataset,
        result,
    )

    top_modifiers = _rank_top_modifiers(
        result,
        n=15,
    )

    science_summary = _build_science_summary(
        plan=plan,
        audit=audit,
        dataset=dataset,
        result=result,
    )

    return PrimaryNeoTRIPAnalysisResult(
        plan=plan,
        plan_sha256=hash_analysis_plan(plan),
        audit=audit,
        dataset=dataset,
        hermes=result,
        science_summary=science_summary,
        top_modifiers=top_modifiers,
    )


def export_primary_neotrip_analysis(
    result: PrimaryNeoTRIPAnalysisResult,
    output_dir: str | Path = "outputs/hermes2/primary_neotrip",
) -> dict[str, Path]:
    """Export the complete primary-analysis record and engine artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts: dict[str, Path] = {}

    engine_dir = output_dir / "engine_outputs"
    engine_artifacts = export_hermes_analysis(
        result.hermes,
        engine_dir,
    )
    for key, path in engine_artifacts.items():
        artifacts[f"engine__{key}"] = path

    audit_dir = output_dir / "locked_cohort_audit"
    audit_artifacts = export_locked_neotrip_science_audit(
        result.audit,
        audit_dir,
    )
    for key, path in audit_artifacts.items():
        artifacts[f"audit__{key}"] = path

    top_modifiers_path = output_dir / "top_15_treatment_modifiers.csv"
    result.top_modifiers.to_csv(
        top_modifiers_path,
        index=False,
    )
    artifacts["top_modifiers"] = top_modifiers_path

    plan_path = output_dir / "primary_analysis_plan.json"
    plan_path.write_text(
        json.dumps(
            asdict(result.plan),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["primary_analysis_plan"] = plan_path

    summary_path = output_dir / "primary_science_summary.json"
    summary_path.write_text(
        json.dumps(
            result.science_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["primary_science_summary"] = summary_path

    manifest = {
        "plan_name": result.plan.plan_name,
        "plan_sha256": result.plan_sha256,
        "engine_tag": result.plan.engine_tag,
        "analysis_scope": result.plan.analysis_scope,
        "patients": int(result.dataset.n_patients),
        "biological_features": int(result.dataset.n_features),
        "all_locked_audit_checks_passed": bool(
            all(result.audit.integrity_checks.values())
        ),
        "engine_artifact_count": int(len(engine_artifacts)),
        "audit_artifact_count": int(len(audit_artifacts)),
    }

    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["analysis_manifest"] = manifest_path

    return artifacts


def summarize_primary_neotrip_analysis(
    result: PrimaryNeoTRIPAnalysisResult,
) -> None:
    """Print a compact primary scientific-analysis summary."""

    s = result.science_summary

    print("=== HERMES 2.0 PRIMARY NEOTRIP ANALYSIS ===")
    print()
    print(f"Plan: {result.plan.plan_name}")
    print(f"Plan SHA256: {result.plan_sha256}")
    print(f"Engine: {result.plan.engine_tag}")
    print()

    print(f"Patients: {s['patients']}")
    print(f"Hallmark features: {s['biological_features']}")
    print(
        "Observed pCR rates: "
        f"CT={s['observed_pcr_rate_CT']:.4f}, "
        f"CT/A={s['observed_pcr_rate_CT_A']:.4f}"
    )
    print(
        "Observed absolute pCR difference (CT/A - CT): "
        f"{s['observed_absolute_pcr_difference_CT_A_minus_CT']:.4f}"
    )
    print()

    print(
        "HERMES cohort mean ITE: "
        f"{s['hermes_cohort_mean_ite']:.6f}"
    )
    print(
        "HERMES cohort median ITE: "
        f"{s['hermes_cohort_median_ite']:.6f}"
    )
    print(
        "Mean patient ITE SD across repeated cross-fitting: "
        f"{s['mean_patient_ite_sd']:.6f}"
    )
    print()

    print(
        "Treatment-modifier interactions: "
        f"nominal={s['nominal_modifier_count']}, "
        f"FDR={s['fdr_modifier_count']}"
    )

    permutation_items = {
        key: value
        for key, value in s.items()
        if key.startswith("permutation_p__")
    }
    if permutation_items:
        print()
        print("Permutation empirical p-values:")
        for key, value in sorted(permutation_items.items()):
            print(f"  {key}: {value:.6f}")

    if not result.top_modifiers.empty:
        print()
        print("Top treatment-modifier candidates:")
        display_columns = [
            column
            for column in (
                "feature",
                "interaction_coefficient",
                "interaction_odds_ratio",
                "interaction_p_value",
                "interaction_fdr",
                "risk_difference_contrast",
            )
            if column in result.top_modifiers.columns
        ]
        print(
            result.top_modifiers[
                display_columns
            ].head(15).to_string(
                index=False
            )
        )

    print()
    print("IMPORTANT:")
    print(
        "These are internal research estimates from the locked NeoTRIP "
        "primary analysis."
    )
    print(
        "Interaction candidates are not validated predictive biomarkers "
        "and patient-level ITEs are not clinical recommendations."
    )
    print(
        "External/orthogonal validation and biological interpretation "
        "remain required."
    )


def main() -> None:
    result = run_primary_neotrip_analysis()

    summarize_primary_neotrip_analysis(
        result
    )

    output_dir = Path(
        "outputs/hermes2/primary_neotrip"
    )

    artifacts = export_primary_neotrip_analysis(
        result,
        output_dir,
    )

    print()
    print(
        f"Primary-analysis artifacts written to: {output_dir}"
    )
    print(
        f"Artifacts: {len(artifacts)}"
    )


if __name__ == "__main__":
    main()