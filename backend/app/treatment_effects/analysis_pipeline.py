"""
HERMES 2.0
Unified Treatment-Effect Analysis Pipeline
==========================================

Purpose
-------
Provide one canonical, reproducible orchestration layer for the validated
HERMES 2.0 treatment-effect components.

The pipeline accepts an already constructed biological feature matrix and
coordinates:

    repeated cross-fitting
        -> patient-level treatment-effect estimates
        -> empirical resampling uncertainty
        -> pathway treatment-modifier discovery
        -> model / patient robustness analysis
        -> permutation-based heterogeneity null validation
        -> biological applicability / OOD diagnostics
        -> unified patient-level table
        -> exportable analysis artifacts

A convenience wrapper can construct the current NeoTRIP/Hallmark development
dataset through feature_builder.build_treatment_effect_dataset().

Scientific scope
----------------
This module is an orchestration and research-analysis layer. It does not turn
internal resampling diagnostics into causal confidence intervals, external
validation, a validated predictive biomarker, or a clinical treatment
recommendation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.treatment_effects.feature_builder import (
    DEFAULT_HALLMARK_GMT,
    TreatmentEffectDataset,
    build_treatment_effect_dataset,
)
from backend.app.treatment_effects.generalizability import (
    ApplicabilityAssessment,
    ApplicabilityReference,
    assess_applicability,
    fit_applicability_reference,
)
from backend.app.treatment_effects.modifier_discovery import (
    ModifierDiscoveryResult,
    discover_treatment_modifiers,
)
from backend.app.treatment_effects.permutation_test import (
    PermutationTestResult,
    run_permutation_test,
)
from backend.app.treatment_effects.repeated_crossfit import (
    RepeatedCrossFitResult,
    repeated_crossfit_treatment_effect_model,
)
from backend.app.treatment_effects.robustness import (
    TreatmentEffectRobustnessResult,
    run_treatment_effect_robustness,
)
from backend.app.treatment_effects.uncertainty import (
    TreatmentEffectUncertaintyResult,
    quantify_treatment_effect_uncertainty,
)


@dataclass(frozen=True)
class HermesAnalysisConfig:
    """Configuration for a canonical HERMES 2.0 analysis run."""

    n_repeats: int = 20
    n_splits: int = 5
    regularization_C: float = 0.10
    max_iter: int = 10000
    base_random_state: int = 42

    uncertainty_alpha: float = 0.05
    minimum_sign_stability: float = 0.90
    minimum_signal_uncertainty_ratio: float = 1.0

    run_modifier_discovery: bool = True
    modifier_fdr_threshold: float = 0.10

    run_robustness: bool = True
    robustness_C_values: tuple[float, ...] = (0.03, 0.10, 0.30)
    robustness_n_splits_values: tuple[int, ...] = (4, 5, 6)
    robustness_n_repeats: int = 5
    robustness_top_fraction: float = 0.25
    robustness_modifier_perturbations: int = 5
    robustness_modifier_subsample_fraction: float = 0.80
    robustness_base_random_state: int = 2026

    run_permutation: bool = True
    permutation_mode: str = "feature_permutation"
    n_permutations: int = 100
    permutation_n_repeats: int = 10
    permutation_base_random_state: int = 2026

    run_applicability: bool = True
    applicability_borderline_quantile: float = 0.95
    applicability_ood_quantile: float = 0.99


@dataclass
class HermesAnalysisResult:
    """Complete HERMES 2.0 analysis bundle."""

    patient_table: pd.DataFrame
    repeated_crossfit: RepeatedCrossFitResult
    uncertainty: TreatmentEffectUncertaintyResult
    modifiers: ModifierDiscoveryResult | None
    robustness: TreatmentEffectRobustnessResult | None
    permutation: PermutationTestResult | None
    applicability_reference: ApplicabilityReference | None
    applicability: ApplicabilityAssessment | None
    summary: dict[str, Any]
    config: HermesAnalysisConfig


def _validate_config(config: HermesAnalysisConfig) -> None:
    if config.n_repeats < 2:
        raise ValueError("n_repeats must be at least 2.")
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if config.regularization_C <= 0:
        raise ValueError("regularization_C must be positive.")
    if config.max_iter < 1:
        raise ValueError("max_iter must be positive.")
    if config.base_random_state < 0:
        raise ValueError("base_random_state must be non-negative.")
    if not 0.0 < config.uncertainty_alpha < 1.0:
        raise ValueError("uncertainty_alpha must be between 0 and 1.")
    if not 0.0 < config.modifier_fdr_threshold < 1.0:
        raise ValueError("modifier_fdr_threshold must be between 0 and 1.")
    if config.run_permutation and config.n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    if config.run_permutation and config.permutation_n_repeats < 2:
        raise ValueError("permutation_n_repeats must be at least 2.")


def _validate_inputs(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame | None]:
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame.")
    if X.index.has_duplicates:
        raise ValueError("X contains duplicate patient IDs.")

    X = X.astype(float).copy()
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("X contains non-finite values.")

    for values, name in ((treatment, "treatment"), (outcome, "outcome")):
        if not isinstance(values, pd.Series):
            raise TypeError(f"{name} must be a pandas Series.")
        if not X.index.equals(values.index):
            raise ValueError(f"X and {name} must have identical patient order.")
        observed = set(pd.unique(values.dropna()))
        if not observed.issubset({0, 1}) or len(observed) != 2:
            raise ValueError(f"{name} must contain both binary values 0 and 1.")

    treatment = treatment.astype(int).copy()
    outcome = outcome.astype(int).copy()

    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame):
            raise TypeError("metadata must be a pandas DataFrame or None.")
        if not X.index.equals(metadata.index):
            raise ValueError("X and metadata must have identical patient order.")
        metadata = metadata.copy()

    return X, treatment, outcome, metadata


def _prefixed(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = table.copy()
    frame.columns = [f"{prefix}{column}" for column in frame.columns]
    return frame


def build_unified_patient_table(
    *,
    patient_index: pd.Index,
    metadata: pd.DataFrame | None,
    repeated_crossfit: RepeatedCrossFitResult,
    uncertainty: TreatmentEffectUncertaintyResult,
    robustness: TreatmentEffectRobustnessResult | None,
    applicability: ApplicabilityAssessment | None,
) -> pd.DataFrame:
    """Join patient-level HERMES outputs without ambiguous column names."""

    table = pd.DataFrame(index=patient_index.copy())

    if metadata is not None:
        table = table.join(_prefixed(metadata, "metadata__"), how="left")

    table = table.join(
        _prefixed(repeated_crossfit.patient_summary, "crossfit__"),
        how="left",
    )
    table = table.join(
        _prefixed(uncertainty.patient_table, "uncertainty__"),
        how="left",
    )

    if robustness is not None:
        table = table.join(
            _prefixed(robustness.patient_robustness, "robustness__"),
            how="left",
        )

    if applicability is not None:
        table = table.join(
            _prefixed(applicability.patient_table, "applicability__"),
            how="left",
        )

    if not table.index.equals(patient_index):
        raise RuntimeError("Patient ordering changed while assembling HERMES outputs.")

    return table


def run_hermes_analysis(
    X: pd.DataFrame,
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    metadata: pd.DataFrame | None = None,
    config: HermesAnalysisConfig | None = None,
) -> HermesAnalysisResult:
    """Run the canonical HERMES 2.0 analysis stack on biological features."""

    if config is None:
        config = HermesAnalysisConfig()
    if not isinstance(config, HermesAnalysisConfig):
        raise TypeError("config must be a HermesAnalysisConfig.")

    _validate_config(config)
    X, treatment, outcome, metadata = _validate_inputs(
        X, treatment, outcome, metadata
    )

    repeated = repeated_crossfit_treatment_effect_model(
        X=X,
        treatment=treatment,
        outcome=outcome,
        n_repeats=config.n_repeats,
        n_splits=config.n_splits,
        C=config.regularization_C,
        max_iter=config.max_iter,
        base_random_state=config.base_random_state,
    )

    uncertainty = quantify_treatment_effect_uncertainty(
        repeated,
        alpha=config.uncertainty_alpha,
        minimum_sign_stability=config.minimum_sign_stability,
        minimum_signal_uncertainty_ratio=config.minimum_signal_uncertainty_ratio,
    )

    modifiers: ModifierDiscoveryResult | None = None
    if config.run_modifier_discovery:
        modifiers = discover_treatment_modifiers(
            X,
            treatment,
            outcome,
            fdr_threshold=config.modifier_fdr_threshold,
            max_iter=config.max_iter,
        )

    robustness: TreatmentEffectRobustnessResult | None = None
    if config.run_robustness:
        robustness = run_treatment_effect_robustness(
            X,
            treatment,
            outcome,
            C_values=config.robustness_C_values,
            n_splits_values=config.robustness_n_splits_values,
            n_repeats=config.robustness_n_repeats,
            max_iter=config.max_iter,
            base_random_state=config.base_random_state,
            top_fraction=config.robustness_top_fraction,
            n_modifier_perturbations=config.robustness_modifier_perturbations,
            modifier_subsample_fraction=config.robustness_modifier_subsample_fraction,
            modifier_base_random_state=config.robustness_base_random_state,
            fdr_threshold=config.modifier_fdr_threshold,
        )

    permutation: PermutationTestResult | None = None
    if config.run_permutation:
        permutation = run_permutation_test(
            X,
            treatment,
            outcome,
            permutation_mode=config.permutation_mode,
            n_permutations=config.n_permutations,
            n_repeats=config.permutation_n_repeats,
            n_splits=config.n_splits,
            C=config.regularization_C,
            max_iter=config.max_iter,
            base_random_state=config.permutation_base_random_state,
            observed_base_random_state=config.base_random_state,
        )

    applicability_reference: ApplicabilityReference | None = None
    applicability: ApplicabilityAssessment | None = None
    if config.run_applicability:
        applicability_reference = fit_applicability_reference(
            X,
            borderline_quantile=config.applicability_borderline_quantile,
            ood_quantile=config.applicability_ood_quantile,
        )
        applicability = assess_applicability(applicability_reference, X)

    patient_table = build_unified_patient_table(
        patient_index=X.index,
        metadata=metadata,
        repeated_crossfit=repeated,
        uncertainty=uncertainty,
        robustness=robustness,
        applicability=applicability,
    )

    summary: dict[str, Any] = {
        "patients": int(len(X)),
        "biological_features": int(X.shape[1]),
        "n_repeats": int(config.n_repeats),
        "n_splits": int(config.n_splits),
        "regularization_C": float(config.regularization_C),
        "cohort_mean_ite": float(repeated.patient_summary["mean_ite"].mean()),
        "cohort_median_ite": float(repeated.patient_summary["mean_ite"].median()),
        "mean_patient_ite_sd": float(repeated.patient_summary["ite_std"].mean()),
        "modifier_discovery_run": bool(modifiers is not None),
        "robustness_run": bool(robustness is not None),
        "permutation_run": bool(permutation is not None),
        "applicability_run": bool(applicability is not None),
        "analysis_scope": "research_internal_validation",
    }

    if "evidence_state" in uncertainty.patient_table.columns:
        evidence_counts = uncertainty.patient_table["evidence_state"].value_counts()
        for state, count in evidence_counts.items():
            summary[f"uncertainty_state__{state}"] = int(count)

    if modifiers is not None:
        summary["nominal_modifier_count"] = int(
            modifiers.summary.get("nominal_interaction_count", 0)
        )
        summary["fdr_modifier_count"] = int(
            modifiers.summary.get("fdr_interaction_count", 0)
        )

    if robustness is not None:
        summary["fraction_robust_patients"] = float(
            robustness.summary.get("fraction_robust_patients", float("nan"))
        )
        summary["mean_pairwise_ite_spearman"] = float(
            robustness.summary.get("mean_pairwise_ite_spearman", float("nan"))
        )

    if permutation is not None:
        for key, value in permutation.empirical_p_values.items():
            summary[f"permutation_p__{key}"] = float(value)

    if applicability is not None:
        summary["fraction_in_distribution"] = float(
            applicability.summary.get("fraction_in_distribution", float("nan"))
        )
        summary["fraction_out_of_distribution"] = float(
            applicability.summary.get("fraction_out_of_distribution", float("nan"))
        )

    return HermesAnalysisResult(
        patient_table=patient_table,
        repeated_crossfit=repeated,
        uncertainty=uncertainty,
        modifiers=modifiers,
        robustness=robustness,
        permutation=permutation,
        applicability_reference=applicability_reference,
        applicability=applicability,
        summary=summary,
        config=config,
    )


def run_neotrip_hermes_analysis(
    *,
    hallmark_gmt_path: Path = DEFAULT_HALLMARK_GMT,
    min_genes: int = 3,
    min_coverage: float = 0.5,
    config: HermesAnalysisConfig | None = None,
) -> tuple[TreatmentEffectDataset, HermesAnalysisResult]:
    """Build the current NeoTRIP/Hallmark dataset and run HERMES 2.0."""

    dataset = build_treatment_effect_dataset(
        hallmark_gmt_path=hallmark_gmt_path,
        min_genes=min_genes,
        min_coverage=min_coverage,
    )

    result = run_hermes_analysis(
        dataset.X,
        dataset.T,
        dataset.Y,
        metadata=dataset.metadata,
        config=config,
    )

    result.summary["dataset_source"] = "NeoTRIP_baseline"
    result.summary["representation_note"] = (
        "Current NeoTRIP/Hallmark development representation. Frozen-reference "
        "transport remains required for independent external-cohort deployment."
    )

    return dataset, result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def export_hermes_analysis(
    result: HermesAnalysisResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export reproducible tabular HERMES analysis artifacts."""

    if not isinstance(result, HermesAnalysisResult):
        raise TypeError("result must be a HermesAnalysisResult.")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}

    def save_frame(name: str, frame: pd.DataFrame) -> None:
        path = output / f"{name}.csv"
        frame.to_csv(path)
        exported[name] = path

    save_frame("patient_level_results", result.patient_table)
    save_frame("ite_by_repeat", result.repeated_crossfit.ite_by_repeat)
    save_frame("repeat_summary", result.repeated_crossfit.repeat_summary)
    save_frame("uncertainty", result.uncertainty.patient_table)

    if result.modifiers is not None:
        save_frame("modifier_discovery", result.modifiers.modifier_table)

    if result.robustness is not None:
        save_frame("robustness_scenarios", result.robustness.scenario_summary)
        save_frame("patient_robustness", result.robustness.patient_robustness)
        save_frame("robustness_pairwise", result.robustness.pairwise_scenario_comparison)
        save_frame("modifier_robustness", result.robustness.modifier_robustness)

    if result.permutation is not None:
        save_frame("permutation_null_statistics", result.permutation.null_statistics)
        permutation_summary = pd.DataFrame(
            {
                "observed": result.permutation.observed_statistics,
                "empirical_p_value": result.permutation.empirical_p_values,
            }
        )
        save_frame("permutation_observed_vs_null", permutation_summary)

    if result.applicability is not None:
        save_frame("applicability", result.applicability.patient_table)
        save_frame("applicability_feature_z_scores", result.applicability.feature_z_scores)

    summary_path = output / "analysis_summary.json"
    payload = {
        "summary": _json_safe(result.summary),
        "config": _json_safe(asdict(result.config)),
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    exported["analysis_summary"] = summary_path

    return exported