"""
HERMES 2.0
Primary Results Report Tests
============================

Tests the read-only reporting layer without refitting HERMES.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.treatment_effects.primary_results_report import (
    build_applicability_state_table,
    build_modifier_reporting_table,
    build_permutation_reporting_table,
    build_primary_results_table,
    build_uncertainty_state_table,
    generate_primary_results_report,
    load_primary_results,
)


def _write_fixture(root: Path) -> Path:
    results = root / "primary_neotrip"
    engine = results / "engine_outputs"
    engine.mkdir(parents=True)

    patients = ["P1", "P2", "P3"]

    manifest = {
        "all_locked_audit_checks_passed": True,
        "analysis_scope": "research_internal_validation",
        "biological_features": 50,
        "engine_tag": "hermes-2.0-engine-v1.0",
        "patients": 241,
        "plan_name": "hermes2_neotrip_primary_locked_v1",
        "plan_sha256": "a" * 64,
    }

    summary = {
        "patients": 241,
        "biological_features": 50,
        "observed_pcr_rate_CT": 0.47,
        "observed_pcr_rate_CT_A": 0.53,
        "observed_absolute_pcr_difference_CT_A_minus_CT": 0.06,
        "hermes_cohort_mean_ite": 0.04,
        "hermes_cohort_median_ite": 0.05,
        "mean_patient_ite_sd": 0.05,
        "mean_pairwise_ite_spearman": 0.94,
        "fraction_robust_patients": 0.75,
        "fraction_in_distribution": 0.88,
        "fraction_out_of_distribution": 0.03,
        "nominal_modifier_count": 0,
        "fdr_modifier_count": 0,
    }

    plan = {
        "plan_name": manifest["plan_name"],
        "n_repeats": 100,
        "n_permutations": 1000,
    }

    (results / "analysis_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (results / "primary_science_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (results / "primary_analysis_plan.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )

    patient_rows = []
    uncertainty_rows = []
    robustness_rows = []
    applicability_rows = []

    states = ["likely_benefit", "indeterminate", "likely_harm"]
    app_states = ["in_distribution", "in_distribution", "out_of_distribution"]

    for i, patient in enumerate(patients):
        mean_ite = [0.10, 0.02, -0.06][i]
        ite_std = [0.03, 0.04, 0.02][i]

        patient_rows.append(
            {
                "Patient_ID": patient,
                "crossfit__mean_ite": mean_ite,
            }
        )
        uncertainty_rows.append(
            {
                "Patient_ID": patient,
                "mean_ite": mean_ite,
                "median_ite": mean_ite,
                "ite_std": ite_std,
                "ite_lower": mean_ite - 0.05,
                "ite_upper": mean_ite + 0.05,
                "interval_width": 0.10,
                "fraction_positive": 0.8,
                "fraction_negative": 0.2,
                "fraction_zero": 0.0,
                "sign_stability": 0.8,
                "signal_uncertainty_ratio": abs(mean_ite) / ite_std,
                "interval_excludes_zero": False,
                "evidence_state": states[i],
            }
        )
        robustness_rows.append(
            {
                "Patient_ID": patient,
                "mean_ite": mean_ite,
                "ite_sensitivity_sd": 0.01,
                "robustness_state": "robust",
            }
        )
        applicability_rows.append(
            {
                "Patient_ID": patient,
                "mahalanobis_distance": 1.0 + i,
                "mahalanobis_reference_percentile": 0.2 + 0.2 * i,
                "max_abs_z": 1.0 + 0.1 * i,
                "max_abs_z_reference_percentile": 0.3 + 0.2 * i,
                "mean_abs_z": 0.5,
                "mean_abs_z_reference_percentile": 0.4,
                "fraction_features_abs_z_gt_2": 0.0,
                "fraction_features_abs_z_gt_3": 0.0,
                "n_ood_flags": int(i == 2),
                "n_borderline_flags": 0,
                "applicability_state": app_states[i],
                "applicability_score": 0.8 - 0.2 * i,
            }
        )

    # The production validator expects the locked 241-patient package.
    # Expand fixture patient tables deterministically to 241 unique IDs.
    def expand(frame: pd.DataFrame) -> pd.DataFrame:
        base = frame.copy()
        rows = []
        for i in range(241):
            source = base.iloc[i % len(base)].copy()
            source["Patient_ID"] = f"P{i:03d}"
            rows.append(source)
        return pd.DataFrame(rows)

    expand(pd.DataFrame(patient_rows)).to_csv(
        engine / "patient_level_results.csv", index=False
    )
    expand(pd.DataFrame(uncertainty_rows)).to_csv(
        engine / "uncertainty.csv", index=False
    )
    expand(pd.DataFrame(robustness_rows)).to_csv(
        engine / "patient_robustness.csv", index=False
    )
    expand(pd.DataFrame(applicability_rows)).to_csv(
        engine / "applicability.csv", index=False
    )

    modifiers = []
    for i in range(50):
        modifiers.append(
            {
                "feature": f"HALLMARK_{i:02d}",
                "interaction_coefficient": -0.2 + i * 0.005,
                "interaction_standard_error": 0.2,
                "interaction_odds_ratio": np.exp(-0.2 + i * 0.005),
                "interaction_or_ci_lower": np.exp(-0.6 + i * 0.005),
                "interaction_or_ci_upper": np.exp(0.2 + i * 0.005),
                "interaction_p_value": 0.10 + i * 0.01,
                "interaction_fdr": 0.90,
                "risk_difference_contrast": -0.05,
                "interaction_direction": "greater_benefit_with_lower_pathway",
                "nominal_interaction": False,
                "fdr_significant_interaction": False,
                "interaction_rank": i + 1,
            }
        )
    pd.DataFrame(modifiers).to_csv(
        engine / "modifier_discovery.csv", index=False
    )

    pd.DataFrame(
        {
            "feature": [f"HALLMARK_{i:02d}" for i in range(50)],
            "interaction_sign_stability": [1.0] * 50,
        }
    ).to_csv(engine / "modifier_robustness.csv", index=False)

    pd.DataFrame(
        {
            "scenario": ["C_0p03__splits_5", "C_0p1__splits_5"],
            "regularization_C": [0.03, 0.10],
            "n_splits": [5, 5],
            "n_repeats": [10, 10],
            "cohort_mean_ite": [0.03, 0.04],
        }
    ).to_csv(engine / "robustness_scenarios.csv", index=False)

    pd.DataFrame(
        {
            "scenario_a": ["A"],
            "scenario_b": ["B"],
            "spearman_ite": [0.95],
        }
    ).to_csv(engine / "robustness_pairwise.csv", index=False)

    null = pd.DataFrame(
        {
            "permutation": np.arange(1, 21),
            "ite_sd_across_patients": np.linspace(0.08, 0.12, 20),
            "ite_iqr": np.linspace(0.10, 0.16, 20),
            "fraction_sign_stability_ge_90pct": np.linspace(0.50, 0.70, 20),
        }
    )
    null.to_csv(engine / "permutation_null_statistics.csv", index=False)

    pd.DataFrame(
        {
            "statistic": [
                "ite_sd_across_patients",
                "ite_iqr",
                "fraction_sign_stability_ge_90pct",
            ],
            "observed": [0.10, 0.13, 0.65],
            "empirical_p_value": [0.50, 0.50, 0.25],
        }
    ).to_csv(engine / "permutation_observed_vs_null.csv", index=False)

    pd.DataFrame(
        {
            "repeat": [1, 2, 3],
            "random_state": [1, 2, 3],
            "oof_auc": [0.65, 0.67, 0.66],
            "oof_brier": [0.24, 0.23, 0.235],
            "mean_ite": [0.03, 0.04, 0.05],
            "median_ite": [0.03, 0.04, 0.05],
            "ite_std": [0.10, 0.11, 0.09],
            "minimum_ite": [-0.2, -0.2, -0.2],
            "maximum_ite": [0.2, 0.2, 0.2],
            "fraction_positive": [0.6, 0.65, 0.62],
            "fraction_negative": [0.4, 0.35, 0.38],
        }
    ).to_csv(engine / "repeat_summary.csv", index=False)

    return results


def test_load_and_validate_locked_reporting_package(tmp_path) -> None:
    results = _write_fixture(tmp_path)
    artifacts = load_primary_results(results)

    assert artifacts.manifest["patients"] == 241
    assert len(artifacts.patient_results) == 241
    assert len(artifacts.modifiers) == 50


def test_primary_results_summary_table(tmp_path) -> None:
    artifacts = load_primary_results(_write_fixture(tmp_path))
    table = build_primary_results_table(artifacts)

    assert "HERMES cohort mean ITE" in set(table["result"])
    assert "FDR-significant Hallmark treatment interactions" in set(
        table["result"]
    )


def test_uncertainty_and_applicability_state_tables(tmp_path) -> None:
    artifacts = load_primary_results(_write_fixture(tmp_path))

    uncertainty = build_uncertainty_state_table(artifacts)
    applicability = build_applicability_state_table(artifacts)

    assert uncertainty["n"].sum() == 241
    assert applicability["n"].sum() == 241


def test_modifier_reporting_preserves_negative_primary_result(tmp_path) -> None:
    artifacts = load_primary_results(_write_fixture(tmp_path))
    table = build_modifier_reporting_table(artifacts)

    assert len(table) == 50
    assert table["nominal_interaction"].sum() == 0
    assert table["fdr_significant_interaction"].sum() == 0


def test_permutation_reporting_adds_null_reference_distribution(tmp_path) -> None:
    artifacts = load_primary_results(_write_fixture(tmp_path))
    table = build_permutation_reporting_table(artifacts)

    assert {"null_mean", "null_sd", "null_q025", "null_q975"}.issubset(
        table.columns
    )
    assert table["null_mean"].notna().all()


def test_complete_report_generation(tmp_path) -> None:
    results = _write_fixture(tmp_path)
    report_dir = tmp_path / "report"

    generated = generate_primary_results_report(
        results_dir=results,
        report_dir=report_dir,
    )

    required = {
        "figure_1_patient_ite_distribution",
        "figure_2_uncertainty_stability",
        "figure_3_permutation_null",
        "figure_4_robustness_scenarios",
        "figure_5_modifier_forest",
        "figure_6_applicability",
        "supplement_repeat_performance",
        "table__primary_results_summary",
        "table__hallmark_modifier_results",
        "table__permutation_results",
        "report_summary",
        "report_manifest",
    }

    assert required.issubset(generated)

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0

    manifest = json.loads(
        generated["report_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["read_only_reporting"] is True
    assert manifest["figures_generated"] == 7
    assert manifest["tables_generated"] == 8