"""
HERMES 2.0
Primary NeoTRIP Results Report
==============================

Purpose
-------
Convert the already-locked HERMES 2.0 primary NeoTRIP analysis artifacts into
a deterministic, conference/paper-oriented reporting package.

This module is READ-ONLY with respect to scientific estimation:
    * it does not refit HERMES;
    * it does not alter the locked analysis plan;
    * it does not reselect thresholds;
    * it does not perform post-hoc biomarker optimization.

It reads the frozen primary-analysis artifacts and produces figures, compact
results tables, and a machine-readable reporting manifest.

Default input
-------------
outputs/hermes2/primary_neotrip

Default output
--------------
outputs/hermes2/primary_neotrip/report

Interpretation
--------------
The report distinguishes:
    1. cohort-level estimated benefit;
    2. patient-level heterogeneity and resampling uncertainty;
    3. modeling robustness;
    4. feature-permutation null behavior;
    5. pathway-level treatment interaction evidence;
    6. biological applicability / OOD status.

A ranked pathway is NOT a validated predictive biomarker unless supported by
the prespecified inferential threshold and subsequent validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path("outputs/hermes2/primary_neotrip")
DEFAULT_REPORT_DIR = DEFAULT_RESULTS_DIR / "report"


@dataclass
class PrimaryResultsArtifacts:
    """Loaded locked HERMES primary-analysis artifacts."""

    root: Path
    manifest: dict[str, Any]
    science_summary: dict[str, Any]
    analysis_plan: dict[str, Any]
    patient_results: pd.DataFrame
    uncertainty: pd.DataFrame
    robustness_patients: pd.DataFrame
    robustness_scenarios: pd.DataFrame
    robustness_pairwise: pd.DataFrame
    applicability: pd.DataFrame
    modifiers: pd.DataFrame
    modifier_robustness: pd.DataFrame
    permutation_null: pd.DataFrame
    permutation_observed_vs_null: pd.DataFrame
    repeat_summary: pd.DataFrame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required HERMES artifact not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required HERMES artifact not found: {path}")
    return pd.read_csv(path)


def _drop_export_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop pandas-export index columns without changing scientific columns."""
    frame = frame.copy()
    unnamed = [
        column
        for column in frame.columns
        if str(column).startswith("Unnamed:")
    ]
    if unnamed:
        frame = frame.drop(columns=unnamed)
    return frame


def load_primary_results(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> PrimaryResultsArtifacts:
    """Load and validate the frozen primary-analysis artifact package."""

    root = Path(results_dir)
    engine = root / "engine_outputs"

    artifacts = PrimaryResultsArtifacts(
        root=root,
        manifest=_read_json(root / "analysis_manifest.json"),
        science_summary=_read_json(root / "primary_science_summary.json"),
        analysis_plan=_read_json(root / "primary_analysis_plan.json"),
        patient_results=_drop_export_index(
            _read_csv(engine / "patient_level_results.csv")
        ),
        uncertainty=_drop_export_index(
            _read_csv(engine / "uncertainty.csv")
        ),
        robustness_patients=_drop_export_index(
            _read_csv(engine / "patient_robustness.csv")
        ),
        robustness_scenarios=_drop_export_index(
            _read_csv(engine / "robustness_scenarios.csv")
        ),
        robustness_pairwise=_drop_export_index(
            _read_csv(engine / "robustness_pairwise.csv")
        ),
        applicability=_drop_export_index(
            _read_csv(engine / "applicability.csv")
        ),
        modifiers=_drop_export_index(
            _read_csv(engine / "modifier_discovery.csv")
        ),
        modifier_robustness=_drop_export_index(
            _read_csv(engine / "modifier_robustness.csv")
        ),
        permutation_null=_drop_export_index(
            _read_csv(engine / "permutation_null_statistics.csv")
        ),
        permutation_observed_vs_null=_read_csv(
            engine / "permutation_observed_vs_null.csv"
        ),
        repeat_summary=_drop_export_index(
            _read_csv(engine / "repeat_summary.csv")
        ),
    )

    validate_primary_results(artifacts)
    return artifacts


def validate_primary_results(
    artifacts: PrimaryResultsArtifacts,
) -> None:
    """Validate reporting inputs without recalculating scientific estimates."""

    manifest = artifacts.manifest
    summary = artifacts.science_summary

    expected_patients = int(manifest["patients"])
    expected_features = int(manifest["biological_features"])

    if expected_patients != 241:
        raise ValueError(
            f"Locked primary package expected 241 patients; "
            f"manifest reports {expected_patients}."
        )

    if expected_features != 50:
        raise ValueError(
            f"Locked primary package expected 50 Hallmark features; "
            f"manifest reports {expected_features}."
        )

    if not bool(manifest["all_locked_audit_checks_passed"]):
        raise ValueError("Locked cohort audit did not pass.")

    if len(artifacts.patient_results) != expected_patients:
        raise ValueError("Patient-level results do not preserve patient count.")

    if len(artifacts.uncertainty) != expected_patients:
        raise ValueError("Uncertainty table does not preserve patient count.")

    if len(artifacts.robustness_patients) != expected_patients:
        raise ValueError("Robustness table does not preserve patient count.")

    if len(artifacts.applicability) != expected_patients:
        raise ValueError("Applicability table does not preserve patient count.")

    if len(artifacts.modifiers) != expected_features:
        raise ValueError("Modifier table does not contain all 50 Hallmarks.")

    if int(summary["patients"]) != expected_patients:
        raise ValueError("Science summary and manifest disagree on patients.")

    patient_id_sets = []
    for table in (
        artifacts.patient_results,
        artifacts.uncertainty,
        artifacts.robustness_patients,
        artifacts.applicability,
    ):
        if "Patient_ID" not in table.columns:
            raise ValueError("A required patient table lacks Patient_ID.")
        if table["Patient_ID"].duplicated().any():
            raise ValueError("Duplicate patient IDs detected in report inputs.")
        patient_id_sets.append(tuple(table["Patient_ID"].astype(str)))

    if not all(ids == patient_id_sets[0] for ids in patient_id_sets[1:]):
        raise ValueError("Patient ordering differs across primary result tables.")

    numeric_tables = {
        "patient_results": artifacts.patient_results,
        "uncertainty": artifacts.uncertainty,
        "robustness_patients": artifacts.robustness_patients,
        "robustness_scenarios": artifacts.robustness_scenarios,
        "applicability": artifacts.applicability,
        "modifiers": artifacts.modifiers,
        "permutation_null": artifacts.permutation_null,
        "repeat_summary": artifacts.repeat_summary,
    }

    for name, table in numeric_tables.items():
        numeric = table.select_dtypes(include=[np.number])
        if not numeric.empty and not np.isfinite(
            numeric.to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Non-finite numeric values detected in {name}.")

    if int(summary["nominal_modifier_count"]) != int(
        artifacts.modifiers["nominal_interaction"].sum()
    ):
        raise ValueError("Nominal modifier count is inconsistent.")

    if int(summary["fdr_modifier_count"]) != int(
        artifacts.modifiers["fdr_significant_interaction"].sum()
    ):
        raise ValueError("FDR modifier count is inconsistent.")


def build_primary_results_table(
    artifacts: PrimaryResultsArtifacts,
) -> pd.DataFrame:
    """Build a compact manuscript/conference summary table."""

    s = artifacts.science_summary

    rows = [
        ("Patients", int(s["patients"]), ""),
        ("Hallmark pathways", int(s["biological_features"]), ""),
        ("Observed pCR rate, CT", float(s["observed_pcr_rate_CT"]), "proportion"),
        (
            "Observed pCR rate, CT/A",
            float(s["observed_pcr_rate_CT_A"]),
            "proportion",
        ),
        (
            "Observed absolute pCR difference, CT/A - CT",
            float(s["observed_absolute_pcr_difference_CT_A_minus_CT"]),
            "probability difference",
        ),
        (
            "HERMES cohort mean ITE",
            float(s["hermes_cohort_mean_ite"]),
            "probability difference",
        ),
        (
            "HERMES cohort median ITE",
            float(s["hermes_cohort_median_ite"]),
            "probability difference",
        ),
        (
            "Mean patient ITE SD",
            float(s["mean_patient_ite_sd"]),
            "repeated cross-fitting",
        ),
        (
            "Mean pairwise ITE Spearman",
            float(s["mean_pairwise_ite_spearman"]),
            "robustness scenarios",
        ),
        (
            "Robust patients",
            float(s["fraction_robust_patients"]),
            "proportion",
        ),
        (
            "In-distribution patients",
            float(s["fraction_in_distribution"]),
            "proportion",
        ),
        (
            "Out-of-distribution patients",
            float(s["fraction_out_of_distribution"]),
            "proportion",
        ),
        (
            "Nominal Hallmark treatment interactions",
            int(s["nominal_modifier_count"]),
            "p < 0.05",
        ),
        (
            "FDR-significant Hallmark treatment interactions",
            int(s["fdr_modifier_count"]),
            "FDR threshold from locked plan",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["result", "value", "interpretation"],
    )


def build_uncertainty_state_table(
    artifacts: PrimaryResultsArtifacts,
) -> pd.DataFrame:
    counts = (
        artifacts.uncertainty["evidence_state"]
        .value_counts(dropna=False)
        .rename_axis("evidence_state")
        .reset_index(name="n")
    )
    counts["fraction"] = counts["n"] / counts["n"].sum()
    return counts.sort_values(
        ["n", "evidence_state"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_applicability_state_table(
    artifacts: PrimaryResultsArtifacts,
) -> pd.DataFrame:
    counts = (
        artifacts.applicability["applicability_state"]
        .value_counts(dropna=False)
        .rename_axis("applicability_state")
        .reset_index(name="n")
    )
    counts["fraction"] = counts["n"] / counts["n"].sum()
    return counts.sort_values(
        ["n", "applicability_state"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def build_modifier_reporting_table(
    artifacts: PrimaryResultsArtifacts,
) -> pd.DataFrame:
    """Return all 50 Hallmarks ranked by prespecified interaction evidence."""

    columns = [
        "feature",
        "interaction_coefficient",
        "interaction_standard_error",
        "interaction_odds_ratio",
        "interaction_or_ci_lower",
        "interaction_or_ci_upper",
        "interaction_p_value",
        "interaction_fdr",
        "risk_difference_contrast",
        "interaction_direction",
        "nominal_interaction",
        "fdr_significant_interaction",
        "interaction_rank",
    ]

    available = [
        column for column in columns if column in artifacts.modifiers.columns
    ]

    table = artifacts.modifiers[available].copy()
    return table.sort_values(
        ["interaction_fdr", "interaction_p_value", "interaction_rank"],
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)


def build_permutation_reporting_table(
    artifacts: PrimaryResultsArtifacts,
) -> pd.DataFrame:
    table = artifacts.permutation_observed_vs_null.copy()

    if "statistic" not in table.columns:
        unnamed = [
            column
            for column in table.columns
            if str(column).startswith("Unnamed:")
        ]
        if unnamed:
            table = table.rename(columns={unnamed[0]: "statistic"})
        else:
            first_column = table.columns[0]
            table = table.rename(columns={first_column: "statistic"})

    null = artifacts.permutation_null
    rows: list[dict[str, Any]] = []

    for _, row in table.iterrows():
        statistic = str(row["statistic"])
        record: dict[str, Any] = {
            "statistic": statistic,
            "observed": row["observed"],
            "empirical_p_value": row["empirical_p_value"],
        }

        if statistic in null.columns:
            series = null[statistic].astype(float)
            record["null_mean"] = float(series.mean())
            record["null_sd"] = float(series.std(ddof=1))
            record["null_q025"] = float(series.quantile(0.025))
            record["null_q975"] = float(series.quantile(0.975))
        else:
            record["null_mean"] = np.nan
            record["null_sd"] = np.nan
            record["null_q025"] = np.nan
            record["null_q975"] = np.nan

        rows.append(record)

    return pd.DataFrame(rows)


def _save_figure(
    fig: plt.Figure,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_patient_ite_distribution(
    artifacts: PrimaryResultsArtifacts,
    path: Path,
) -> None:
    """Figure 1: ranked patient ITEs with repeated-cross-fit intervals."""

    table = artifacts.uncertainty.sort_values(
        "mean_ite",
        ascending=True,
        kind="mergesort",
    ).reset_index(drop=True)

    x = np.arange(len(table))
    mean = table["mean_ite"].to_numpy(dtype=float)
    lower = table["ite_lower"].to_numpy(dtype=float)
    upper = table["ite_upper"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.vlines(x, lower, upper, linewidth=0.6, alpha=0.35)
    ax.scatter(x, mean, s=12)
    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Patients ranked by mean estimated treatment effect")
    ax.set_ylabel("Estimated incremental pCR probability (CT/A - CT)")
    ax.set_title(
        "HERMES patient-level treatment-effect estimates\n"
        "100 repeated cross-fits; intervals summarize resampling variability"
    )
    ax.text(
        0.01,
        0.02,
        "Intervals are repeated-cross-fit uncertainty summaries, "
        "not formal causal confidence intervals.",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
    )
    _save_figure(fig, path)


def plot_uncertainty_stability(
    artifacts: PrimaryResultsArtifacts,
    path: Path,
) -> None:
    """Figure 2: treatment-effect magnitude versus resampling uncertainty."""

    table = artifacts.uncertainty

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    for state, subset in table.groupby("evidence_state", sort=True):
        ax.scatter(
            subset["mean_ite"],
            subset["ite_std"],
            s=25,
            alpha=0.75,
            label=str(state),
        )

    ax.axvline(0.0, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Mean estimated treatment effect")
    ax.set_ylabel("ITE SD across repeated cross-fitting")
    ax.set_title("Magnitude and stability of patient-level HERMES estimates")
    ax.legend(title="Evidence state", frameon=False)
    _save_figure(fig, path)


def plot_permutation_null(
    artifacts: PrimaryResultsArtifacts,
    path: Path,
) -> None:
    """Figure 3: prespecified observed heterogeneity statistics vs null."""

    reporting = build_permutation_reporting_table(artifacts)

    reporting = reporting[
        reporting["empirical_p_value"].notna()
        & reporting["null_mean"].notna()
    ].copy()

    reporting["short_name"] = (
        reporting["statistic"]
        .str.replace("fraction_sign_stability_ge_90pct", "sign stability ≥90%", regex=False)
        .str.replace("fraction_unanimous_sign", "unanimous sign", regex=False)
        .str.replace("median_stability_signal_ratio", "median signal/uncertainty", regex=False)
        .str.replace("ite_absolute_90th_percentile", "|ITE| 90th pct", regex=False)
        .str.replace("ite_mean_absolute_deviation", "ITE mean abs deviation", regex=False)
        .str.replace("ite_sd_across_patients", "ITE SD", regex=False)
        .str.replace("ite_max_absolute", "max |ITE|", regex=False)
        .str.replace("ite_iqr", "ITE IQR", regex=False)
    )

    y = np.arange(len(reporting))

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.hlines(
        y,
        reporting["null_q025"],
        reporting["null_q975"],
        linewidth=3,
        alpha=0.45,
        label="Permutation null 95% interval",
    )
    ax.scatter(
        reporting["null_mean"],
        y,
        marker="|",
        s=110,
        label="Permutation null mean",
    )
    ax.scatter(
        reporting["observed"],
        y,
        s=35,
        label="Observed HERMES statistic",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(reporting["short_name"])
    ax.set_xlabel("Statistic value")
    ax.set_title(
        "Observed treatment-effect heterogeneity versus feature-permutation null"
    )
    ax.legend(frameon=False)

    for i, p_value in enumerate(reporting["empirical_p_value"]):
        ax.text(
            ax.get_xlim()[1],
            i,
            f"  p={p_value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    _save_figure(fig, path)


def plot_robustness_scenarios(
    artifacts: PrimaryResultsArtifacts,
    path: Path,
) -> None:
    """Figure 4: cohort mean ITE under the prespecified robustness grid."""

    table = artifacts.robustness_scenarios.copy()

    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for c_value, subset in table.groupby("regularization_C", sort=True):
        subset = subset.sort_values("n_splits")
        ax.plot(
            subset["n_splits"],
            subset["cohort_mean_ite"],
            marker="o",
            label=f"C={c_value:g}",
        )

    ax.axhline(
        float(artifacts.science_summary["hermes_cohort_mean_ite"]),
        linestyle="--",
        linewidth=1.0,
        label="Locked primary estimate",
    )
    ax.set_xlabel("Cross-fitting folds")
    ax.set_ylabel("Cohort mean estimated treatment effect")
    ax.set_title("Sensitivity of cohort-level HERMES estimates")
    ax.set_xticks(sorted(table["n_splits"].unique()))
    ax.legend(frameon=False)
    _save_figure(fig, path)


def plot_modifier_forest(
    artifacts: PrimaryResultsArtifacts,
    path: Path,
    *,
    top_n: int = 15,
) -> None:
    """Figure 5: top-ranked pathway treatment interactions with 95% CIs."""

    table = build_modifier_reporting_table(artifacts).head(top_n).copy()
    table = table.sort_values(
        "interaction_odds_ratio",
        ascending=True,
    )

    labels = (
        table["feature"]
        .str.replace("HALLMARK_", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    y = np.arange(len(table))
    odds_ratio = table["interaction_odds_ratio"].to_numpy(dtype=float)
    lower = table["interaction_or_ci_lower"].to_numpy(dtype=float)
    upper = table["interaction_or_ci_upper"].to_numpy(dtype=float)

    xerr = np.vstack(
        [
            odds_ratio - lower,
            upper - odds_ratio,
        ]
    )

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    ax.errorbar(
        odds_ratio,
        y,
        xerr=xerr,
        fmt="o",
        capsize=2,
    )
    ax.axvline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Treatment × pathway interaction odds ratio (95% CI)")
    ax.set_title(
        "Top-ranked Hallmark treatment interactions\n"
        "No pathway met nominal or FDR significance in the locked analysis"
    )
    _save_figure(fig, path)


def plot_applicability(
    artifacts: PrimaryResultsArtifacts,
    path: Path,
) -> None:
    """Figure 6: applicability distribution for patient-level estimates."""

    table = artifacts.applicability.copy()

    fig, ax = plt.subplots(figsize=(7.2, 5.3))

    for state, subset in table.groupby("applicability_state", sort=True):
        ax.scatter(
            subset["mahalanobis_reference_percentile"],
            subset["max_abs_z_reference_percentile"],
            s=25,
            alpha=0.75,
            label=str(state),
        )

    ax.set_xlabel("Mahalanobis reference percentile")
    ax.set_ylabel("Maximum |z| reference percentile")
    ax.set_title("HERMES biological applicability / OOD characterization")
    ax.legend(title="Applicability", frameon=False)
    _save_figure(fig, path)


def plot_repeat_performance(
    artifacts: PrimaryResultsArtifacts,
    path: Path,
) -> None:
    """Supplement: out-of-fold discrimination/calibration across repeats."""

    table = artifacts.repeat_summary.sort_values("repeat")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(
        table["repeat"],
        table["oof_auc"],
        linewidth=1.0,
        label="OOF AUC",
    )
    ax.axhline(
        table["oof_auc"].mean(),
        linestyle="--",
        linewidth=1.0,
        label=f"Mean AUC={table['oof_auc'].mean():.3f}",
    )
    ax.set_xlabel("Repeated cross-fit iteration")
    ax.set_ylabel("Out-of-fold AUC")
    ax.set_title("Predictive discrimination across repeated cross-fitting")
    ax.legend(frameon=False)
    _save_figure(fig, path)


def generate_primary_results_report(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    report_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate the complete read-only reporting package."""

    artifacts = load_primary_results(results_dir)

    if report_dir is None:
        report_dir = Path(results_dir) / "report"
    else:
        report_dir = Path(report_dir)

    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    summary_table = build_primary_results_table(artifacts)
    uncertainty_states = build_uncertainty_state_table(artifacts)
    applicability_states = build_applicability_state_table(artifacts)
    modifier_table = build_modifier_reporting_table(artifacts)
    permutation_table = build_permutation_reporting_table(artifacts)

    tables = {
        "primary_results_summary": summary_table,
        "uncertainty_states": uncertainty_states,
        "applicability_states": applicability_states,
        "hallmark_modifier_results": modifier_table,
        "permutation_results": permutation_table,
        "robustness_scenarios": artifacts.robustness_scenarios,
        "robustness_pairwise": artifacts.robustness_pairwise,
        "repeat_summary": artifacts.repeat_summary,
    }

    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        generated[f"table__{name}"] = path

    figure_functions = {
        "figure_1_patient_ite_distribution": plot_patient_ite_distribution,
        "figure_2_uncertainty_stability": plot_uncertainty_stability,
        "figure_3_permutation_null": plot_permutation_null,
        "figure_4_robustness_scenarios": plot_robustness_scenarios,
        "figure_5_modifier_forest": plot_modifier_forest,
        "figure_6_applicability": plot_applicability,
        "supplement_repeat_performance": plot_repeat_performance,
    }

    for name, function in figure_functions.items():
        path = figures_dir / f"{name}.png"
        function(artifacts, path)
        generated[name] = path

    state_counts = {
        str(row["evidence_state"]): int(row["n"])
        for _, row in uncertainty_states.iterrows()
    }

    applicability_counts = {
        str(row["applicability_state"]): int(row["n"])
        for _, row in applicability_states.iterrows()
    }

    report_summary = {
        "plan_name": artifacts.manifest["plan_name"],
        "plan_sha256": artifacts.manifest["plan_sha256"],
        "engine_tag": artifacts.manifest["engine_tag"],
        "analysis_scope": artifacts.manifest["analysis_scope"],
        "patients": int(artifacts.manifest["patients"]),
        "biological_features": int(artifacts.manifest["biological_features"]),
        "cohort_mean_ite": float(
            artifacts.science_summary["hermes_cohort_mean_ite"]
        ),
        "observed_absolute_pcr_difference": float(
            artifacts.science_summary[
                "observed_absolute_pcr_difference_CT_A_minus_CT"
            ]
        ),
        "nominal_modifier_count": int(
            artifacts.science_summary["nominal_modifier_count"]
        ),
        "fdr_modifier_count": int(
            artifacts.science_summary["fdr_modifier_count"]
        ),
        "fraction_robust_patients": float(
            artifacts.science_summary["fraction_robust_patients"]
        ),
        "mean_pairwise_ite_spearman": float(
            artifacts.science_summary["mean_pairwise_ite_spearman"]
        ),
        "fraction_in_distribution": float(
            artifacts.science_summary["fraction_in_distribution"]
        ),
        "fraction_out_of_distribution": float(
            artifacts.science_summary["fraction_out_of_distribution"]
        ),
        "uncertainty_state_counts": state_counts,
        "applicability_state_counts": applicability_counts,
        "reporting_interpretation": (
            "Locked primary analysis found a modest positive cohort-level "
            "estimated treatment effect but no nominal or FDR-significant "
            "Hallmark treatment modifiers. Ranked pathways are exploratory "
            "descriptive candidates only."
        ),
        "external_validation_required": True,
    }

    summary_path = report_dir / "report_summary.json"
    summary_path.write_text(
        json.dumps(
            report_summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    generated["report_summary"] = summary_path

    manifest_path = report_dir / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_plan_sha256": artifacts.manifest["plan_sha256"],
                "source_engine_tag": artifacts.manifest["engine_tag"],
                "read_only_reporting": True,
                "figures_generated": int(len(figure_functions)),
                "tables_generated": int(len(tables)),
                "generated_files": {
                    key: str(path)
                    for key, path in sorted(generated.items())
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    generated["report_manifest"] = manifest_path

    return generated


def summarize_primary_results_report(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> None:
    """Print the key locked results without altering the analysis."""

    artifacts = load_primary_results(results_dir)
    s = artifacts.science_summary

    print("=== HERMES 2.0 PRIMARY RESULTS REPORT ===")
    print()
    print(f"Plan: {artifacts.manifest['plan_name']}")
    print(f"Plan SHA256: {artifacts.manifest['plan_sha256']}")
    print(f"Engine: {artifacts.manifest['engine_tag']}")
    print()

    print(f"Patients: {int(s['patients'])}")
    print(f"Hallmark pathways: {int(s['biological_features'])}")
    print(
        "Observed absolute pCR difference (CT/A - CT): "
        f"{float(s['observed_absolute_pcr_difference_CT_A_minus_CT']):.4f}"
    )
    print(
        "HERMES mean ITE: "
        f"{float(s['hermes_cohort_mean_ite']):.4f}"
    )
    print(
        "Mean patient ITE SD: "
        f"{float(s['mean_patient_ite_sd']):.4f}"
    )
    print()

    print(
        "Treatment modifiers: "
        f"nominal={int(s['nominal_modifier_count'])}, "
        f"FDR={int(s['fdr_modifier_count'])}"
    )
    print(
        "Robust patients: "
        f"{float(s['fraction_robust_patients']):.3f}"
    )
    print(
        "Mean pairwise ITE Spearman: "
        f"{float(s['mean_pairwise_ite_spearman']):.3f}"
    )
    print(
        "In distribution: "
        f"{float(s['fraction_in_distribution']):.3f}"
    )
    print(
        "Out of distribution: "
        f"{float(s['fraction_out_of_distribution']):.3f}"
    )
    print()
    print(
        "Interpretation: the locked primary analysis supports a modest "
        "cohort-level estimated benefit but does not support a Hallmark-level "
        "predictive treatment modifier."
    )
    print(
        "Ranked modifiers remain exploratory; external/orthogonal validation "
        "is required."
    )


def main() -> None:
    generated = generate_primary_results_report()
    summarize_primary_results_report()

    print()
    print(f"Report written to: {DEFAULT_REPORT_DIR}")
    print(f"Generated files: {len(generated)}")


if __name__ == "__main__":
    main()