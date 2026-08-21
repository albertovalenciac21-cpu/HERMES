"""
HERMES 2.0
Locked NeoTRIP Biological Characterization
===========================================

Purpose
-------
Characterize the biology associated with the already-locked HERMES patient-
level treatment-effect estimates without refitting, retuning, or changing the
primary HERMES model.

This is an EXPLORATORY biological characterization layer. It asks:

    * Which Hallmark pathway states vary across the HERMES ITE spectrum?
    * Are those pathway/ITE relationships stable across repeated cross-fits?
    * How are estimated treatment effects distributed across TNBC subtypes?
    * Which biological programs distinguish the highest- versus lowest-ITE
      patients?

Important scientific boundary
-----------------------------
These analyses do NOT convert ranked pathways into validated predictive
biomarkers. The locked primary analysis found no nominal or FDR-significant
treatment x Hallmark interactions. Therefore pathway/ITE associations here
are descriptive characterization of the model-implied treatment-effect
landscape and require orthogonal/external validation.

No model is selected or tuned from these results.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend.app.treatment_effects.feature_builder import (
    TreatmentEffectDataset,
    build_treatment_effect_dataset,
)


DEFAULT_PRIMARY_RESULTS_DIR = Path("outputs/hermes2/primary_neotrip")
DEFAULT_BIOLOGY_DIR = Path("outputs/hermes2/biological_characterization")


@dataclass(frozen=True)
class BiologicalCharacterizationPlan:
    """Locked exploratory characterization specification."""

    plan_name: str = "hermes2_neotrip_biological_characterization_v1"
    source_primary_plan: str = "hermes2_neotrip_primary_locked_v1"
    source_engine_tag: str = "hermes-2.0-engine-v1.0"

    high_ite_quantile: float = 0.75
    low_ite_quantile: float = 0.25
    pathway_reporting_top_n: int = 15

    analysis_scope: str = "exploratory_post_primary_characterization"
    predictive_biomarker_claims_allowed: bool = False
    external_validation_required: bool = True


LOCKED_BIOLOGY_PLAN = BiologicalCharacterizationPlan()


@dataclass
class BiologicalCharacterizationResult:
    """Complete HERMES biological characterization package."""

    plan: BiologicalCharacterizationPlan
    source_manifest: dict[str, Any]
    dataset: TreatmentEffectDataset
    patient_table: pd.DataFrame
    pathway_associations: pd.DataFrame
    subtype_summary: pd.DataFrame
    subtype_by_ite_group: pd.DataFrame
    ite_group_summary: pd.DataFrame
    top_positive_pathways: pd.DataFrame
    top_negative_pathways: pd.DataFrame


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Deterministic Benjamini-Hochberg FDR correction."""

    p = p_values.astype(float).to_numpy()
    n = len(p)

    if n == 0:
        return pd.Series(dtype=float, index=p_values.index)

    order = np.argsort(p)
    ranked = p[order]

    adjusted = ranked * n / np.arange(1, n + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out = np.empty(n, dtype=float)
    out[order] = adjusted

    return pd.Series(
        out,
        index=p_values.index,
        name="spearman_fdr",
    )


def _cohens_d(high: pd.Series, low: pd.Series) -> float:
    """Standardized mean difference: high-ITE group minus low-ITE group."""

    high = high.astype(float)
    low = low.astype(float)

    n1 = len(high)
    n0 = len(low)

    if n1 < 2 or n0 < 2:
        return float("nan")

    var1 = float(high.var(ddof=1))
    var0 = float(low.var(ddof=1))

    pooled_denominator = n1 + n0 - 2
    if pooled_denominator <= 0:
        return float("nan")

    pooled_var = (
        (n1 - 1) * var1
        + (n0 - 1) * var0
    ) / pooled_denominator

    if pooled_var <= 0 or not np.isfinite(pooled_var):
        return 0.0

    return float(
        (high.mean() - low.mean()) / np.sqrt(pooled_var)
    )


def _load_primary_outputs(
    results_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(
        (results_dir / "analysis_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    ite = pd.read_csv(
        results_dir / "engine_outputs" / "ite_by_repeat.csv"
    )
    uncertainty = pd.read_csv(
        results_dir / "engine_outputs" / "uncertainty.csv"
    )

    if "Patient_ID" not in ite.columns:
        raise ValueError("Locked ITE-by-repeat artifact lacks Patient_ID.")
    if "Patient_ID" not in uncertainty.columns:
        raise ValueError("Locked uncertainty artifact lacks Patient_ID.")

    ite = ite.set_index("Patient_ID")
    uncertainty = uncertainty.set_index("Patient_ID")

    return manifest, ite, uncertainty


def build_characterization_patient_table(
    dataset: TreatmentEffectDataset,
    ite_by_repeat: pd.DataFrame,
    uncertainty: pd.DataFrame,
    *,
    plan: BiologicalCharacterizationPlan = LOCKED_BIOLOGY_PLAN,
) -> pd.DataFrame:
    """Build the locked patient-level biological-characterization table."""

    if not dataset.X.index.astype(str).equals(
        pd.Index(ite_by_repeat.index.astype(str))
    ):
        raise ValueError(
            "Canonical NeoTRIP patients and locked HERMES ITE rows differ."
        )

    mean_ite = ite_by_repeat.mean(axis=1)
    median_ite = ite_by_repeat.median(axis=1)
    repeat_sd = ite_by_repeat.std(axis=1, ddof=1)

    low_cut = float(mean_ite.quantile(plan.low_ite_quantile))
    high_cut = float(mean_ite.quantile(plan.high_ite_quantile))

    group = pd.Series(
        "middle_ite",
        index=mean_ite.index,
        dtype="object",
    )
    group.loc[mean_ite <= low_cut] = "low_ite"
    group.loc[mean_ite >= high_cut] = "high_ite"

    table = pd.DataFrame(
        {
            "Patient_ID": dataset.X.index.astype(str),
            "treatment": dataset.T.to_numpy(dtype=int),
            "outcome": dataset.Y.to_numpy(dtype=int),
            "tnbc_type": (
                dataset.metadata["tnbc_type"].astype(str).to_numpy()
            ),
            "mean_ite": mean_ite.to_numpy(dtype=float),
            "median_ite": median_ite.to_numpy(dtype=float),
            "ite_repeat_sd": repeat_sd.to_numpy(dtype=float),
            "ite_group": group.to_numpy(),
        }
    )

    if "evidence_state" in uncertainty.columns:
        table["evidence_state"] = (
            uncertainty.loc[
                dataset.X.index.astype(str),
                "evidence_state",
            ].astype(str).to_numpy()
        )

    return table


def compute_pathway_associations(
    X: pd.DataFrame,
    ite_by_repeat: pd.DataFrame,
    patient_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Characterize each Hallmark pathway across the locked ITE landscape.

    Reports:
        * Spearman pathway vs mean ITE;
        * BH-FDR across 50 pathways;
        * high- vs low-ITE standardized mean difference;
        * repeated-cross-fit correlation mean/SD/sign stability.

    These statistics are exploratory characterization, not a replacement for
    the locked treatment x pathway interaction tests.
    """

    mean_ite = pd.Series(
        patient_table["mean_ite"].to_numpy(dtype=float),
        index=patient_table["Patient_ID"].astype(str),
    )

    groups = pd.Series(
        patient_table["ite_group"].to_numpy(),
        index=patient_table["Patient_ID"].astype(str),
    )

    X = X.copy()
    X.index = X.index.astype(str)

    if not X.index.equals(mean_ite.index):
        raise ValueError("Pathway matrix and patient characterization differ.")

    rows: list[dict[str, Any]] = []

    for pathway in X.columns:
        values = X[pathway].astype(float)

        rho, p_value = spearmanr(
            values.to_numpy(dtype=float),
            mean_ite.to_numpy(dtype=float),
        )

        repeat_rhos: list[float] = []
        for repeat in ite_by_repeat.columns:
            repeat_values = pd.Series(
                ite_by_repeat[repeat].to_numpy(dtype=float),
                index=ite_by_repeat.index.astype(str),
            )
            repeat_rho, _ = spearmanr(
                values.to_numpy(dtype=float),
                repeat_values.loc[X.index].to_numpy(dtype=float),
            )
            if np.isfinite(repeat_rho):
                repeat_rhos.append(float(repeat_rho))

        repeat_array = np.asarray(repeat_rhos, dtype=float)

        high = values.loc[groups.eq("high_ite")]
        low = values.loc[groups.eq("low_ite")]

        rows.append(
            {
                "pathway": str(pathway),
                "spearman_rho_mean_ite": float(rho),
                "spearman_p_value": float(p_value),
                "high_ite_mean_pathway_score": float(high.mean()),
                "low_ite_mean_pathway_score": float(low.mean()),
                "high_minus_low_pathway_score": float(
                    high.mean() - low.mean()
                ),
                "high_vs_low_cohens_d": _cohens_d(high, low),
                "repeat_spearman_mean": float(
                    repeat_array.mean()
                ),
                "repeat_spearman_sd": float(
                    repeat_array.std(ddof=1)
                    if len(repeat_array) > 1
                    else 0.0
                ),
                "repeat_spearman_sign_stability": float(
                    max(
                        (repeat_array > 0).mean(),
                        (repeat_array < 0).mean(),
                    )
                ),
                "n_high_ite": int(len(high)),
                "n_low_ite": int(len(low)),
            }
        )

    table = pd.DataFrame(rows)
    table["spearman_fdr"] = _benjamini_hochberg(
        table["spearman_p_value"]
    )

    table["absolute_spearman_rho"] = (
        table["spearman_rho_mean_ite"].abs()
    )
    table["absolute_cohens_d"] = (
        table["high_vs_low_cohens_d"].abs()
    )

    table = table.sort_values(
        [
            "absolute_spearman_rho",
            "repeat_spearman_sign_stability",
            "absolute_cohens_d",
        ],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)

    table["characterization_rank"] = np.arange(
        1,
        len(table) + 1,
    )

    return table


def build_subtype_summary(
    patient_table: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize locked HERMES ITEs by TNBC subtype."""

    grouped = (
        patient_table.groupby(
            "tnbc_type",
            observed=False,
        )["mean_ite"]
        .agg(
            patients="count",
            mean_ite="mean",
            median_ite="median",
            ite_sd="std",
            minimum_ite="min",
            maximum_ite="max",
        )
        .reset_index()
    )

    high_fraction = (
        patient_table.assign(
            is_high=patient_table["ite_group"].eq("high_ite")
        )
        .groupby("tnbc_type", observed=False)["is_high"]
        .mean()
        .rename("fraction_high_ite")
        .reset_index()
    )

    low_fraction = (
        patient_table.assign(
            is_low=patient_table["ite_group"].eq("low_ite")
        )
        .groupby("tnbc_type", observed=False)["is_low"]
        .mean()
        .rename("fraction_low_ite")
        .reset_index()
    )

    return (
        grouped
        .merge(high_fraction, on="tnbc_type", how="left")
        .merge(low_fraction, on="tnbc_type", how="left")
        .sort_values("mean_ite", ascending=False)
        .reset_index(drop=True)
    )


def build_subtype_by_ite_group(
    patient_table: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-tabulate TNBC subtype against low/middle/high ITE groups."""

    table = pd.crosstab(
        patient_table["tnbc_type"],
        patient_table["ite_group"],
        dropna=False,
    )

    for column in ("low_ite", "middle_ite", "high_ite"):
        if column not in table.columns:
            table[column] = 0

    table = table[
        ["low_ite", "middle_ite", "high_ite"]
    ].copy()

    table["total"] = table.sum(axis=1)

    for column in ("low_ite", "middle_ite", "high_ite"):
        table[f"fraction_{column}"] = (
            table[column] / table["total"]
        )

    return table.reset_index()


def build_ite_group_summary(
    patient_table: pd.DataFrame,
) -> pd.DataFrame:
    """Describe patient counts and ITE distribution by prespecified group."""

    return (
        patient_table.groupby(
            "ite_group",
            observed=False,
        )["mean_ite"]
        .agg(
            patients="count",
            mean_ite="mean",
            median_ite="median",
            ite_sd="std",
            minimum_ite="min",
            maximum_ite="max",
        )
        .reset_index()
    )


def run_locked_biological_characterization(
    *,
    primary_results_dir: str | Path = DEFAULT_PRIMARY_RESULTS_DIR,
    plan: BiologicalCharacterizationPlan = LOCKED_BIOLOGY_PLAN,
) -> BiologicalCharacterizationResult:
    """Run the read-only biological characterization of locked HERMES ITEs."""

    primary_results_dir = Path(primary_results_dir)

    manifest, ite_by_repeat, uncertainty = _load_primary_outputs(
        primary_results_dir
    )

    if manifest["plan_name"] != plan.source_primary_plan:
        raise ValueError(
            "Biological characterization source plan does not match "
            "the locked primary plan."
        )

    if manifest["engine_tag"] != plan.source_engine_tag:
        raise ValueError(
            "Biological characterization source engine does not match "
            "the locked HERMES engine."
        )

    dataset = build_treatment_effect_dataset(
        min_genes=3,
        min_coverage=0.50,
    )
    dataset.validate()

    patient_table = build_characterization_patient_table(
        dataset,
        ite_by_repeat,
        uncertainty,
        plan=plan,
    )

    pathway_associations = compute_pathway_associations(
        dataset.X,
        ite_by_repeat,
        patient_table,
    )

    subtype_summary = build_subtype_summary(
        patient_table
    )

    subtype_by_ite_group = build_subtype_by_ite_group(
        patient_table
    )

    ite_group_summary = build_ite_group_summary(
        patient_table
    )

    positive = (
        pathway_associations.loc[
            pathway_associations["spearman_rho_mean_ite"] > 0
        ]
        .sort_values(
            "spearman_rho_mean_ite",
            ascending=False,
        )
        .head(plan.pathway_reporting_top_n)
        .copy()
    )

    negative = (
        pathway_associations.loc[
            pathway_associations["spearman_rho_mean_ite"] < 0
        ]
        .sort_values(
            "spearman_rho_mean_ite",
            ascending=True,
        )
        .head(plan.pathway_reporting_top_n)
        .copy()
    )

    return BiologicalCharacterizationResult(
        plan=plan,
        source_manifest=manifest,
        dataset=dataset,
        patient_table=patient_table,
        pathway_associations=pathway_associations,
        subtype_summary=subtype_summary,
        subtype_by_ite_group=subtype_by_ite_group,
        ite_group_summary=ite_group_summary,
        top_positive_pathways=positive,
        top_negative_pathways=negative,
    )


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


def plot_pathway_ite_correlations(
    result: BiologicalCharacterizationResult,
    path: Path,
    *,
    top_n: int = 20,
) -> None:
    """Ranked pathway correlations with locked HERMES patient mean ITE."""

    table = (
        result.pathway_associations
        .sort_values(
            "absolute_spearman_rho",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "spearman_rho_mean_ite",
            ascending=True,
        )
    )

    labels = (
        table["pathway"]
        .str.replace("HALLMARK_", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.barh(
        y,
        table["spearman_rho_mean_ite"],
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Spearman correlation with patient mean HERMES ITE")
    ax.set_title(
        "Hallmark pathways across the locked HERMES treatment-effect spectrum"
    )
    _save_figure(fig, path)


def plot_high_vs_low_pathway_effects(
    result: BiologicalCharacterizationResult,
    path: Path,
    *,
    top_n: int = 20,
) -> None:
    """High- vs low-ITE quartile standardized pathway differences."""

    table = (
        result.pathway_associations
        .sort_values(
            "absolute_cohens_d",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            "high_vs_low_cohens_d",
            ascending=True,
        )
    )

    labels = (
        table["pathway"]
        .str.replace("HALLMARK_", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    y = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    ax.barh(
        y,
        table["high_vs_low_cohens_d"],
    )
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        "Standardized pathway difference: high-ITE minus low-ITE (Cohen d)"
    )
    ax.set_title(
        "Biological programs distinguishing the locked HERMES ITE extremes"
    )
    _save_figure(fig, path)


def plot_subtype_ite_distribution(
    result: BiologicalCharacterizationResult,
    path: Path,
) -> None:
    """Distribution of HERMES ITEs across TNBC subtypes."""

    table = result.patient_table.copy()

    subtype_order = (
        result.subtype_summary["tnbc_type"].astype(str).tolist()
    )

    data = [
        table.loc[
            table["tnbc_type"].eq(subtype),
            "mean_ite",
        ].to_numpy(dtype=float)
        for subtype in subtype_order
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.boxplot(
        data,
        tick_labels=subtype_order,
        showfliers=False,
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("TNBC subtype")
    ax.set_ylabel("Patient mean HERMES ITE")
    ax.set_title(
        "Locked HERMES estimated treatment effects across TNBC subtypes"
    )
    _save_figure(fig, path)


def plot_pathway_repeat_stability(
    result: BiologicalCharacterizationResult,
    path: Path,
    *,
    top_n: int = 20,
) -> None:
    """Relationship between pathway/ITE magnitude and repeat sign stability."""

    table = (
        result.pathway_associations
        .sort_values(
            "absolute_spearman_rho",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.scatter(
        table["absolute_spearman_rho"],
        table["repeat_spearman_sign_stability"],
        s=35,
    )

    for _, row in table.head(10).iterrows():
        label = (
            str(row["pathway"])
            .replace("HALLMARK_", "")
            .replace("_", " ")
        )
        ax.annotate(
            label,
            (
                row["absolute_spearman_rho"],
                row["repeat_spearman_sign_stability"],
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )

    ax.set_xlabel("|Spearman correlation| with mean HERMES ITE")
    ax.set_ylabel("Repeated-cross-fit correlation sign stability")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(
        "Stability of pathway associations across repeated HERMES estimates"
    )
    _save_figure(fig, path)


def export_biological_characterization(
    result: BiologicalCharacterizationResult,
    output_dir: str | Path = DEFAULT_BIOLOGY_DIR,
) -> dict[str, Path]:
    """Export characterization tables, figures, and scientific manifest."""

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    tables = {
        "patient_biological_characterization": result.patient_table,
        "hallmark_ite_associations": result.pathway_associations,
        "tnbc_subtype_summary": result.subtype_summary,
        "tnbc_subtype_by_ite_group": result.subtype_by_ite_group,
        "ite_group_summary": result.ite_group_summary,
        "top_positive_pathways": result.top_positive_pathways,
        "top_negative_pathways": result.top_negative_pathways,
    }

    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        generated[f"table__{name}"] = path

    figure_functions = {
        "figure_1_pathway_ite_correlations": plot_pathway_ite_correlations,
        "figure_2_high_vs_low_pathway_effects": (
            plot_high_vs_low_pathway_effects
        ),
        "figure_3_subtype_ite_distribution": plot_subtype_ite_distribution,
        "figure_4_pathway_repeat_stability": plot_pathway_repeat_stability,
    }

    for name, function in figure_functions.items():
        path = figures_dir / f"{name}.png"
        function(result, path)
        generated[name] = path

    manifest = {
        "plan_name": result.plan.plan_name,
        "source_primary_plan": result.plan.source_primary_plan,
        "source_engine_tag": result.plan.source_engine_tag,
        "source_plan_sha256": result.source_manifest["plan_sha256"],
        "patients": int(result.dataset.n_patients),
        "hallmark_pathways": int(result.dataset.n_features),
        "high_ite_quantile": result.plan.high_ite_quantile,
        "low_ite_quantile": result.plan.low_ite_quantile,
        "analysis_scope": result.plan.analysis_scope,
        "predictive_biomarker_claims_allowed": (
            result.plan.predictive_biomarker_claims_allowed
        ),
        "external_validation_required": (
            result.plan.external_validation_required
        ),
        "scientific_boundary": (
            "Pathway/ITE associations characterize the model-implied "
            "treatment-effect landscape. They do not supersede the locked "
            "primary treatment x pathway interaction analysis, which found "
            "no nominal or FDR-significant Hallmark modifiers."
        ),
        "generated_files": {
            key: str(path)
            for key, path in sorted(generated.items())
        },
    }

    manifest_path = output_dir / "biological_characterization_manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    generated["biological_characterization_manifest"] = manifest_path

    return generated


def summarize_biological_characterization(
    result: BiologicalCharacterizationResult,
) -> None:
    """Print the exploratory biological characterization."""

    print("=== HERMES 2.0 BIOLOGICAL CHARACTERIZATION ===")
    print()
    print(f"Plan: {result.plan.plan_name}")
    print(
        f"Source plan: {result.source_manifest['plan_name']}"
    )
    print(
        f"Source SHA256: {result.source_manifest['plan_sha256']}"
    )
    print(
        f"Patients: {result.dataset.n_patients} | "
        f"Hallmark pathways: {result.dataset.n_features}"
    )
    print()

    print("ITE-group summary:")
    print(result.ite_group_summary.to_string(index=False))
    print()

    print("TNBC subtype summary:")
    print(result.subtype_summary.to_string(index=False))
    print()

    display_columns = [
        "pathway",
        "spearman_rho_mean_ite",
        "spearman_p_value",
        "spearman_fdr",
        "high_vs_low_cohens_d",
        "repeat_spearman_mean",
        "repeat_spearman_sd",
        "repeat_spearman_sign_stability",
    ]

    print("Top positive pathway/ITE associations:")
    print(
        result.top_positive_pathways[
            display_columns
        ].head(10).to_string(index=False)
    )
    print()

    print("Top negative pathway/ITE associations:")
    print(
        result.top_negative_pathways[
            display_columns
        ].head(10).to_string(index=False)
    )
    print()

    print("IMPORTANT:")
    print(
        "These pathway associations are exploratory characterization of "
        "locked HERMES estimates."
    )
    print(
        "They are not validated predictive biomarkers and do not replace "
        "the primary treatment x pathway interaction tests."
    )
    print(
        "Orthogonal/external biological validation is required before "
        "mechanistic or clinical claims."
    )


def main() -> None:
    result = run_locked_biological_characterization()
    summarize_biological_characterization(result)

    generated = export_biological_characterization(result)

    print()
    print(
        f"Biological characterization written to: {DEFAULT_BIOLOGY_DIR}"
    )
    print(f"Generated files: {len(generated)}")


if __name__ == "__main__":
    main()