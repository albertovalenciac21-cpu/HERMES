"""
HERMES 2.0
Locked NeoTRIP Scientific Cohort Audit
======================================

Purpose
-------
Create a deterministic, publication-oriented audit of the exact NeoTRIP
baseline cohort and biological representation used by HERMES 2.0.

This module is intentionally descriptive. It does not inspect treatment-effect
estimates or select favorable biomarkers. The goal is to lock the analysis
population, treatment/outcome encoding, transcriptomic QC, Hallmark coverage,
and contextual variables before downstream HERMES scientific interpretation.

NeoTRIP analysis contrast
-------------------------
Treatment:
    0 = CT
    1 = CT/A (atezolizumab + chemotherapy)

Outcome:
    0 = residual disease (RD)
    1 = pathologic complete response (pCR)

Primary HERMES 2.0 scientific question:
    Does baseline tumor biology modify the incremental probability of pCR
    associated with adding atezolizumab to chemotherapy?

Important
---------
This audit establishes cohort integrity and a prespecified computational
analysis contract. It does not establish causal transportability, external
validation, predictive-biomarker validity, or clinical utility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.treatment_effects.cohort_loader import (
    DEFAULT_CLINICAL_PATH,
    DEFAULT_EXPRESSION_PATH,
    load_neotrip_baseline,
)
from backend.app.treatment_effects.feature_builder import (
    DEFAULT_HALLMARK_GMT,
)
from backend.app.treatment_effects.preprocessing import (
    preprocess_neotrip_baseline,
)
from backend.app.treatment_effects.representations import (
    load_gmt,
    score_gene_sets,
)


@dataclass(frozen=True)
class LockedNeoTRIPAnalysisSpec:
    """Prespecified HERMES 2.0 scientific-analysis contract."""

    study: str = "NeoTRIP"
    study_accession: str = "GSE319641"
    disease: str = "triple-negative breast cancer"
    disease_setting: str = "early"
    sample_timepoint: str = "pretreatment_baseline"

    treatment_control: str = "CT"
    treatment_experimental: str = "CT/A"
    treatment_experimental_description: str = (
        "atezolizumab + carboplatin + nab-paclitaxel"
    )
    treatment_control_description: str = (
        "carboplatin + nab-paclitaxel"
    )

    endpoint: str = "pathologic_complete_response"
    endpoint_positive_label: str = "pCR"
    endpoint_negative_label: str = "RD"

    estimand: str = (
        "individualized incremental probability of pCR associated with "
        "adding atezolizumab to chemotherapy"
    )

    biological_representation: str = "MSigDB Hallmark pathways"
    hallmark_collection: str = "h.all.v2026.1.Hs.symbols.gmt"
    minimum_genes_per_pathway: int = 3
    minimum_pathway_gene_coverage: float = 0.50

    transcriptomic_qc_variance_threshold: float = 0.01
    transcriptomic_qc_expression_prevalence_threshold: float = 0.10
    transcriptomic_qc_minimum_expression_value: float = 0.0

    treatment_effect_engine_tag: str = "hermes-2.0-engine-v1.0"
    analysis_scope: str = "research_internal_validation"
    external_validation_required: bool = True


@dataclass
class NeoTRIPScienceAudit:
    """Complete locked cohort-audit bundle."""

    cohort_summary: dict[str, Any]
    treatment_outcome_table: pd.DataFrame
    pcr_by_arm: pd.DataFrame
    subtype_table: pd.DataFrame
    batch_table: pd.DataFrame
    missingness_table: pd.DataFrame
    transcriptomic_qc: pd.DataFrame
    hallmark_coverage: pd.DataFrame
    integrity_checks: dict[str, bool]
    analysis_spec: LockedNeoTRIPAnalysisSpec


def _count_table(
    series: pd.Series,
    *,
    column_name: str,
) -> pd.DataFrame:
    counts = (
        series.astype("string")
        .fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename("n")
        .to_frame()
    )

    counts.index.name = column_name
    counts["fraction"] = (
        counts["n"] / counts["n"].sum()
    )

    return counts


def _build_missingness_table(
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "missing_n": clinical.isna().sum(),
            "missing_fraction": clinical.isna().mean(),
            "unique_nonmissing": clinical.nunique(dropna=True),
        }
    )

    table.index.name = "clinical_variable"

    return table


def _build_transcriptomic_qc_table(
    processed,
) -> pd.DataFrame:
    report = processed.report

    metrics = {
        "patients": report.n_patients,
        "genes_input": report.n_genes_input,
        "genes_after_filtering": report.n_genes_after_filtering,
        "genes_removed": report.genes_removed,
        "retention_fraction": report.retention_fraction,
        "missing_values": report.n_missing_values,
        "constant_genes": report.n_constant_genes,
        "low_variance_genes": report.n_low_variance_genes,
        "low_information_genes": report.n_low_information_genes,
        "variance_threshold": report.variance_threshold,
        "expression_prevalence_threshold": (
            report.expression_prevalence_threshold
        ),
        "minimum_expression_value": report.minimum_expression_value,
    }

    return pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": list(metrics.values()),
        }
    ).set_index("metric")


def run_locked_neotrip_science_audit(
    *,
    expression_path: Path = DEFAULT_EXPRESSION_PATH,
    clinical_path: Path = DEFAULT_CLINICAL_PATH,
    hallmark_gmt_path: Path = DEFAULT_HALLMARK_GMT,
    spec: LockedNeoTRIPAnalysisSpec | None = None,
) -> NeoTRIPScienceAudit:
    """Run the deterministic HERMES 2.0 NeoTRIP cohort audit."""

    if spec is None:
        spec = LockedNeoTRIPAnalysisSpec()

    cohort = load_neotrip_baseline(
        expression_path=expression_path,
        clinical_path=clinical_path,
    )

    clinical = cohort.clinical.copy()

    processed = preprocess_neotrip_baseline(
        variance_threshold=spec.transcriptomic_qc_variance_threshold,
        expression_prevalence_threshold=(
            spec.transcriptomic_qc_expression_prevalence_threshold
        ),
        minimum_expression_value=(
            spec.transcriptomic_qc_minimum_expression_value
        ),
    )

    gene_sets = load_gmt(
        hallmark_gmt_path
    )

    representation = score_gene_sets(
        processed.expression,
        gene_sets,
        min_genes=spec.minimum_genes_per_pathway,
        min_coverage=spec.minimum_pathway_gene_coverage,
        standardize_genes=True,
    )

    treatment_outcome_table = pd.crosstab(
        clinical["Arm"],
        clinical["pCR"],
        dropna=False,
    )

    treatment_outcome_table.index.name = "treatment_arm"
    treatment_outcome_table.columns.name = "outcome"

    pcr_by_arm = (
        clinical.groupby("Arm", observed=False)["pCR.num"]
        .agg(
            patients="count",
            pcr="sum",
            pcr_rate="mean",
        )
    )

    pcr_by_arm.index.name = "treatment_arm"

    subtype_table = _count_table(
        clinical["TNBCtype"],
        column_name="tnbc_type",
    )

    batch_table = _count_table(
        clinical["Batch_correlation"],
        column_name="batch",
    )

    missingness_table = _build_missingness_table(
        clinical
    )

    transcriptomic_qc = _build_transcriptomic_qc_table(
        processed
    )

    hallmark_coverage = representation.coverage.copy()

    patient_ids_records = [
        record.patient_id
        for record in cohort.records
    ]

    patient_ids_clinical = (
        clinical.index.astype(str).tolist()
    )

    patient_ids_expression = (
        cohort.expression.index.astype(str).tolist()
    )

    patient_ids_processed = (
        processed.expression.index.astype(str).tolist()
    )

    patient_ids_hallmark = (
        representation.scores.index.astype(str).tolist()
    )

    integrity_checks = {
        "expected_241_baseline_patients": (
            cohort.n_patients == 241
        ),
        "clinical_expression_patient_alignment": (
            patient_ids_clinical
            == patient_ids_expression
        ),
        "record_clinical_patient_alignment": (
            patient_ids_records
            == patient_ids_clinical
        ),
        "processed_patient_alignment": (
            patient_ids_processed
            == patient_ids_clinical
        ),
        "hallmark_patient_alignment": (
            patient_ids_hallmark
            == patient_ids_clinical
        ),
        "unique_patient_ids": (
            len(patient_ids_clinical)
            == len(set(patient_ids_clinical))
        ),
        "unique_expression_genes": (
            not cohort.expression.columns.duplicated().any()
        ),
        "unique_processed_genes": (
            not processed.expression.columns.duplicated().any()
        ),
        "finite_expression": bool(
            np.isfinite(
                cohort.expression.to_numpy(dtype=float)
            ).all()
        ),
        "finite_processed_expression": bool(
            np.isfinite(
                processed.expression.to_numpy(dtype=float)
            ).all()
        ),
        "finite_hallmark_scores": bool(
            np.isfinite(
                representation.scores.to_numpy(dtype=float)
            ).all()
        ),
        "both_randomized_treatment_arms_present": (
            set(clinical["Arm"].astype(str).unique())
            == {"CT", "CT/A"}
        ),
        "both_binary_outcomes_present": (
            set(clinical["pCR.num"].astype(int).unique())
            == {0, 1}
        ),
        "all_hallmark_sets_retained": (
            representation.n_representations
            == len(gene_sets)
        ),
    }

    cohort_summary = {
        "patients": int(cohort.n_patients),
        "raw_expression_genes": int(cohort.n_genes),
        "processed_expression_genes": int(
            processed.expression.shape[1]
        ),
        "hallmark_sets_loaded": int(len(gene_sets)),
        "hallmark_sets_retained": int(
            representation.n_representations
        ),
        "control_patients_CT": int(
            clinical["Arm"].eq("CT").sum()
        ),
        "experimental_patients_CT_A": int(
            clinical["Arm"].eq("CT/A").sum()
        ),
        "pcr_patients": int(
            clinical["pCR.num"].eq(1).sum()
        ),
        "residual_disease_patients": int(
            clinical["pCR.num"].eq(0).sum()
        ),
        "pcr_rate_CT": float(
            clinical.loc[
                clinical["Arm"].eq("CT"),
                "pCR.num",
            ].mean()
        ),
        "pcr_rate_CT_A": float(
            clinical.loc[
                clinical["Arm"].eq("CT/A"),
                "pCR.num",
            ].mean()
        ),
        "absolute_pcr_rate_difference_CT_A_minus_CT": float(
            clinical.loc[
                clinical["Arm"].eq("CT/A"),
                "pCR.num",
            ].mean()
            - clinical.loc[
                clinical["Arm"].eq("CT"),
                "pCR.num",
            ].mean()
        ),
        "clinical_variables": int(
            clinical.shape[1]
        ),
        "tnbc_subtypes_observed": int(
            clinical["TNBCtype"].nunique(dropna=True)
        ),
        "batches_observed": int(
            clinical["Batch_correlation"].nunique(dropna=True)
        ),
        "all_integrity_checks_passed": bool(
            all(integrity_checks.values())
        ),
    }

    return NeoTRIPScienceAudit(
        cohort_summary=cohort_summary,
        treatment_outcome_table=treatment_outcome_table,
        pcr_by_arm=pcr_by_arm,
        subtype_table=subtype_table,
        batch_table=batch_table,
        missingness_table=missingness_table,
        transcriptomic_qc=transcriptomic_qc,
        hallmark_coverage=hallmark_coverage,
        integrity_checks=integrity_checks,
        analysis_spec=spec,
    )


def export_locked_neotrip_science_audit(
    audit: NeoTRIPScienceAudit,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export the locked cohort audit to reproducible CSV/JSON artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts: dict[str, Path] = {}

    tables = {
        "treatment_outcome_table": audit.treatment_outcome_table,
        "pcr_by_arm": audit.pcr_by_arm,
        "tnbc_subtype_distribution": audit.subtype_table,
        "batch_distribution": audit.batch_table,
        "clinical_missingness": audit.missingness_table,
        "transcriptomic_qc": audit.transcriptomic_qc,
        "hallmark_coverage": audit.hallmark_coverage,
    }

    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path)
        artifacts[name] = path

    summary_path = output_dir / "cohort_audit_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "cohort_summary": audit.cohort_summary,
                "integrity_checks": audit.integrity_checks,
                "analysis_spec": asdict(audit.analysis_spec),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["cohort_audit_summary"] = summary_path

    spec_path = output_dir / "locked_analysis_spec.json"
    spec_path.write_text(
        json.dumps(
            asdict(audit.analysis_spec),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifacts["locked_analysis_spec"] = spec_path

    return artifacts


def summarize_locked_neotrip_science_audit(
    audit: NeoTRIPScienceAudit,
) -> None:
    """Print the locked audit in a compact human-readable format."""

    s = audit.cohort_summary

    print(
        "=== HERMES 2.0 LOCKED NEOTRIP SCIENCE AUDIT ==="
    )
    print()

    print(f"Patients: {s['patients']}")
    print(
        "Treatment allocation: "
        f"CT={s['control_patients_CT']}, "
        f"CT/A={s['experimental_patients_CT_A']}"
    )
    print(
        "Outcome allocation: "
        f"pCR={s['pcr_patients']}, "
        f"RD={s['residual_disease_patients']}"
    )
    print(
        "pCR rate: "
        f"CT={s['pcr_rate_CT']:.4f}, "
        f"CT/A={s['pcr_rate_CT_A']:.4f}"
    )
    print(
        "Absolute pCR difference (CT/A - CT): "
        f"{s['absolute_pcr_rate_difference_CT_A_minus_CT']:.4f}"
    )

    print()
    print(
        "Transcriptome: "
        f"{s['raw_expression_genes']} raw genes -> "
        f"{s['processed_expression_genes']} after QC"
    )
    print(
        "Hallmark pathways: "
        f"{s['hallmark_sets_retained']}/"
        f"{s['hallmark_sets_loaded']} retained"
    )
    print(
        f"TNBC subtypes: {s['tnbc_subtypes_observed']}"
    )
    print(
        f"Batches: {s['batches_observed']}"
    )
    print()

    print("pCR by treatment arm:")
    print(
        audit.pcr_by_arm.to_string()
    )
    print()

    print("TNBC subtype distribution:")
    print(
        audit.subtype_table.to_string()
    )
    print()

    print("Batch distribution:")
    print(
        audit.batch_table.to_string()
    )
    print()

    print("Integrity checks:")
    for key, value in audit.integrity_checks.items():
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        "All integrity checks passed: "
        f"{s['all_integrity_checks_passed']}"
    )
    print()
    print(
        "IMPORTANT:"
    )
    print(
        "This audit locks the cohort and analysis contract before "
        "treatment-effect interpretation."
    )
    print(
        "No HERMES patient-level treatment-effect estimates or modifier "
        "rankings are used by this audit."
    )
    print(
        "External validation remains required for claims of predictive "
        "biomarker validity or clinical generalizability."
    )


def main() -> None:
    audit = run_locked_neotrip_science_audit()

    summarize_locked_neotrip_science_audit(
        audit
    )

    output_dir = Path(
        "outputs/hermes2/science_audit"
    )

    artifacts = export_locked_neotrip_science_audit(
        audit,
        output_dir,
    )

    print()
    print(
        f"Audit artifacts written to: {output_dir}"
    )

    for name, path in artifacts.items():
        print(
            f"  {name}: {path}"
        )


if __name__ == "__main__":
    main()