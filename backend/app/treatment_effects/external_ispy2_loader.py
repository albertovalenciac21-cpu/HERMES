"""
HERMES 2.0
I-SPY2 External Cohort Loader and Audit
=======================================

SOFT-based replacement loader for GSE194040.

This version NEVER downloads data. It reads the two files already stored at:

data/hermes2/cohorts/ispy2_gse194040/
    GSE194040_family.soft.gz
    GSE194040_ISPY2ResID_AgilentGeneExp_990_FrshFrzn_meanCol_geneLevel_n988.txt.gz

Scientific guardrail:
GSE194040 is microarray data, so this loader does not transport NeoTRIP
raw-scale means/SDs or raw-scale coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


GSE_ACCESSION = "GSE194040"

DEFAULT_EXTERNAL_DIR = Path("data/hermes2/cohorts/ispy2_gse194040")

DEFAULT_EXPRESSION_PATH = DEFAULT_EXTERNAL_DIR / (
    "GSE194040_ISPY2ResID_AgilentGeneExp_990_"
    "FrshFrzn_meanCol_geneLevel_n988.txt.gz"
)

DEFAULT_SOFT_PATH = DEFAULT_EXTERNAL_DIR / "GSE194040_family.soft.gz"

DEFAULT_AUDIT_DIR = Path(
    "outputs/hermes2/external_validation/ispy2_cohort_audit"
)


@dataclass
class ISPY2ExternalCohort:
    expression: pd.DataFrame
    clinical: pd.DataFrame
    treatment: pd.Series
    outcome: pd.Series
    audit: dict[str, Any]

    @property
    def n_patients(self) -> int:
        return int(self.expression.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.expression.shape[1])


def _normalize_ispy2_patient_id(value: object) -> str:
    text = str(value).strip().strip('"')

    match = re.search(r"ISPY2[_\- ]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"ISPY2_{match.group(1)}"

    numeric = re.fullmatch(r"\d+", text)
    if numeric:
        return f"ISPY2_{text}"

    numeric_platform = re.fullmatch(r"(\d+)-GPL\d+", text, flags=re.IGNORECASE)
    if numeric_platform:
        return f"ISPY2_{numeric_platform.group(1)}"

    return text


def _parse_characteristic(value: str) -> tuple[str, str] | None:
    """
    Parse one GEO characteristic and canonicalize known clinical keys.

    Real GSE194040 SOFT uses lower-case keys such as:
        her2, hr, pcr, mp, arm

    HERMES stores them canonically as:
        HER2, HR, pCR, MP, Arm
    """

    value = str(value).strip().strip('"')
    if ":" not in value:
        return None

    key, raw = value.split(":", 1)
    key = key.strip()
    raw = raw.strip()

    if not key:
        return None

    canonical = {
        "her2": "HER2",
        "hr": "HR",
        "pcr": "pCR",
        "mp": "MP",
        "arm": "Arm",
        "arm (short name)": "Arm_short_name",
        "tissue": "tissue",
    }

    normalized_key = " ".join(key.lower().split())
    key = canonical.get(normalized_key, key)

    return key, raw


def parse_geo_family_soft_metadata(path: str | Path) -> pd.DataFrame:
    """
    Parse sample-level metadata from GSE194040_family.soft.gz.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"I-SPY2 family SOFT not found: {path}")

    opener = gzip.open if path.suffix == ".gz" else open

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")

            if line.startswith("^SAMPLE ="):
                if current is not None:
                    records.append(current)

                current = {"GSM": line.split("=", 1)[1].strip()}
                continue

            if current is None:
                continue

            if line.startswith("!Sample_title ="):
                title = line.split("=", 1)[1].strip()
                current["Sample_title"] = title
                current["Patient_ID"] = _normalize_ispy2_patient_id(title)
                continue

            if line.startswith("!Sample_characteristics_ch1 ="):
                characteristic = line.split("=", 1)[1].strip()
                parsed = _parse_characteristic(characteristic)
                if parsed is not None:
                    key, value = parsed
                    current[key] = value
                continue

            if line.startswith("^") and not line.startswith("^SAMPLE ="):
                records.append(current)
                current = None

    if current is not None:
        records.append(current)

    if not records:
        raise ValueError("No SAMPLE records were parsed from GEO family SOFT.")

    clinical = pd.DataFrame(records)

    required = {"Patient_ID", "GSM", "HR", "HER2", "pCR", "Arm"}
    missing = sorted(required - set(clinical.columns))
    if missing:
        raise ValueError(
            "GEO family SOFT missing required fields: "
            f"{missing}"
        )

    for column in ("HR", "HER2", "pCR"):
        clinical[column] = pd.to_numeric(
            clinical[column], errors="raise"
        ).astype(int)

    if "MP" in clinical.columns:
        clinical["MP"] = pd.to_numeric(
            clinical["MP"], errors="coerce"
        )

    # Do not reject duplicate patient IDs at the full GEO-family level.
    # GSE194040 contains some patients assayed on more than one Agilent
    # platform. Duplicate resolution is performed deterministically after
    # the locked treatment/TNBC cohort is selected.
    return clinical.reset_index(drop=True)


def _find_gene_symbol_column(columns: list[str]) -> str:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(column).lower()): str(column)
        for column in columns
    }

    preferred = (
        "genesymbol",
        "genesymbols",
        "hgncsymbol",
        "hgncapprovedsymbol",
        "symbol",
        "gene",
        "genename",
    )

    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]

    if columns:
        return str(columns[0])

    raise ValueError("Expression matrix has no columns.")


def _is_sample_column(column: object) -> bool:
    """
    Recognize all public GSE194040 sample-column formats used by the
    processed gene-level matrix.

    Examples:
        ISPY2_100001
        629606
        629606-GPL16233
        629606-GPL20078
    """
    text = str(column).strip().strip('"')

    return bool(
        re.fullmatch(
            r"ISPY2[_\- ]?\d+(?:-GPL\d+)?",
            text,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            r"\d{5,}(?:-GPL\d+)?",
            text,
            flags=re.IGNORECASE,
        )
    )


def load_ispy2_gene_expression(path: str | Path) -> pd.DataFrame:
    """
    Load the normalized GSE194040 gene-level matrix.

    Returns rows=patients, columns=unique upper-case gene symbols.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"I-SPY2 expression file not found: {path}")

    raw = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        low_memory=False,
    )

    if raw.empty:
        raise ValueError("I-SPY2 expression file is empty.")

    columns = [str(c) for c in raw.columns]
    gene_column = _find_gene_symbol_column(columns)

    gene_symbols = (
        raw[gene_column]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_gene = (
        gene_symbols.ne("")
        & gene_symbols.str.lower().ne("nan")
    )

    sample_columns = [
        str(column)
        for column in raw.columns
        if str(column) != gene_column and _is_sample_column(column)
    ]

    if not sample_columns:
        raise ValueError(
            "No I-SPY2 patient columns were identified in expression matrix. "
            f"Leading columns: {columns[:12]}"
        )

    values = raw.loc[
        valid_gene,
        sample_columns,
    ].apply(pd.to_numeric, errors="coerce")

    values.index = gene_symbols.loc[valid_gene]
    values = values.loc[~values.isna().all(axis=1)]
    values = values.groupby(level=0).mean()

    expression = values.T
    expression.index = [
        _normalize_ispy2_patient_id(sample)
        for sample in expression.index
    ]
    expression.index.name = "Patient_ID"

    # The public processed matrix can contain the same biopsy measured on
    # multiple Agilent platforms (e.g. 629606-GPL16233 and 629606-GPL20078).
    # Because GEO distributes this matrix as normalized/batch-corrected
    # gene-level expression, collapse technical/platform replicate columns
    # to one patient-level profile by the arithmetic mean.
    if expression.index.duplicated().any():
        expression = expression.groupby(level=0, sort=False).mean()

    expression = expression.loc[
        :,
        ~expression.isna().any(axis=0),
    ]

    if expression.empty:
        raise ValueError("No complete expression genes remain.")

    if not np.isfinite(expression.to_numpy(dtype=float)).all():
        raise ValueError("I-SPY2 expression contains non-finite values.")

    return expression


def _normalize_arm(value: object) -> str:
    return " ".join(
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
        .split()
    )


def _classify_locked_arm(value: object) -> int | None:
    """
    0 = HER2-negative paclitaxel control
    1 = paclitaxel + pembrolizumab
    None = all other I-SPY2 arms
    """

    arm = _normalize_arm(value)

    if "paclitax" in arm and "pembrolizumab" in arm:
        return 1

    if "paclitax" in arm and "control arm" in arm and "her2-" in arm:
        return 0

    if arm in {"paclitaxel", "paclitaxil"}:
        return 0

    return None


def select_locked_ispy2_tnbc_cohort(
    clinical_all: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Patient_ID", "HR", "HER2", "pCR", "Arm"}
    missing = sorted(required - set(clinical_all.columns))
    if missing:
        raise ValueError(
            f"I-SPY2 clinical table missing required fields: {missing}"
        )

    clinical = clinical_all.copy()
    clinical["locked_treatment"] = clinical["Arm"].map(
        _classify_locked_arm
    )

    cohort = clinical.loc[
        clinical["HR"].eq(0)
        & clinical["HER2"].eq(0)
        & clinical["locked_treatment"].notna()
    ].copy()

    cohort["locked_treatment"] = cohort[
        "locked_treatment"
    ].astype(int)

    if cohort.empty:
        raise ValueError(
            "No TNBC pembrolizumab/control patients were selected."
        )

    if set(cohort["locked_treatment"].unique()) != {0, 1}:
        raise ValueError(
            "Selected I-SPY2 cohort must contain both treatment arms."
        )

    if set(cohort["pCR"].unique()) != {0, 1}:
        raise ValueError(
            "Selected I-SPY2 cohort must contain both pCR outcomes."
        )

    if cohort["Patient_ID"].duplicated().any():
        key_columns = [
            "HR",
            "HER2",
            "pCR",
            "Arm",
            "locked_treatment",
        ]

        discordant: list[str] = []
        for patient_id, group in cohort.groupby("Patient_ID", sort=False):
            if len(group) <= 1:
                continue

            for column in key_columns:
                if group[column].astype(str).nunique(dropna=False) > 1:
                    discordant.append(str(patient_id))
                    break

        if discordant:
            raise ValueError(
                "Duplicate locked-cohort patients have discordant clinical "
                f"annotations: {sorted(discordant)[:10]}"
            )

        cohort = cohort.drop_duplicates(
            subset=["Patient_ID"],
            keep="first",
        )

    return cohort.reset_index(drop=True)


def build_ispy2_external_cohort(
    *,
    expression_path: str | Path = DEFAULT_EXPRESSION_PATH,
    soft_path: str | Path = DEFAULT_SOFT_PATH,
) -> ISPY2ExternalCohort:
    clinical_all = parse_geo_family_soft_metadata(soft_path)
    clinical = select_locked_ispy2_tnbc_cohort(clinical_all)

    expression_all = load_ispy2_gene_expression(expression_path)

    clinical_ids = pd.Index(
        clinical["Patient_ID"].astype(str),
        name="Patient_ID",
    )

    missing_expression = sorted(
        set(clinical_ids) - set(expression_all.index.astype(str))
    )

    if missing_expression:
        raise ValueError(
            "Selected I-SPY2 clinical patients missing expression: "
            f"{missing_expression[:20]}"
        )

    expression = expression_all.loc[clinical_ids].copy()

    clinical = (
        clinical.set_index("Patient_ID", drop=False)
        .loc[expression.index]
        .copy()
    )

    treatment = clinical["locked_treatment"].astype(int).rename("T")
    outcome = clinical["pCR"].astype(int).rename("Y")

    if not expression.index.equals(clinical.index):
        raise RuntimeError(
            "I-SPY2 clinical table is not aligned with expression."
        )
    if not expression.index.equals(treatment.index):
        raise RuntimeError(
            "I-SPY2 treatment vector is not aligned with expression."
        )
    if not expression.index.equals(outcome.index):
        raise RuntimeError(
            "I-SPY2 outcome vector is not aligned with expression."
        )

    audit = {
        "accession": GSE_ACCESSION,
        "patients": int(len(expression)),
        "genes": int(expression.shape[1]),
        "control_patients": int(treatment.eq(0).sum()),
        "pembrolizumab_patients": int(treatment.eq(1).sum()),
        "control_pcr": int(outcome.loc[treatment.eq(0)].sum()),
        "pembrolizumab_pcr": int(outcome.loc[treatment.eq(1)].sum()),
        "control_pcr_rate": float(
            outcome.loc[treatment.eq(0)].mean()
        ),
        "pembrolizumab_pcr_rate": float(
            outcome.loc[treatment.eq(1)].mean()
        ),
        "hr_negative_all": bool(clinical["HR"].eq(0).all()),
        "her2_negative_all": bool(clinical["HER2"].eq(0).all()),
        "both_treatment_arms_present": bool(
            set(treatment.unique()) == {0, 1}
        ),
        "both_outcomes_present": bool(
            set(outcome.unique()) == {0, 1}
        ),
        "unique_patient_ids": bool(
            not expression.index.duplicated().any()
        ),
        "finite_expression": bool(
            np.isfinite(expression.to_numpy(dtype=float)).all()
        ),
        "clinical_expression_alignment": bool(
            expression.index.equals(clinical.index)
        ),
        "technical_platform_replicates_collapsed_by_mean": True,
        "direct_neotrip_raw_scale_transport_used": False,
    }

    required_checks = (
        "hr_negative_all",
        "her2_negative_all",
        "both_treatment_arms_present",
        "both_outcomes_present",
        "unique_patient_ids",
        "finite_expression",
        "clinical_expression_alignment",
    )

    audit["all_integrity_checks_passed"] = bool(
        all(audit[key] for key in required_checks)
    )

    if not audit["all_integrity_checks_passed"]:
        raise RuntimeError(
            "I-SPY2 external cohort failed integrity audit."
        )

    return ISPY2ExternalCohort(
        expression=expression,
        clinical=clinical,
        treatment=treatment,
        outcome=outcome,
        audit=audit,
    )


def export_ispy2_cohort_audit(
    cohort: ISPY2ExternalCohort,
    output_dir: str | Path = DEFAULT_AUDIT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clinical_path = output_dir / "locked_ispy2_tnbc_clinical.csv"
    cohort.clinical.to_csv(clinical_path, index=False)

    expression_summary = pd.DataFrame(
        {
            "Patient_ID": cohort.expression.index.astype(str),
            "mean_expression": cohort.expression.mean(axis=1).to_numpy(),
            "sd_expression": cohort.expression.std(
                axis=1, ddof=0
            ).to_numpy(),
            "minimum_expression": cohort.expression.min(axis=1).to_numpy(),
            "maximum_expression": cohort.expression.max(axis=1).to_numpy(),
        }
    )

    expression_summary_path = (
        output_dir / "locked_ispy2_expression_summary.csv"
    )
    expression_summary.to_csv(expression_summary_path, index=False)

    audit_path = output_dir / "locked_ispy2_cohort_audit.json"
    audit_path.write_text(
        json.dumps(cohort.audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "clinical": clinical_path,
        "expression_summary": expression_summary_path,
        "cohort_audit": audit_path,
    }


def main() -> None:
    print("=== HERMES 2.0 I-SPY2 EXTERNAL COHORT AUDIT ===")
    print()

    missing = [
        path
        for path in (DEFAULT_EXPRESSION_PATH, DEFAULT_SOFT_PATH)
        if not path.exists()
    ]

    if missing:
        print("Required public files are not present locally:")
        for path in missing:
            print(f"  - {path}")
        raise FileNotFoundError(
            "I-SPY2 external-validation inputs are incomplete."
        )

    cohort = build_ispy2_external_cohort()

    print(f"Accession: {GSE_ACCESSION}")
    print(f"Patients: {cohort.n_patients}")
    print(f"Genes: {cohort.n_genes}")
    print(
        "Arms: "
        f"control={cohort.audit['control_patients']}, "
        f"pembrolizumab={cohort.audit['pembrolizumab_patients']}"
    )
    print(
        "pCR rates: "
        f"control={cohort.audit['control_pcr_rate']:.4f}, "
        f"pembrolizumab={cohort.audit['pembrolizumab_pcr_rate']:.4f}"
    )
    print(
        "All integrity checks passed: "
        f"{cohort.audit['all_integrity_checks_passed']}"
    )
    print()
    print("IMPORTANT:")
    print(
        "This is a cross-platform external replication cohort. "
        "No NeoTRIP raw-scale model transport has been performed."
    )
    print(
        "No external HERMES treatment-effect model has been fitted by this "
        "loader/audit."
    )

    artifacts = export_ispy2_cohort_audit(cohort)

    print()
    print(f"Audit artifacts written to: {DEFAULT_AUDIT_DIR}")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
