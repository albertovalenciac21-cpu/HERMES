"""
HERMES 2.0
I-SPY2 Locked Control-Arm Audit
===============================

This module audits the already-constructed GSE194040 TNBC external cohort
before external HERMES treatment-effect modeling. It does not alter cohort
membership and does not fit or tune a model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from backend.app.treatment_effects.external_ispy2_loader import (
    ISPY2ExternalCohort,
    build_ispy2_external_cohort,
)


DEFAULT_OUTPUT_DIR = Path(
    "outputs/hermes2/external_validation/ispy2_control_arm_audit"
)

PUBLISHED_CONTEXT_TNBC_TOTAL = 114
PUBLISHED_CONTEXT_TNBC_PEMBRO = 29
PUBLISHED_CONTEXT_TNBC_CONTROL = 85


@dataclass(frozen=True)
class ISPY2ControlAuditResult:
    patients: int
    control_patients: int
    pembrolizumab_patients: int
    control_pcr_rate: float
    pembrolizumab_pcr_rate: float

    unique_control_arm_labels: tuple[str, ...]
    unique_pembrolizumab_arm_labels: tuple[str, ...]

    control_all_hr_negative: bool
    control_all_her2_negative: bool
    pembro_all_hr_negative: bool
    pembro_all_her2_negative: bool

    control_exact_locked_arm: bool
    pembro_exact_locked_arm: bool
    no_control_pembro_contamination: bool
    no_pembro_control_contamination: bool

    binary_treatment: bool
    binary_outcome: bool
    patient_ids_unique: bool
    expression_alignment: bool

    all_integrity_checks_passed: bool

    published_context_tnbc_total: int = PUBLISHED_CONTEXT_TNBC_TOTAL
    published_context_tnbc_pembro: int = PUBLISHED_CONTEXT_TNBC_PEMBRO
    published_context_tnbc_control: int = PUBLISHED_CONTEXT_TNBC_CONTROL
    published_context_used_for_selection: bool = False


def _normalized_arm(value: object) -> str:
    return " ".join(
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
        .split()
    )


def _is_locked_control_arm(value: object) -> bool:
    """
    Return True only for the locked HER2-negative paclitaxel control arm.

    Accepted public GEO representations:
      - Paclitaxil (Control arm: HER2- subset)
      - Paclitaxel (Control arm: HER2- subset)
      - Paclitaxel / Paclitaxil (shortened representation)
    """
    arm = _normalized_arm(value)

    if (
        "paclitax" in arm
        and "control arm" in arm
        and "her2-" in arm
    ):
        return True

    return arm in {"paclitaxel", "paclitaxil"}


def _is_locked_pembro_arm(value: object) -> bool:
    """
    Return True only for the locked pembrolizumab + paclitaxel arm.
    """
    arm = _normalized_arm(value)

    return (
        "paclitax" in arm
        and "pembrolizumab" in arm
    )


def audit_locked_ispy2_control_arm(
    cohort: ISPY2ExternalCohort,
) -> ISPY2ControlAuditResult:
    clinical = cohort.clinical.copy()
    treatment = cohort.treatment
    outcome = cohort.outcome

    if not clinical.index.equals(treatment.index):
        raise ValueError("Clinical/treatment indices are not aligned.")
    if not clinical.index.equals(outcome.index):
        raise ValueError("Clinical/outcome indices are not aligned.")
    if not clinical.index.equals(cohort.expression.index):
        raise ValueError("Clinical/expression indices are not aligned.")

    control = clinical.loc[treatment.eq(0)].copy()
    pembro = clinical.loc[treatment.eq(1)].copy()

    if control.empty or pembro.empty:
        raise ValueError("Both locked treatment arms are required.")

    control_labels = tuple(
        sorted(control["Arm"].astype(str).unique().tolist())
    )
    pembro_labels = tuple(
        sorted(pembro["Arm"].astype(str).unique().tolist())
    )

    checks = {
        "control_all_hr_negative": bool(control["HR"].eq(0).all()),
        "control_all_her2_negative": bool(control["HER2"].eq(0).all()),
        "pembro_all_hr_negative": bool(pembro["HR"].eq(0).all()),
        "pembro_all_her2_negative": bool(pembro["HER2"].eq(0).all()),
        "control_exact_locked_arm": bool(
            control["Arm"].map(_is_locked_control_arm).all()
        ),
        "pembro_exact_locked_arm": bool(
            pembro["Arm"].map(_is_locked_pembro_arm).all()
        ),
        "no_control_pembro_contamination": bool(
            ~control["Arm"].map(_is_locked_pembro_arm).any()
        ),
        "no_pembro_control_contamination": bool(
            ~pembro["Arm"].map(_is_locked_control_arm).any()
        ),
        "binary_treatment": bool(set(treatment.unique()) == {0, 1}),
        "binary_outcome": bool(set(outcome.unique()) == {0, 1}),
        "patient_ids_unique": bool(not clinical.index.duplicated().any()),
        "expression_alignment": bool(
            cohort.expression.index.equals(clinical.index)
        ),
    }

    result = ISPY2ControlAuditResult(
        patients=int(len(clinical)),
        control_patients=int(treatment.eq(0).sum()),
        pembrolizumab_patients=int(treatment.eq(1).sum()),
        control_pcr_rate=float(outcome.loc[treatment.eq(0)].mean()),
        pembrolizumab_pcr_rate=float(
            outcome.loc[treatment.eq(1)].mean()
        ),
        unique_control_arm_labels=control_labels,
        unique_pembrolizumab_arm_labels=pembro_labels,
        all_integrity_checks_passed=bool(all(checks.values())),
        **checks,
    )

    if not result.all_integrity_checks_passed:
        failed = [
            key for key, value in checks.items()
            if not value
        ]
        raise ValueError(
            "Locked I-SPY2 control-arm audit failed: "
            f"{failed}"
        )

    return result


def export_control_arm_audit(
    result: ISPY2ControlAuditResult,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "locked_ispy2_control_arm_audit.json"
    json_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary_path = output_dir / "locked_ispy2_control_arm_summary.csv"

    pd.DataFrame(
        [
            {
                "arm": "control",
                "patients": result.control_patients,
                "pcr_rate": result.control_pcr_rate,
                "labels": " | ".join(result.unique_control_arm_labels),
            },
            {
                "arm": "pembrolizumab",
                "patients": result.pembrolizumab_patients,
                "pcr_rate": result.pembrolizumab_pcr_rate,
                "labels": " | ".join(
                    result.unique_pembrolizumab_arm_labels
                ),
            },
        ]
    ).to_csv(summary_path, index=False)

    return {
        "audit_json": json_path,
        "arm_summary": summary_path,
    }


def main() -> None:
    print("=== HERMES 2.0 I-SPY2 CONTROL-ARM AUDIT ===")
    print()

    cohort = build_ispy2_external_cohort()
    result = audit_locked_ispy2_control_arm(cohort)

    print(f"Patients: {result.patients}")
    print(
        f"Arms: control={result.control_patients}, "
        f"pembrolizumab={result.pembrolizumab_patients}"
    )
    print(
        f"pCR rates: control={result.control_pcr_rate:.4f}, "
        f"pembrolizumab={result.pembrolizumab_pcr_rate:.4f}"
    )
    print()

    print("Control labels:")
    for label in result.unique_control_arm_labels:
        print(f"  - {label}")

    print("Pembrolizumab labels:")
    for label in result.unique_pembrolizumab_arm_labels:
        print(f"  - {label}")

    print()
    print(
        "All control-arm integrity checks passed: "
        f"{result.all_integrity_checks_passed}"
    )
    print()
    print("IMPORTANT:")
    print(
        "Published/context counts are descriptive only and were not used "
        "to alter cohort membership."
    )
    print(
        "This audit does not fit or tune an external HERMES model."
    )

    generated = export_control_arm_audit(result)

    print()
    print(f"Artifacts written to: {DEFAULT_OUTPUT_DIR}")
    for name, path in generated.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
