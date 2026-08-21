"""
HERMES 2.0
I-SPY2 Locked Control-Arm Audit Tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.treatment_effects.external_ispy2_loader import (
    ISPY2ExternalCohort,
)

from backend.app.treatment_effects.external_ispy2_control_audit import (
    _is_locked_control_arm,
    _is_locked_pembro_arm,
    audit_locked_ispy2_control_arm,
    export_control_arm_audit,
)


def _fixture_cohort() -> ISPY2ExternalCohort:
    ids = pd.Index(
        ["ISPY2_1", "ISPY2_2", "ISPY2_3", "ISPY2_4"],
        name="Patient_ID",
    )

    clinical = pd.DataFrame(
        {
            "Patient_ID": ids,
            "HR": [0, 0, 0, 0],
            "HER2": [0, 0, 0, 0],
            "pCR": [0, 1, 1, 1],
            "Arm": [
                "Paclitaxil (Control arm: HER2- subset)",
                "Paclitaxil (Control arm: HER2- subset)",
                "Paclitaxel + Pembrolizumab",
                "Paclitaxel + Pembrolizumab",
            ],
            "locked_treatment": [0, 0, 1, 1],
        },
        index=ids,
    )

    expression = pd.DataFrame(
        np.arange(12, dtype=float).reshape(4, 3),
        index=ids,
        columns=["A", "B", "C"],
    )

    treatment = pd.Series(
        [0, 0, 1, 1],
        index=ids,
        name="T",
    )

    outcome = pd.Series(
        [0, 1, 1, 1],
        index=ids,
        name="Y",
    )

    return ISPY2ExternalCohort(
        expression=expression,
        clinical=clinical,
        treatment=treatment,
        outcome=outcome,
        audit={},
    )


def test_arm_classifiers_are_strict() -> None:
    assert _is_locked_control_arm(
        "Paclitaxil (Control arm: HER2- subset)"
    )
    assert _is_locked_pembro_arm(
        "Paclitaxel + Pembrolizumab"
    )

    assert not _is_locked_control_arm(
        "Paclitaxel + Pembrolizumab"
    )
    assert not _is_locked_pembro_arm(
        "Paclitaxel + Ganitumab"
    )


def test_control_arm_audit_passes_clean_locked_cohort() -> None:
    result = audit_locked_ispy2_control_arm(
        _fixture_cohort()
    )

    assert result.patients == 4
    assert result.control_patients == 2
    assert result.pembrolizumab_patients == 2
    assert result.all_integrity_checks_passed is True
    assert result.published_context_used_for_selection is False


def test_control_arm_audit_rejects_contamination() -> None:
    cohort = _fixture_cohort()

    cohort.clinical.loc[
        "ISPY2_1",
        "Arm",
    ] = "Paclitaxel + Pembrolizumab"

    try:
        audit_locked_ispy2_control_arm(cohort)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Control-arm contamination was not rejected."
        )


def test_control_arm_audit_rejects_non_tnbc_control() -> None:
    cohort = _fixture_cohort()
    cohort.clinical.loc["ISPY2_1", "HR"] = 1

    try:
        audit_locked_ispy2_control_arm(cohort)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "HR-positive patient in locked TNBC cohort was not rejected."
        )


def test_control_arm_audit_export_contract(tmp_path) -> None:
    result = audit_locked_ispy2_control_arm(
        _fixture_cohort()
    )

    generated = export_control_arm_audit(
        result,
        output_dir=tmp_path,
    )

    assert set(generated) == {
        "audit_json",
        "arm_summary",
    }

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0
