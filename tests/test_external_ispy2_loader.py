"""
HERMES 2.0
SOFT-based I-SPY2 Loader Tests
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from backend.app.treatment_effects.external_ispy2_loader import (
    _classify_locked_arm,
    _normalize_ispy2_patient_id,
    build_ispy2_external_cohort,
    export_ispy2_cohort_audit,
    load_ispy2_gene_expression,
    parse_geo_family_soft_metadata,
    select_locked_ispy2_tnbc_cohort,
)


def _write_soft(path) -> None:
    samples = [
        ("GSM1", "ISPY2_100001", 0, 0, 0,
         "Paclitaxil (Control arm: HER2- subset)"),
        ("GSM2", "ISPY2_100002", 0, 0, 1,
         "Paclitaxil (Control arm: HER2- subset)"),
        ("GSM3", "ISPY2_100003", 0, 0, 0,
         "Paclitaxel + Pembrolizumab"),
        ("GSM4", "ISPY2_100004", 0, 0, 1,
         "Paclitaxel + Pembrolizumab"),
        ("GSM5", "ISPY2_100005", 1, 0, 1,
         "Paclitaxel + Pembrolizumab"),
        ("GSM6", "ISPY2_100006", 0, 1, 1,
         "Paclitaxel + Pembrolizumab"),
    ]

    lines = []
    for gsm, title, hr, her2, pcr, arm in samples:
        lines.extend(
            [
                f"^SAMPLE = {gsm}",
                f"!Sample_title = {title}",
                f"!Sample_characteristics_ch1 = hr: {hr}",
                f"!Sample_characteristics_ch1 = her2: {her2}",
                f"!Sample_characteristics_ch1 = pcr: {pcr}",
                "!Sample_characteristics_ch1 = mp: 1",
                f"!Sample_characteristics_ch1 = arm: {arm}",
            ]
        )

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_expression(path) -> None:
    frame = pd.DataFrame(
        {
            "GeneSymbol": ["STAT1", "CXCL9", "VIM", "VIM", ""],
            "ISPY2_100001": [1.0, 2.0, 3.0, 5.0, 0.0],
            "ISPY2_100002": [2.0, 3.0, 4.0, 6.0, 0.0],
            "ISPY2_100003": [3.0, 4.0, 5.0, 7.0, 0.0],
            "ISPY2_100004": [4.0, 5.0, 6.0, 8.0, 0.0],
            "ISPY2_100005": [5.0, 6.0, 7.0, 9.0, 0.0],
            "ISPY2_100006": [6.0, 7.0, 8.0, 10.0, 0.0],
        }
    )
    frame.to_csv(path, sep="\t", index=False, compression="gzip")


def test_soft_parser_recovers_required_clinical_fields(tmp_path) -> None:
    path = tmp_path / "family.soft.gz"
    _write_soft(path)

    clinical = parse_geo_family_soft_metadata(path)

    assert len(clinical) == 6
    assert {
        "Patient_ID",
        "GSM",
        "HR",
        "HER2",
        "pCR",
        "MP",
        "Arm",
    }.issubset(clinical.columns)
    assert clinical["Patient_ID"].iloc[0] == "ISPY2_100001"


def test_locked_arm_classifier_is_conservative() -> None:
    assert _classify_locked_arm(
        "Paclitaxil (Control arm: HER2- subset)"
    ) == 0
    assert _classify_locked_arm(
        "Paclitaxel + Pembrolizumab"
    ) == 1
    assert _classify_locked_arm(
        "Paclitaxel + Trastuzumab"
    ) is None


def test_locked_tnbc_selection_requires_hr_and_her2_negative(
    tmp_path,
) -> None:
    path = tmp_path / "family.soft.gz"
    _write_soft(path)

    clinical = parse_geo_family_soft_metadata(path)
    selected = select_locked_ispy2_tnbc_cohort(clinical)

    assert len(selected) == 4
    assert selected["HR"].eq(0).all()
    assert selected["HER2"].eq(0).all()
    assert set(selected["locked_treatment"]) == {0, 1}


def test_gene_expression_loader_averages_duplicate_genes(
    tmp_path,
) -> None:
    path = tmp_path / "expression.txt.gz"
    _write_expression(path)

    expression = load_ispy2_gene_expression(path)

    assert expression.shape == (6, 3)
    assert list(expression.columns) == ["CXCL9", "STAT1", "VIM"]
    assert np.isclose(
        expression.loc["ISPY2_100001", "VIM"],
        4.0,
    )


def test_end_to_end_external_cohort_alignment(tmp_path) -> None:
    soft = tmp_path / "family.soft.gz"
    expression = tmp_path / "expression.txt.gz"

    _write_soft(soft)
    _write_expression(expression)

    cohort = build_ispy2_external_cohort(
        expression_path=expression,
        soft_path=soft,
    )

    assert cohort.n_patients == 4
    assert cohort.n_genes == 3
    assert cohort.expression.index.equals(cohort.clinical.index)
    assert cohort.expression.index.equals(cohort.treatment.index)
    assert cohort.expression.index.equals(cohort.outcome.index)
    assert cohort.audit["all_integrity_checks_passed"] is True


def test_external_cohort_audit_export_contract(tmp_path) -> None:
    soft = tmp_path / "family.soft.gz"
    expression = tmp_path / "expression.txt.gz"

    _write_soft(soft)
    _write_expression(expression)

    cohort = build_ispy2_external_cohort(
        expression_path=expression,
        soft_path=soft,
    )

    generated = export_ispy2_cohort_audit(
        cohort,
        output_dir=tmp_path / "audit",
    )

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0

    audit = json.loads(
        generated["cohort_audit"].read_text(encoding="utf-8")
    )
    assert audit["all_integrity_checks_passed"] is True


def test_numeric_platform_expression_id_normalization() -> None:
    assert _normalize_ispy2_patient_id(
        "629606-GPL16233"
    ) == "ISPY2_629606"

    assert _normalize_ispy2_patient_id(
        "629606-GPL20078"
    ) == "ISPY2_629606"


def test_expression_platform_replicates_are_collapsed_by_mean(
    tmp_path,
) -> None:
    frame = pd.DataFrame(
        {
            "GeneSymbol": ["STAT1", "VIM"],
            "629606-GPL16233": [1.0, 3.0],
            "629606-GPL20078": [5.0, 7.0],
            "700001-GPL16233": [2.0, 4.0],
        }
    )
    path = tmp_path / "expression_platform_reps.txt.gz"
    frame.to_csv(
        path,
        sep="\t",
        index=False,
        compression="gzip",
    )

    expression = load_ispy2_gene_expression(path)

    assert list(expression.index) == [
        "ISPY2_629606",
        "ISPY2_700001",
    ]
    assert np.isclose(
        expression.loc["ISPY2_629606", "STAT1"],
        3.0,
    )
    assert np.isclose(
        expression.loc["ISPY2_629606", "VIM"],
        5.0,
    )


def test_irrelevant_duplicate_platform_samples_do_not_block_locked_cohort(
    tmp_path,
) -> None:
    path = tmp_path / "family_with_irrelevant_duplicate.soft.gz"

    lines = [
        "^SAMPLE = GSM_A",
        "!Sample_title = ISPY2_629606-GPL16233",
        "!Sample_characteristics_ch1 = hr: 0",
        "!Sample_characteristics_ch1 = her2: 0",
        "!Sample_characteristics_ch1 = pcr: 0",
        "!Sample_characteristics_ch1 = arm: Paclitaxel + Ganitumab",
        "^SAMPLE = GSM_B",
        "!Sample_title = ISPY2_629606-GPL20078",
        "!Sample_characteristics_ch1 = hr: 0",
        "!Sample_characteristics_ch1 = her2: 0",
        "!Sample_characteristics_ch1 = pcr: 0",
        "!Sample_characteristics_ch1 = arm: Paclitaxel + Ganitumab",
        "^SAMPLE = GSM_C",
        "!Sample_title = ISPY2_100001",
        "!Sample_characteristics_ch1 = hr: 0",
        "!Sample_characteristics_ch1 = her2: 0",
        "!Sample_characteristics_ch1 = pcr: 0",
        "!Sample_characteristics_ch1 = arm: Paclitaxil (Control arm: HER2- subset)",
        "^SAMPLE = GSM_D",
        "!Sample_title = ISPY2_100002",
        "!Sample_characteristics_ch1 = hr: 0",
        "!Sample_characteristics_ch1 = her2: 0",
        "!Sample_characteristics_ch1 = pcr: 1",
        "!Sample_characteristics_ch1 = arm: Paclitaxel + Pembrolizumab",
    ]

    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    clinical = parse_geo_family_soft_metadata(path)
    assert clinical["Patient_ID"].eq("ISPY2_629606").sum() == 2

    selected = select_locked_ispy2_tnbc_cohort(clinical)
    assert set(selected["Patient_ID"]) == {
        "ISPY2_100001",
        "ISPY2_100002",
    }
