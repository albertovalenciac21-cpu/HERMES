"""
HERMES 2.0
Biological Characterization Tests
=================================

Tests the exploratory biological-characterization layer without requiring
generated primary-analysis outputs.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.treatment_effects.feature_builder import (
    TreatmentEffectDataset,
)
from backend.app.treatment_effects.biological_characterization import (
    BiologicalCharacterizationPlan,
    BiologicalCharacterizationResult,
    _benjamini_hochberg,
    build_characterization_patient_table,
    build_ite_group_summary,
    build_subtype_by_ite_group,
    build_subtype_summary,
    compute_pathway_associations,
    export_biological_characterization,
)


def _dataset() -> TreatmentEffectDataset:
    rng = np.random.default_rng(14)
    n = 80

    index = pd.Index(
        [f"P{i:03d}" for i in range(n)],
        name="Patient_ID",
    )

    latent = rng.normal(size=n)

    X = pd.DataFrame(
        {
            "HALLMARK_POSITIVE": latent + rng.normal(0, 0.2, size=n),
            "HALLMARK_NEGATIVE": -latent + rng.normal(0, 0.2, size=n),
            "HALLMARK_NOISE_1": rng.normal(size=n),
            "HALLMARK_NOISE_2": rng.normal(size=n),
        },
        index=index,
    )

    T = pd.Series(
        np.tile([0, 1], n // 2),
        index=index,
        name="T",
        dtype=int,
    )

    Y = pd.Series(
        (latent + 0.3 * T.to_numpy() + rng.normal(size=n) > 0).astype(int),
        index=index,
        name="Y",
        dtype=int,
    )

    subtype = np.resize(
        np.array(["BL1", "M", "IM", "LAR"]),
        n,
    )

    metadata = pd.DataFrame(
        {
            "tnbc_type": subtype,
            "treatment_label": T.map({0: "CT", 1: "CT/A"}),
            "outcome_label": Y.map({0: "RD", 1: "pCR"}),
        },
        index=index,
    )

    dataset = TreatmentEffectDataset(
        X=X,
        T=T,
        Y=Y,
        metadata=metadata,
        summary={},
    )
    dataset.validate()
    return dataset


def _ite(dataset: TreatmentEffectDataset) -> pd.DataFrame:
    rng = np.random.default_rng(15)

    latent = dataset.X["HALLMARK_POSITIVE"].to_numpy(dtype=float)

    frame = pd.DataFrame(
        index=dataset.X.index,
    )

    for repeat in range(6):
        frame[f"repeat_{repeat + 1:03d}"] = (
            0.05
            + 0.08 * latent
            + rng.normal(0, 0.02, size=len(dataset.X))
        )

    return frame


def _uncertainty(dataset: TreatmentEffectDataset) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "evidence_state": np.resize(
                np.array(
                    [
                        "likely_benefit",
                        "indeterminate",
                        "likely_harm",
                    ]
                ),
                dataset.n_patients,
            )
        },
        index=dataset.X.index,
    )


def _plan() -> BiologicalCharacterizationPlan:
    return BiologicalCharacterizationPlan(
        plan_name="biology_test",
        pathway_reporting_top_n=2,
    )


def _result() -> BiologicalCharacterizationResult:
    dataset = _dataset()
    ite = _ite(dataset)
    patient = build_characterization_patient_table(
        dataset,
        ite,
        _uncertainty(dataset),
        plan=_plan(),
    )
    pathways = compute_pathway_associations(
        dataset.X,
        ite,
        patient,
    )

    return BiologicalCharacterizationResult(
        plan=_plan(),
        source_manifest={
            "plan_name": "hermes2_neotrip_primary_locked_v1",
            "plan_sha256": "a" * 64,
            "engine_tag": "hermes-2.0-engine-v1.0",
        },
        dataset=dataset,
        patient_table=patient,
        pathway_associations=pathways,
        subtype_summary=build_subtype_summary(patient),
        subtype_by_ite_group=build_subtype_by_ite_group(patient),
        ite_group_summary=build_ite_group_summary(patient),
        top_positive_pathways=(
            pathways.loc[
                pathways["spearman_rho_mean_ite"] > 0
            ].head(2)
        ),
        top_negative_pathways=(
            pathways.loc[
                pathways["spearman_rho_mean_ite"] < 0
            ].head(2)
        ),
    )


def test_benjamini_hochberg_is_monotone_and_bounded() -> None:
    p = pd.Series([0.001, 0.02, 0.03, 0.50])
    q = _benjamini_hochberg(p)

    assert ((q >= 0) & (q <= 1)).all()
    assert q.iloc[0] <= q.iloc[1] <= q.iloc[2] <= q.iloc[3]


def test_patient_characterization_preserves_all_patients() -> None:
    dataset = _dataset()
    ite = _ite(dataset)

    table = build_characterization_patient_table(
        dataset,
        ite,
        _uncertainty(dataset),
        plan=_plan(),
    )

    assert len(table) == dataset.n_patients
    assert table["Patient_ID"].is_unique
    assert set(table["ite_group"]) == {
        "low_ite",
        "middle_ite",
        "high_ite",
    }

    assert table["ite_group"].eq("high_ite").sum() >= 20
    assert table["ite_group"].eq("low_ite").sum() >= 20


def test_pathway_characterization_recovers_known_direction() -> None:
    dataset = _dataset()
    ite = _ite(dataset)

    patient = build_characterization_patient_table(
        dataset,
        ite,
        _uncertainty(dataset),
        plan=_plan(),
    )

    table = compute_pathway_associations(
        dataset.X,
        ite,
        patient,
    ).set_index("pathway")

    assert (
        table.loc[
            "HALLMARK_POSITIVE",
            "spearman_rho_mean_ite",
        ]
        > 0.8
    )

    assert (
        table.loc[
            "HALLMARK_NEGATIVE",
            "spearman_rho_mean_ite",
        ]
        < -0.8
    )

    assert (
        table.loc[
            "HALLMARK_POSITIVE",
            "repeat_spearman_sign_stability",
        ]
        == 1.0
    )


def test_subtype_and_ite_group_tables_conserve_patient_count() -> None:
    result = _result()

    assert result.subtype_summary["patients"].sum() == (
        result.dataset.n_patients
    )

    assert result.subtype_by_ite_group["total"].sum() == (
        result.dataset.n_patients
    )

    assert result.ite_group_summary["patients"].sum() == (
        result.dataset.n_patients
    )


def test_characterization_plan_prohibits_biomarker_claims() -> None:
    plan = BiologicalCharacterizationPlan()

    assert plan.source_primary_plan == (
        "hermes2_neotrip_primary_locked_v1"
    )
    assert plan.source_engine_tag == "hermes-2.0-engine-v1.0"
    assert plan.predictive_biomarker_claims_allowed is False
    assert plan.external_validation_required is True


def test_biological_characterization_export_contract(tmp_path) -> None:
    result = _result()

    generated = export_biological_characterization(
        result,
        output_dir=tmp_path / "biology",
    )

    required = {
        "table__patient_biological_characterization",
        "table__hallmark_ite_associations",
        "table__tnbc_subtype_summary",
        "table__tnbc_subtype_by_ite_group",
        "table__ite_group_summary",
        "table__top_positive_pathways",
        "table__top_negative_pathways",
        "figure_1_pathway_ite_correlations",
        "figure_2_high_vs_low_pathway_effects",
        "figure_3_subtype_ite_distribution",
        "figure_4_pathway_repeat_stability",
        "biological_characterization_manifest",
    }

    assert required.issubset(generated)

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0

    manifest = json.loads(
        generated[
            "biological_characterization_manifest"
        ].read_text(
            encoding="utf-8"
        )
    )

    assert manifest["predictive_biomarker_claims_allowed"] is False
    assert manifest["external_validation_required"] is True