"""
HERMES 2.0
Locked NeoTRIP Benchmark Tests
==============================

Tests comparator construction, fair repeated-split behavior, and reporting
contracts. The tests do not require the generated primary output directory.
"""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pandas as pd

from backend.app.treatment_effects.feature_builder import TreatmentEffectDataset
from backend.app.treatment_effects.benchmark_analysis import (
    BenchmarkPlan,
    COMPARATOR_MODELS,
    HERMES_MODEL_NAME,
    LockedBenchmarkResult,
    _ite_concordance,
    _summarize_model,
    export_locked_benchmark,
    run_comparator_benchmark,
)


def _synthetic_dataset(seed: int = 7) -> TreatmentEffectDataset:
    rng = np.random.default_rng(seed)

    n = 120
    index = pd.Index(
        [f"P{i:03d}" for i in range(n)],
        name="Patient_ID",
    )

    X = pd.DataFrame(
        rng.normal(size=(n, 6)),
        index=index,
        columns=[f"HALLMARK_{i}" for i in range(6)],
    )

    T = pd.Series(
        np.tile([0, 1], n // 2),
        index=index,
        name="T",
        dtype=int,
    )

    subtype_values = np.resize(
        np.array(["BL1", "M", "IM", "LAR"]),
        n,
    )

    linear = (
        -0.2
        + 0.35 * T.to_numpy()
        + 0.30 * X["HALLMARK_0"].to_numpy()
        - 0.25 * X["HALLMARK_1"].to_numpy()
        + 0.30
        * T.to_numpy()
        * X["HALLMARK_2"].to_numpy()
    )
    probability = 1.0 / (1.0 + np.exp(-linear))
    Y = pd.Series(
        rng.binomial(1, probability),
        index=index,
        name="Y",
        dtype=int,
    )

    # Guarantee all four treatment × outcome strata for test stability.
    for treatment in (0, 1):
        ids = index[T.eq(treatment)]
        Y.loc[ids[:5]] = 0
        Y.loc[ids[5:10]] = 1

    metadata = pd.DataFrame(
        {
            "treatment_label": T.map({0: "CT", 1: "CT/A"}),
            "outcome_label": Y.map({0: "RD", 1: "pCR"}),
            "tnbc_type": subtype_values,
        },
        index=index,
    )

    dataset = TreatmentEffectDataset(
        X=X,
        T=T,
        Y=Y,
        metadata=metadata,
        summary={"patients": n, "features": X.shape[1]},
    )
    dataset.validate()
    return dataset


def _light_plan() -> BenchmarkPlan:
    return BenchmarkPlan(
        plan_name="benchmark_test",
        n_splits=3,
        regularization_C=0.10,
        max_iter=3000,
    )


def test_comparator_models_are_prespecified() -> None:
    assert COMPARATOR_MODELS == (
        "treatment_only",
        "hallmark_main_effects",
        "tnbc_subtype_interactions",
    )


def test_comparator_benchmark_runs_same_repeats_for_every_model() -> None:
    dataset = _synthetic_dataset()
    states = [11, 22, 33]

    result = run_comparator_benchmark(
        dataset,
        random_states=states,
        plan=_light_plan(),
    )

    assert set(result.ite_by_model) == set(COMPARATOR_MODELS)
    assert len(result.repeat_metrics) == len(COMPARATOR_MODELS) * len(states)

    for model_name in COMPARATOR_MODELS:
        matrix = result.ite_by_model[model_name]
        assert matrix.shape == (dataset.n_patients, len(states))
        assert np.isfinite(matrix.to_numpy(dtype=float)).all()

        subset = result.repeat_metrics.loc[
            result.repeat_metrics["model"].eq(model_name)
        ]
        assert subset["random_state"].tolist() == states


def test_treatment_only_has_no_patient_ranking_within_repeat() -> None:
    result = run_comparator_benchmark(
        _synthetic_dataset(),
        random_states=[11, 22, 33],
        plan=_light_plan(),
    )

    matrix = result.ite_by_model["treatment_only"]

    # Fold-specific models can yield a few constants across folds, but
    # treatment-only does not use biological patient features.
    assert all(
        matrix[column].nunique() <= _light_plan().n_splits
        for column in matrix.columns
    )


def test_model_summary_reports_predictive_and_heterogeneity_metrics() -> None:
    result = run_comparator_benchmark(
        _synthetic_dataset(),
        random_states=[11, 22, 33],
        plan=_light_plan(),
    )

    required = {
        "model",
        "mean_oof_auc",
        "mean_oof_brier",
        "mean_oof_log_loss",
        "mean_cohort_ite",
        "sd_patient_mean_ite",
        "mean_patient_repeat_sd",
        "mean_pairwise_ite_spearman",
    }

    assert required.issubset(result.model_summary.columns)
    assert len(result.model_summary) == 3


def test_ite_concordance_handles_constant_comparator() -> None:
    table = pd.DataFrame(
        {
            "Patient_ID": ["A", "B", "C", "D"],
            f"{HERMES_MODEL_NAME}__mean_ite": [0.1, 0.2, -0.1, 0.0],
            "treatment_only__mean_ite": [0.05, 0.05, 0.05, 0.05],
            "hallmark_main_effects__mean_ite": [0.08, 0.16, -0.05, 0.02],
        }
    )

    result = _ite_concordance(table)

    assert len(result) == 3

    constant_row = result.loc[
        (
            result["model_a"].eq(HERMES_MODEL_NAME)
            & result["model_b"].eq("treatment_only")
        )
        | (
            result["model_b"].eq(HERMES_MODEL_NAME)
            & result["model_a"].eq("treatment_only")
        )
    ].iloc[0]

    assert np.isnan(constant_row["spearman_patient_mean_ite"])


def test_locked_benchmark_export_contract(tmp_path) -> None:
    dataset = _synthetic_dataset()
    comparator = run_comparator_benchmark(
        dataset,
        random_states=[11, 22, 33],
        plan=_light_plan(),
    )

    hermes_ite = comparator.ite_by_model["hallmark_main_effects"].copy()
    hermes_ite = hermes_ite + np.linspace(
        -0.02,
        0.02,
        dataset.n_patients,
    ).reshape(-1, 1)

    hermes_repeat = comparator.repeat_metrics.loc[
        comparator.repeat_metrics["model"].eq("hallmark_main_effects")
    ].copy()
    hermes_repeat["model"] = HERMES_MODEL_NAME

    hermes_summary = pd.DataFrame(
        [
            _summarize_model(
                model_name=HERMES_MODEL_NAME,
                repeat_metrics=hermes_repeat,
                ite_by_repeat=hermes_ite,
            )
        ]
    )

    combined = pd.concat(
        [comparator.model_summary, hermes_summary],
        ignore_index=True,
    )
    baseline_auc = float(
        combined.loc[
            combined["model"].eq("treatment_only"),
            "mean_oof_auc",
        ].iloc[0]
    )
    baseline_brier = float(
        combined.loc[
            combined["model"].eq("treatment_only"),
            "mean_oof_brier",
        ].iloc[0]
    )
    combined["delta_auc_vs_treatment_only"] = (
        combined["mean_oof_auc"] - baseline_auc
    )
    combined["delta_brier_vs_treatment_only"] = (
        combined["mean_oof_brier"] - baseline_brier
    )

    patient = pd.DataFrame(
        {
            "Patient_ID": dataset.X.index.astype(str),
            "tnbc_type": dataset.metadata["tnbc_type"].astype(str).to_numpy(),
            "treatment": dataset.T.to_numpy(dtype=int),
            "outcome": dataset.Y.to_numpy(dtype=int),
            f"{HERMES_MODEL_NAME}__mean_ite": hermes_ite.mean(axis=1).to_numpy(),
            f"{HERMES_MODEL_NAME}__ite_repeat_sd": hermes_ite.std(
                axis=1, ddof=1
            ).to_numpy(),
        }
    )
    for model, matrix in comparator.ite_by_model.items():
        patient[f"{model}__mean_ite"] = matrix.mean(axis=1).to_numpy()
        patient[f"{model}__ite_repeat_sd"] = matrix.std(
            axis=1, ddof=1
        ).to_numpy()

    result = LockedBenchmarkResult(
        plan=_light_plan(),
        source_manifest={
            "plan_name": "hermes2_neotrip_primary_locked_v1",
            "plan_sha256": "a" * 64,
            "engine_tag": "hermes-2.0-engine-v1.0",
        },
        source_primary_plan={"n_splits": 3},
        comparator=comparator,
        hermes_repeat_metrics=hermes_repeat,
        hermes_ite_by_repeat=hermes_ite,
        combined_model_summary=combined,
        patient_ite_comparison=patient,
        ite_concordance=_ite_concordance(patient),
    )

    generated = export_locked_benchmark(
        result,
        output_dir=tmp_path / "benchmark",
    )

    required = {
        "table__model_summary",
        "table__comparator_repeat_metrics",
        "table__hermes_repeat_metrics",
        "table__patient_ite_comparison",
        "table__ite_concordance",
        "figure_1_benchmark_auc",
        "figure_2_benchmark_brier",
        "figure_3_heterogeneity_comparison",
        "figure_4_hermes_vs_main_effects",
        "benchmark_manifest",
    }

    assert required.issubset(generated)

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0

    manifest = json.loads(
        generated["benchmark_manifest"].read_text(encoding="utf-8")
    )
    assert manifest["read_only_reference_to_locked_hermes"] is True