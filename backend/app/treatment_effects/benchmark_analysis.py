"""
HERMES 2.0
Locked NeoTRIP Benchmark Analysis
=================================

Purpose
-------
Benchmark the frozen HERMES 2.0 Hallmark interaction model against simpler,
prespecified comparators using the SAME NeoTRIP patients, SAME joint
treatment-outcome stratification, and SAME repeated cross-fit random states
used by the locked primary HERMES analysis.

This module does not modify or refit the locked primary HERMES result. It
loads the frozen HERMES repeated-cross-fit outputs and fits only the simpler
comparators.

Comparators
-----------
1. treatment_only
       Logistic outcome model with randomized treatment only.
       ITE is constant across patients within each fitted fold/model.

2. hallmark_main_effects
       Treatment + 50 Hallmark pathway main effects, with no explicit
       treatment x pathway interactions.
       Patient-to-patient absolute risk differences can still vary because
       logistic probabilities are nonlinear, but the model contains no
       pathway-specific predictive interaction terms.

3. tnbc_subtype_interactions
       Treatment + TNBC subtype + treatment x TNBC subtype interactions.
       This is a conventional low-dimensional clinical/biological subgroup
       comparator.

Reference model
---------------
4. HERMES Hallmark interactions
       The already-locked primary result:
           treatment + Hallmark main effects + treatment x Hallmark
           interactions, estimated with repeated cross-fitting.

Interpretation
--------------
Predictive AUC/Brier/log-loss benchmark observed outcomes; they do NOT prove
individual treatment-effect accuracy. Heterogeneity/stability comparisons are
descriptive because individual causal effects are not observed.

No model is selected or tuned on the basis of these benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from backend.app.treatment_effects.crossfit import build_joint_strata
from backend.app.treatment_effects.feature_builder import (
    TreatmentEffectDataset,
    build_treatment_effect_dataset,
)


BenchmarkModel = Literal[
    "treatment_only",
    "hallmark_main_effects",
    "tnbc_subtype_interactions",
]

COMPARATOR_MODELS: tuple[BenchmarkModel, ...] = (
    "treatment_only",
    "hallmark_main_effects",
    "tnbc_subtype_interactions",
)

HERMES_MODEL_NAME = "hermes_hallmark_interactions"

DEFAULT_PRIMARY_RESULTS_DIR = Path("outputs/hermes2/primary_neotrip")
DEFAULT_BENCHMARK_DIR = Path("outputs/hermes2/benchmark")


@dataclass(frozen=True)
class BenchmarkPlan:
    """Prespecified locked benchmark configuration."""

    plan_name: str = "hermes2_neotrip_benchmark_v1"
    source_primary_plan: str = "hermes2_neotrip_primary_locked_v1"
    source_engine_tag: str = "hermes-2.0-engine-v1.0"

    n_splits: int = 5
    regularization_C: float = 0.10
    max_iter: int = 10000

    comparator_models: tuple[str, ...] = COMPARATOR_MODELS

    analysis_scope: str = "research_internal_validation"
    external_validation_required: bool = True


LOCKED_BENCHMARK_PLAN = BenchmarkPlan()


@dataclass
class ComparatorBenchmarkResult:
    """Repeated cross-fit outputs for the simpler benchmark models."""

    repeat_metrics: pd.DataFrame
    ite_by_model: dict[str, pd.DataFrame]
    patient_summary: pd.DataFrame
    model_summary: pd.DataFrame


@dataclass
class LockedBenchmarkResult:
    """Complete HERMES-vs-comparator benchmark package."""

    plan: BenchmarkPlan
    source_manifest: dict[str, Any]
    source_primary_plan: dict[str, Any]
    comparator: ComparatorBenchmarkResult
    hermes_repeat_metrics: pd.DataFrame
    hermes_ite_by_repeat: pd.DataFrame
    combined_model_summary: pd.DataFrame
    patient_ite_comparison: pd.DataFrame
    ite_concordance: pd.DataFrame


def _validate_dataset(dataset: TreatmentEffectDataset) -> None:
    dataset.validate()

    if "tnbc_type" not in dataset.metadata.columns:
        raise ValueError("Benchmark requires metadata['tnbc_type'].")

    if dataset.metadata["tnbc_type"].isna().any():
        raise ValueError("TNBC subtype contains missing values.")

    if not dataset.metadata.index.equals(dataset.X.index):
        raise ValueError("Metadata and Hallmark matrix are not aligned.")


def _subtype_design_matrix(dataset: TreatmentEffectDataset) -> pd.DataFrame:
    """Create a deterministic K-1 one-hot TNBC subtype representation."""

    subtype = dataset.metadata["tnbc_type"].astype(str)

    categories = sorted(subtype.unique().tolist())
    if len(categories) < 2:
        raise ValueError("At least two TNBC subtypes are required.")

    categorical = pd.Categorical(
        subtype,
        categories=categories,
        ordered=False,
    )

    dummies = pd.get_dummies(
        categorical,
        prefix="TNBC",
        drop_first=True,
        dtype=float,
    )

    dummies.index = dataset.X.index
    return dummies


def _design_treatment_only(
    treatment: np.ndarray,
) -> np.ndarray:
    return np.asarray(treatment, dtype=float).reshape(-1, 1)


def _design_main_effects(
    X_scaled: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    treatment = np.asarray(treatment, dtype=float).reshape(-1, 1)
    return np.column_stack([treatment, X_scaled])


def _design_subtype_interactions(
    subtype_matrix: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    treatment = np.asarray(treatment, dtype=float).reshape(-1, 1)
    interactions = subtype_matrix * treatment
    return np.column_stack(
        [
            treatment,
            subtype_matrix,
            interactions,
        ]
    )


def _build_designs(
    model_name: BenchmarkModel,
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    subtype_train: pd.DataFrame,
    subtype_test: pd.DataFrame,
    T_train: pd.Series,
    T_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return train, observed-test, control-test, treated-test design matrices.

    Fold-specific Hallmark scaling is fitted on training patients only.
    """

    if model_name == "treatment_only":
        train = _design_treatment_only(T_train.to_numpy(dtype=int))
        observed = _design_treatment_only(T_test.to_numpy(dtype=int))
        control = _design_treatment_only(
            np.zeros(len(T_test), dtype=int)
        )
        treated = _design_treatment_only(
            np.ones(len(T_test), dtype=int)
        )
        return train, observed, control, treated

    if model_name == "hallmark_main_effects":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train.to_numpy(dtype=float)
        )
        X_test_scaled = scaler.transform(
            X_test.to_numpy(dtype=float)
        )

        train = _design_main_effects(
            X_train_scaled,
            T_train.to_numpy(dtype=int),
        )
        observed = _design_main_effects(
            X_test_scaled,
            T_test.to_numpy(dtype=int),
        )
        control = _design_main_effects(
            X_test_scaled,
            np.zeros(len(T_test), dtype=int),
        )
        treated = _design_main_effects(
            X_test_scaled,
            np.ones(len(T_test), dtype=int),
        )
        return train, observed, control, treated

    if model_name == "tnbc_subtype_interactions":
        train_subtype = subtype_train.to_numpy(dtype=float)
        test_subtype = subtype_test.to_numpy(dtype=float)

        train = _design_subtype_interactions(
            train_subtype,
            T_train.to_numpy(dtype=int),
        )
        observed = _design_subtype_interactions(
            test_subtype,
            T_test.to_numpy(dtype=int),
        )
        control = _design_subtype_interactions(
            test_subtype,
            np.zeros(len(T_test), dtype=int),
        )
        treated = _design_subtype_interactions(
            test_subtype,
            np.ones(len(T_test), dtype=int),
        )
        return train, observed, control, treated

    raise ValueError(f"Unsupported benchmark model: {model_name}")


def _run_one_repeat(
    dataset: TreatmentEffectDataset,
    *,
    model_name: BenchmarkModel,
    n_splits: int,
    C: float,
    max_iter: int,
    random_state: int,
    subtype_matrix: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Run one fair repeated-cross-fit comparator replicate."""

    strata = build_joint_strata(dataset.T, dataset.Y)

    splitter = StratifiedKFold(
        n_splits=int(n_splits),
        shuffle=True,
        random_state=int(random_state),
    )

    observed_probability = pd.Series(
        index=dataset.X.index,
        dtype=float,
        name="observed_probability",
    )
    ite = pd.Series(
        index=dataset.X.index,
        dtype=float,
        name="ite",
    )

    for train_idx, test_idx in splitter.split(dataset.X, strata):
        train_ids = dataset.X.index[train_idx]
        test_ids = dataset.X.index[test_idx]

        X_train = dataset.X.loc[train_ids]
        X_test = dataset.X.loc[test_ids]

        subtype_train = subtype_matrix.loc[train_ids]
        subtype_test = subtype_matrix.loc[test_ids]

        T_train = dataset.T.loc[train_ids]
        T_test = dataset.T.loc[test_ids]
        Y_train = dataset.Y.loc[train_ids]

        train_design, observed_design, control_design, treated_design = (
            _build_designs(
                model_name,
                X_train=X_train,
                X_test=X_test,
                subtype_train=subtype_train,
                subtype_test=subtype_test,
                T_train=T_train,
                T_test=T_test,
            )
        )

        model = LogisticRegression(
            C=float(C),
            solver="lbfgs",
            max_iter=int(max_iter),
        )
        model.fit(
            train_design,
            Y_train.to_numpy(dtype=int),
        )

        observed_probability.loc[test_ids] = model.predict_proba(
            observed_design
        )[:, 1]

        mu0 = model.predict_proba(control_design)[:, 1]
        mu1 = model.predict_proba(treated_design)[:, 1]
        ite.loc[test_ids] = mu1 - mu0

    if observed_probability.isna().any() or ite.isna().any():
        raise RuntimeError(
            f"{model_name} failed to produce complete out-of-fold estimates."
        )

    if not np.isfinite(observed_probability.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{model_name} produced non-finite probabilities.")

    if not np.isfinite(ite.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{model_name} produced non-finite ITE estimates.")

    return observed_probability, ite


def _mean_pairwise_spearman(ite_by_repeat: pd.DataFrame) -> float:
    """Mean upper-triangle Spearman correlation; NaN for constant rankings."""

    corr = ite_by_repeat.corr(method="spearman").to_numpy(dtype=float)
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]

    if len(upper) == 0:
        return float("nan")

    return float(upper.mean())


def _summarize_model(
    *,
    model_name: str,
    repeat_metrics: pd.DataFrame,
    ite_by_repeat: pd.DataFrame,
) -> dict[str, Any]:
    patient_mean = ite_by_repeat.mean(axis=1)
    patient_sd = ite_by_repeat.std(axis=1, ddof=1)

    return {
        "model": model_name,
        "n_repeats": int(ite_by_repeat.shape[1]),
        "mean_oof_auc": float(repeat_metrics["oof_auc"].mean()),
        "sd_oof_auc": float(repeat_metrics["oof_auc"].std(ddof=1)),
        "mean_oof_brier": float(repeat_metrics["oof_brier"].mean()),
        "sd_oof_brier": float(repeat_metrics["oof_brier"].std(ddof=1)),
        "mean_oof_log_loss": float(
            repeat_metrics["oof_log_loss"].mean()
        ),
        "sd_oof_log_loss": float(
            repeat_metrics["oof_log_loss"].std(ddof=1)
        ),
        "mean_cohort_ite": float(
            repeat_metrics["mean_ite"].mean()
        ),
        "sd_cohort_ite_across_repeats": float(
            repeat_metrics["mean_ite"].std(ddof=1)
        ),
        "sd_patient_mean_ite": float(patient_mean.std(ddof=1)),
        "mean_patient_repeat_sd": float(patient_sd.mean()),
        "mean_pairwise_ite_spearman": _mean_pairwise_spearman(
            ite_by_repeat
        ),
        "fraction_patient_mean_ite_positive": float(
            (patient_mean > 0).mean()
        ),
        "fraction_patient_mean_ite_negative": float(
            (patient_mean < 0).mean()
        ),
    }


def run_comparator_benchmark(
    dataset: TreatmentEffectDataset,
    *,
    random_states: Iterable[int],
    plan: BenchmarkPlan = LOCKED_BENCHMARK_PLAN,
) -> ComparatorBenchmarkResult:
    """
    Run simpler comparator models with exactly the supplied split seeds.

    Supplying the locked HERMES random states guarantees paired repeated
    cross-fitting partitions across models.
    """

    _validate_dataset(dataset)

    random_states = [int(x) for x in random_states]
    if len(random_states) < 2:
        raise ValueError("Benchmark requires at least two repeated splits.")
    if len(set(random_states)) != len(random_states):
        raise ValueError("Benchmark random states must be unique.")

    subtype_matrix = _subtype_design_matrix(dataset)

    all_repeat_metrics: list[dict[str, Any]] = []
    ite_by_model: dict[str, pd.DataFrame] = {}

    for model_name in plan.comparator_models:
        repeat_ites = pd.DataFrame(index=dataset.X.index)

        for repeat_index, random_state in enumerate(random_states, start=1):
            observed_probability, ite = _run_one_repeat(
                dataset,
                model_name=model_name,  # type: ignore[arg-type]
                n_splits=plan.n_splits,
                C=plan.regularization_C,
                max_iter=plan.max_iter,
                random_state=random_state,
                subtype_matrix=subtype_matrix,
            )

            repeat_name = f"repeat_{repeat_index:03d}"
            repeat_ites[repeat_name] = ite

            all_repeat_metrics.append(
                {
                    "model": model_name,
                    "repeat": repeat_index,
                    "random_state": random_state,
                    "oof_auc": float(
                        roc_auc_score(
                            dataset.Y.to_numpy(dtype=int),
                            observed_probability.to_numpy(dtype=float),
                        )
                    ),
                    "oof_brier": float(
                        brier_score_loss(
                            dataset.Y.to_numpy(dtype=int),
                            observed_probability.to_numpy(dtype=float),
                        )
                    ),
                    "oof_log_loss": float(
                        log_loss(
                            dataset.Y.to_numpy(dtype=int),
                            observed_probability.to_numpy(dtype=float),
                            labels=[0, 1],
                        )
                    ),
                    "mean_ite": float(ite.mean()),
                    "median_ite": float(ite.median()),
                    "ite_sd": float(ite.std(ddof=1)),
                    "fraction_positive": float((ite > 0).mean()),
                    "fraction_negative": float((ite < 0).mean()),
                }
            )

        ite_by_model[model_name] = repeat_ites

    repeat_metrics = pd.DataFrame(all_repeat_metrics)

    model_summary_rows = []
    patient_frames = []

    for model_name, ite_matrix in ite_by_model.items():
        model_repeat_metrics = repeat_metrics.loc[
            repeat_metrics["model"].eq(model_name)
        ].copy()

        model_summary_rows.append(
            _summarize_model(
                model_name=model_name,
                repeat_metrics=model_repeat_metrics,
                ite_by_repeat=ite_matrix,
            )
        )

        patient_frames.append(
            pd.DataFrame(
                {
                    "Patient_ID": dataset.X.index.astype(str),
                    "model": model_name,
                    "mean_ite": ite_matrix.mean(axis=1).to_numpy(dtype=float),
                    "median_ite": ite_matrix.median(axis=1).to_numpy(dtype=float),
                    "ite_repeat_sd": ite_matrix.std(
                        axis=1,
                        ddof=1,
                    ).to_numpy(dtype=float),
                    "fraction_positive": (
                        ite_matrix.gt(0).mean(axis=1).to_numpy(dtype=float)
                    ),
                }
            )
        )

    return ComparatorBenchmarkResult(
        repeat_metrics=repeat_metrics,
        ite_by_model=ite_by_model,
        patient_summary=pd.concat(
            patient_frames,
            ignore_index=True,
        ),
        model_summary=pd.DataFrame(model_summary_rows),
    )


def _load_locked_primary(
    results_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
]:
    manifest = json.loads(
        (results_dir / "analysis_manifest.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (results_dir / "primary_analysis_plan.json").read_text(encoding="utf-8")
    )

    repeat_summary = pd.read_csv(
        results_dir / "engine_outputs" / "repeat_summary.csv"
    )
    ite = pd.read_csv(
        results_dir / "engine_outputs" / "ite_by_repeat.csv"
    )

    if "Patient_ID" not in ite.columns:
        raise ValueError("Locked HERMES ITE matrix lacks Patient_ID.")

    ite = ite.set_index("Patient_ID")

    return manifest, plan, repeat_summary, ite


def _hermes_repeat_metrics(
    repeat_summary: pd.DataFrame,
) -> pd.DataFrame:
    table = repeat_summary.copy()
    table.insert(0, "model", HERMES_MODEL_NAME)

    # Locked HERMES did not export repeat-level log loss.
    table["oof_log_loss"] = np.nan

    columns = [
        "model",
        "repeat",
        "random_state",
        "oof_auc",
        "oof_brier",
        "oof_log_loss",
        "mean_ite",
        "median_ite",
        "ite_std",
        "fraction_positive",
        "fraction_negative",
    ]
    return table[columns]


def _combine_model_summaries(
    comparator: ComparatorBenchmarkResult,
    hermes_repeat: pd.DataFrame,
    hermes_ite: pd.DataFrame,
) -> pd.DataFrame:
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
        [
            comparator.model_summary,
            hermes_summary,
        ],
        ignore_index=True,
    )

    treatment_only_auc = float(
        combined.loc[
            combined["model"].eq("treatment_only"),
            "mean_oof_auc",
        ].iloc[0]
    )
    treatment_only_brier = float(
        combined.loc[
            combined["model"].eq("treatment_only"),
            "mean_oof_brier",
        ].iloc[0]
    )

    combined["delta_auc_vs_treatment_only"] = (
        combined["mean_oof_auc"] - treatment_only_auc
    )
    combined["delta_brier_vs_treatment_only"] = (
        combined["mean_oof_brier"] - treatment_only_brier
    )

    return combined


def _patient_ite_comparison(
    dataset: TreatmentEffectDataset,
    comparator: ComparatorBenchmarkResult,
    hermes_ite: pd.DataFrame,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Patient_ID": dataset.X.index.astype(str),
            "tnbc_type": dataset.metadata["tnbc_type"].astype(str).to_numpy(),
            "treatment": dataset.T.to_numpy(dtype=int),
            "outcome": dataset.Y.to_numpy(dtype=int),
            f"{HERMES_MODEL_NAME}__mean_ite": hermes_ite.mean(
                axis=1
            ).to_numpy(dtype=float),
            f"{HERMES_MODEL_NAME}__ite_repeat_sd": hermes_ite.std(
                axis=1,
                ddof=1,
            ).to_numpy(dtype=float),
        }
    )

    for model_name, ite_matrix in comparator.ite_by_model.items():
        table[f"{model_name}__mean_ite"] = ite_matrix.mean(
            axis=1
        ).to_numpy(dtype=float)
        table[f"{model_name}__ite_repeat_sd"] = ite_matrix.std(
            axis=1,
            ddof=1,
        ).to_numpy(dtype=float)

    return table


def _ite_concordance(
    patient_comparison: pd.DataFrame,
) -> pd.DataFrame:
    ite_columns = [
        column
        for column in patient_comparison.columns
        if column.endswith("__mean_ite")
    ]

    rows: list[dict[str, Any]] = []
    for i, first in enumerate(ite_columns):
        for second in ite_columns[i + 1 :]:
            x = patient_comparison[first]
            y = patient_comparison[second]

            if x.nunique() <= 1 or y.nunique() <= 1:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(x.corr(y, method="pearson"))
                spearman = float(x.corr(y, method="spearman"))

            rows.append(
                {
                    "model_a": first.replace("__mean_ite", ""),
                    "model_b": second.replace("__mean_ite", ""),
                    "pearson_patient_mean_ite": pearson,
                    "spearman_patient_mean_ite": spearman,
                    "mean_absolute_ite_difference": float(
                        np.abs(x.to_numpy() - y.to_numpy()).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


def run_locked_neotrip_benchmark(
    *,
    primary_results_dir: str | Path = DEFAULT_PRIMARY_RESULTS_DIR,
    plan: BenchmarkPlan = LOCKED_BENCHMARK_PLAN,
) -> LockedBenchmarkResult:
    """Run the prespecified benchmark against the frozen primary HERMES run."""

    primary_results_dir = Path(primary_results_dir)

    manifest, primary_plan, hermes_repeat_raw, hermes_ite = (
        _load_locked_primary(primary_results_dir)
    )

    if manifest["plan_name"] != plan.source_primary_plan:
        raise ValueError(
            "Benchmark source plan does not match the locked primary plan."
        )

    if manifest["engine_tag"] != plan.source_engine_tag:
        raise ValueError(
            "Benchmark source engine does not match the locked engine tag."
        )

    if int(primary_plan["n_splits"]) != int(plan.n_splits):
        raise ValueError(
            "Benchmark n_splits must match the locked primary analysis."
        )

    random_states = hermes_repeat_raw["random_state"].astype(int).tolist()

    dataset = build_treatment_effect_dataset(
        min_genes=3,
        min_coverage=0.50,
    )
    _validate_dataset(dataset)

    if not dataset.X.index.astype(str).equals(
        pd.Index(hermes_ite.index.astype(str), name=hermes_ite.index.name)
    ):
        raise ValueError(
            "Locked HERMES ITE rows do not align with canonical NeoTRIP."
        )

    comparator = run_comparator_benchmark(
        dataset,
        random_states=random_states,
        plan=plan,
    )

    hermes_repeat = _hermes_repeat_metrics(hermes_repeat_raw)

    combined_summary = _combine_model_summaries(
        comparator,
        hermes_repeat,
        hermes_ite,
    )

    patient_comparison = _patient_ite_comparison(
        dataset,
        comparator,
        hermes_ite,
    )

    concordance = _ite_concordance(patient_comparison)

    return LockedBenchmarkResult(
        plan=plan,
        source_manifest=manifest,
        source_primary_plan=primary_plan,
        comparator=comparator,
        hermes_repeat_metrics=hermes_repeat,
        hermes_ite_by_repeat=hermes_ite,
        combined_model_summary=combined_summary,
        patient_ite_comparison=patient_comparison,
        ite_concordance=concordance,
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_auc(
    result: LockedBenchmarkResult,
    path: Path,
) -> None:
    """Conference figure: repeated-split OOF AUC by model."""

    combined = pd.concat(
        [
            result.comparator.repeat_metrics,
            result.hermes_repeat_metrics,
        ],
        ignore_index=True,
    )

    model_order = [
        "treatment_only",
        "hallmark_main_effects",
        "tnbc_subtype_interactions",
        HERMES_MODEL_NAME,
    ]

    data = [
        combined.loc[
            combined["model"].eq(model),
            "oof_auc",
        ].dropna().to_numpy(dtype=float)
        for model in model_order
    ]

    labels = [
        "Treatment only",
        "Hallmark main effects",
        "TNBC subtype interactions",
        "HERMES Hallmark interactions",
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Out-of-fold AUC")
    ax.set_title(
        "Observed-outcome discrimination across identical repeated splits"
    )
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, path)


def plot_benchmark_brier(
    result: LockedBenchmarkResult,
    path: Path,
) -> None:
    """Conference figure: repeated-split Brier score by model."""

    combined = pd.concat(
        [
            result.comparator.repeat_metrics,
            result.hermes_repeat_metrics,
        ],
        ignore_index=True,
    )

    model_order = [
        "treatment_only",
        "hallmark_main_effects",
        "tnbc_subtype_interactions",
        HERMES_MODEL_NAME,
    ]

    data = [
        combined.loc[
            combined["model"].eq(model),
            "oof_brier",
        ].dropna().to_numpy(dtype=float)
        for model in model_order
    ]

    labels = [
        "Treatment only",
        "Hallmark main effects",
        "TNBC subtype interactions",
        "HERMES Hallmark interactions",
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Out-of-fold Brier score (lower is better)")
    ax.set_title(
        "Observed-outcome probability error across identical repeated splits"
    )
    ax.tick_params(axis="x", rotation=20)
    _save_figure(fig, path)


def plot_heterogeneity_comparison(
    result: LockedBenchmarkResult,
    path: Path,
) -> None:
    """Conference figure: descriptive magnitude of model-implied heterogeneity."""

    table = result.combined_model_summary.copy()

    model_order = [
        "treatment_only",
        "hallmark_main_effects",
        "tnbc_subtype_interactions",
        HERMES_MODEL_NAME,
    ]
    table = table.set_index("model").loc[model_order].reset_index()

    labels = [
        "Treatment only",
        "Hallmark main effects",
        "TNBC subtype interactions",
        "HERMES Hallmark interactions",
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.bar(
        np.arange(len(table)),
        table["sd_patient_mean_ite"],
    )
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("SD of patient mean estimated ITE")
    ax.set_title(
        "Model-implied treatment-effect heterogeneity\n"
        "(descriptive; individual causal effects are unobserved)"
    )
    _save_figure(fig, path)


def plot_hermes_vs_main_effects(
    result: LockedBenchmarkResult,
    path: Path,
) -> None:
    """Ablation figure: HERMES ITE vs no-interaction Hallmark model."""

    table = result.patient_ite_comparison

    x = table["hallmark_main_effects__mean_ite"]
    y = table[f"{HERMES_MODEL_NAME}__mean_ite"]

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(x, y, s=24, alpha=0.75)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Hallmark main-effects model: mean ITE")
    ax.set_ylabel("HERMES interaction model: mean ITE")
    ax.set_title(
        "Effect of explicit treatment × biology interactions"
    )

    if x.nunique() > 1 and y.nunique() > 1:
        rho = x.corr(y, method="spearman")
        ax.text(
            0.03,
            0.97,
            f"Spearman ρ={rho:.3f}",
            transform=ax.transAxes,
            va="top",
        )

    _save_figure(fig, path)


def export_locked_benchmark(
    result: LockedBenchmarkResult,
    output_dir: str | Path = DEFAULT_BENCHMARK_DIR,
) -> dict[str, Path]:
    """Export tables, patient comparisons, and conference benchmark figures."""

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}

    tables = {
        "model_summary": result.combined_model_summary,
        "comparator_repeat_metrics": result.comparator.repeat_metrics,
        "hermes_repeat_metrics": result.hermes_repeat_metrics,
        "patient_ite_comparison": result.patient_ite_comparison,
        "ite_concordance": result.ite_concordance,
    }

    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        generated[f"table__{name}"] = path

    for model_name, ite_matrix in result.comparator.ite_by_model.items():
        path = tables_dir / f"{model_name}__ite_by_repeat.csv"
        ite_matrix.to_csv(path)
        generated[f"table__{model_name}__ite_by_repeat"] = path

    figure_functions = {
        "figure_1_benchmark_auc": plot_benchmark_auc,
        "figure_2_benchmark_brier": plot_benchmark_brier,
        "figure_3_heterogeneity_comparison": plot_heterogeneity_comparison,
        "figure_4_hermes_vs_main_effects": plot_hermes_vs_main_effects,
    }

    for name, function in figure_functions.items():
        path = figures_dir / f"{name}.png"
        function(result, path)
        generated[name] = path

    manifest = {
        "benchmark_plan": {
            "plan_name": result.plan.plan_name,
            "source_primary_plan": result.plan.source_primary_plan,
            "source_engine_tag": result.plan.source_engine_tag,
            "n_splits": result.plan.n_splits,
            "regularization_C": result.plan.regularization_C,
            "comparator_models": list(result.plan.comparator_models),
            "analysis_scope": result.plan.analysis_scope,
            "external_validation_required": (
                result.plan.external_validation_required
            ),
        },
        "source_plan_sha256": result.source_manifest["plan_sha256"],
        "source_repeats": int(
            result.hermes_ite_by_repeat.shape[1]
        ),
        "patients": int(
            result.hermes_ite_by_repeat.shape[0]
        ),
        "read_only_reference_to_locked_hermes": True,
        "interpretation": (
            "Observed-outcome predictive metrics and model-implied ITE "
            "heterogeneity are benchmarking descriptors. They do not establish "
            "accuracy of unobserved individual causal effects."
        ),
        "generated_files": {
            key: str(path)
            for key, path in sorted(generated.items())
        },
    }

    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    generated["benchmark_manifest"] = manifest_path

    return generated


def summarize_locked_benchmark(
    result: LockedBenchmarkResult,
) -> None:
    print("=== HERMES 2.0 LOCKED NEOTRIP BENCHMARK ===")
    print()
    print(f"Benchmark plan: {result.plan.plan_name}")
    print(f"Source HERMES plan: {result.source_manifest['plan_name']}")
    print(f"Source SHA256: {result.source_manifest['plan_sha256']}")
    print(
        f"Patients: {result.hermes_ite_by_repeat.shape[0]} | "
        f"Repeated splits: {result.hermes_ite_by_repeat.shape[1]}"
    )
    print()
    print(
        result.combined_model_summary[
            [
                "model",
                "mean_oof_auc",
                "mean_oof_brier",
                "mean_cohort_ite",
                "sd_patient_mean_ite",
                "mean_patient_repeat_sd",
                "mean_pairwise_ite_spearman",
                "delta_auc_vs_treatment_only",
                "delta_brier_vs_treatment_only",
            ]
        ].to_string(index=False)
    )
    print()
    print("Patient-level mean-ITE concordance:")
    print(result.ite_concordance.to_string(index=False))
    print()
    print("IMPORTANT:")
    print(
        "AUC/Brier compare observed-outcome prediction, not individual "
        "treatment-effect truth."
    )
    print(
        "ITE dispersion and concordance are descriptive because individual "
        "causal effects are unobserved."
    )
    print(
        "The locked HERMES primary result is used as a frozen reference and "
        "is not refit or tuned by this benchmark."
    )


def main() -> None:
    result = run_locked_neotrip_benchmark()
    summarize_locked_benchmark(result)

    generated = export_locked_benchmark(result)

    print()
    print(f"Benchmark written to: {DEFAULT_BENCHMARK_DIR}")
    print(f"Generated files: {len(generated)}")


if __name__ == "__main__":
    main()