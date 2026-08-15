"""
HERMES 2.0
Generalizability, Applicability, and Out-of-Distribution Assessment
===================================================================

Purpose
-------
Quantify whether a patient or external cohort is biologically similar
enough to the HERMES reference population for model outputs to be
interpreted with appropriate caution.

This module deliberately separates:

    prediction
        -> what HERMES estimates

from:

    uncertainty
        -> how variable the estimate is across model/resampling fits

from:

    applicability / OOD assessment
        -> how similar the biological input is to the population used
           to develop HERMES

and from:

    external validation
        -> whether predictions actually perform correctly in an
           independent cohort

Applicability is evaluated using biological features only.

Treatment assignments and outcomes are NOT used to determine whether
a patient is in-distribution.

Current feature space
---------------------
HERMES 2.0 currently operates on pathway-level biological
representations such as MSigDB Hallmark pathway scores.

IMPORTANT
---------
The current NeoTRIP pathway representation is constructed using
cohort-level gene standardization upstream. Therefore, true external
cohort deployment will require a reference-fitted representation
transform so external samples are mapped using training-cohort
parameters rather than independently standardized.

This module is designed so that external pathway matrices can be
assessed once that representation step is available.

Applicability categories generated here are research outputs.

They do NOT establish:
    - causal validity
    - treatment recommendation safety
    - predictive biomarker validation
    - clinical utility
    - successful external validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import percentileofscore
from sklearn.covariance import LedoitWolf


# =============================================================
# Result containers
# =============================================================


@dataclass
class ApplicabilityReference:
    """
    Reference biological distribution used for HERMES applicability
    assessment.
    """

    feature_names: tuple[str, ...]

    mean: pd.Series
    standard_deviation: pd.Series

    covariance: pd.DataFrame
    precision: pd.DataFrame

    reference_mahalanobis: pd.Series
    reference_max_abs_z: pd.Series
    reference_mean_abs_z: pd.Series

    mahalanobis_borderline_threshold: float
    mahalanobis_ood_threshold: float

    max_abs_z_borderline_threshold: float
    max_abs_z_ood_threshold: float

    mean_abs_z_borderline_threshold: float
    mean_abs_z_ood_threshold: float

    summary: dict[str, Any]


@dataclass
class ApplicabilityAssessment:
    """
    Patient-level applicability results.
    """

    patient_table: pd.DataFrame
    feature_z_scores: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class CohortShiftAssessment:
    """
    Cohort-level biological shift results.
    """

    feature_shift_table: pd.DataFrame
    target_patient_assessment: ApplicabilityAssessment
    summary: dict[str, Any]


# =============================================================
# Validation
# =============================================================


def _validate_matrix(
    X: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """
    Validate a biological feature matrix.
    """

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if X.index.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate patient/sample IDs."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate feature names."
        )

    numeric = X.astype(
        float
    )

    if not np.isfinite(
        numeric.to_numpy()
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return numeric


def _validate_feature_compatibility(
    reference: ApplicabilityReference,
    X: pd.DataFrame,
) -> None:
    """
    Require exact feature compatibility with the fitted reference.
    """

    target_features = tuple(
        str(column)
        for column in X.columns
    )

    if target_features != reference.feature_names:
        reference_set = set(
            reference.feature_names
        )

        target_set = set(
            target_features
        )

        missing = sorted(
            reference_set
            - target_set
        )

        extra = sorted(
            target_set
            - reference_set
        )

        if missing or extra:
            raise ValueError(
                "Target feature set does not match the HERMES "
                "applicability reference. "
                f"Missing={missing[:10]}, extra={extra[:10]}"
            )

        raise ValueError(
            "Target feature order does not match the HERMES "
            "applicability reference."
        )


def _validate_quantiles(
    borderline_quantile: float,
    ood_quantile: float,
) -> None:
    """
    Validate empirical applicability thresholds.
    """

    if not (
        0.50
        < borderline_quantile
        < 1.0
    ):
        raise ValueError(
            "borderline_quantile must be between 0.50 and 1.0."
        )

    if not (
        borderline_quantile
        < ood_quantile
        < 1.0
    ):
        raise ValueError(
            "ood_quantile must be greater than borderline_quantile "
            "and less than 1.0."
        )


# =============================================================
# Core mathematical helpers
# =============================================================


def _safe_reference_sd(
    X: pd.DataFrame,
    *,
    minimum_sd: float = 1e-8,
) -> pd.Series:
    """
    Compute reference feature SDs with protection against zero variance.
    """

    sd = X.std(
        axis=0,
        ddof=1,
    ).astype(
        float
    )

    if (
        sd
        < minimum_sd
    ).any():

        problematic = (
            sd[
                sd
                < minimum_sd
            ]
            .index
            .tolist()
        )

        preview = ", ".join(
            str(x)
            for x in problematic[:10]
        )

        raise ValueError(
            "Reference matrix contains effectively constant features: "
            f"{preview}"
        )

    return sd


def standardized_feature_deviation(
    X: pd.DataFrame,
    mean: pd.Series,
    standard_deviation: pd.Series,
) -> pd.DataFrame:
    """
    Standardize target biological features against a fitted reference.
    """

    z = (
        X
        - mean
    ).divide(
        standard_deviation,
        axis=1,
    )

    if not np.isfinite(
        z.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Standardized biological deviations contain non-finite values."
        )

    return z


def mahalanobis_distance(
    X: pd.DataFrame,
    mean: pd.Series,
    precision: pd.DataFrame,
) -> pd.Series:
    """
    Compute squared-root Mahalanobis distance from the reference center.

    A Ledoit-Wolf shrinkage covariance is used when fitting the reference,
    which is substantially more stable than an unregularized covariance
    inverse when biological features are correlated.
    """

    centered = (
        X
        - mean
    ).to_numpy(
        dtype=float
    )

    precision_array = (
        precision
        .loc[
            X.columns,
            X.columns,
        ]
        .to_numpy(
            dtype=float
        )
    )

    squared = np.einsum(
        "ij,jk,ik->i",
        centered,
        precision_array,
        centered,
    )

    squared = np.maximum(
        squared,
        0.0,
    )

    return pd.Series(
        np.sqrt(
            squared
        ),
        index=X.index,
        name="mahalanobis_distance",
        dtype=float,
    )


def _empirical_percentile_rank(
    value: float,
    reference_values: pd.Series,
) -> float:
    """
    Percentile position of one score relative to the reference cohort.
    """

    percentile = percentileofscore(
        reference_values.to_numpy(
            dtype=float
        ),
        float(
            value
        ),
        kind="weak",
    )

    return float(
        percentile
        / 100.0
    )


def _empirical_percentile_series(
    values: pd.Series,
    reference_values: pd.Series,
) -> pd.Series:
    """
    Vectorized empirical percentile ranks.
    """

    return pd.Series(
        [
            _empirical_percentile_rank(
                value,
                reference_values,
            )
            for value in values
        ],
        index=values.index,
        dtype=float,
    )


# =============================================================
# Fit applicability reference
# =============================================================


def fit_applicability_reference(
    X_reference: pd.DataFrame,
    *,
    borderline_quantile: float = 0.95,
    ood_quantile: float = 0.99,
) -> ApplicabilityReference:
    """
    Fit the HERMES biological applicability reference.

    Thresholds are empirical quantiles of the reference population rather
    than arbitrary theoretical cutoffs.

    Parameters
    ----------
    X_reference:
        Biological feature matrix representing the HERMES development
        population.

    borderline_quantile:
        Reference quantile above which observations are considered
        biologically unusual.

    ood_quantile:
        More extreme reference quantile above which observations are
        considered out-of-distribution for research purposes.
    """

    X_reference = _validate_matrix(
        X_reference,
        name="X_reference",
    )

    _validate_quantiles(
        borderline_quantile,
        ood_quantile,
    )

    if X_reference.shape[
        0
    ] < 20:
        raise ValueError(
            "At least 20 reference patients are required."
        )

    mean = X_reference.mean(
        axis=0
    ).astype(
        float
    )

    standard_deviation = _safe_reference_sd(
        X_reference
    )

    covariance_estimator = LedoitWolf(
        assume_centered=False
    )

    covariance_estimator.fit(
        X_reference.to_numpy(
            dtype=float
        )
    )

    covariance = pd.DataFrame(
        covariance_estimator.covariance_,
        index=X_reference.columns,
        columns=X_reference.columns,
    )

    precision = pd.DataFrame(
        covariance_estimator.precision_,
        index=X_reference.columns,
        columns=X_reference.columns,
    )

    reference_z = standardized_feature_deviation(
        X_reference,
        mean,
        standard_deviation,
    )

    reference_mahalanobis = mahalanobis_distance(
        X_reference,
        mean,
        precision,
    )

    reference_max_abs_z = (
        reference_z
        .abs()
        .max(
            axis=1
        )
        .rename(
            "max_abs_z"
        )
    )

    reference_mean_abs_z = (
        reference_z
        .abs()
        .mean(
            axis=1
        )
        .rename(
            "mean_abs_z"
        )
    )

    mahalanobis_borderline_threshold = float(
        reference_mahalanobis.quantile(
            borderline_quantile
        )
    )

    mahalanobis_ood_threshold = float(
        reference_mahalanobis.quantile(
            ood_quantile
        )
    )

    max_abs_z_borderline_threshold = float(
        reference_max_abs_z.quantile(
            borderline_quantile
        )
    )

    max_abs_z_ood_threshold = float(
        reference_max_abs_z.quantile(
            ood_quantile
        )
    )

    mean_abs_z_borderline_threshold = float(
        reference_mean_abs_z.quantile(
            borderline_quantile
        )
    )

    mean_abs_z_ood_threshold = float(
        reference_mean_abs_z.quantile(
            ood_quantile
        )
    )

    summary: dict[
        str,
        Any,
    ] = {
        "reference_patients":
            int(
                X_reference.shape[
                    0
                ]
            ),

        "biological_features":
            int(
                X_reference.shape[
                    1
                ]
            ),

        "borderline_quantile":
            float(
                borderline_quantile
            ),

        "ood_quantile":
            float(
                ood_quantile
            ),

        "mahalanobis_borderline_threshold":
            mahalanobis_borderline_threshold,

        "mahalanobis_ood_threshold":
            mahalanobis_ood_threshold,

        "max_abs_z_borderline_threshold":
            max_abs_z_borderline_threshold,

        "max_abs_z_ood_threshold":
            max_abs_z_ood_threshold,

        "mean_abs_z_borderline_threshold":
            mean_abs_z_borderline_threshold,

        "mean_abs_z_ood_threshold":
            mean_abs_z_ood_threshold,

        "covariance_estimator":
            "LedoitWolf",
    }

    return ApplicabilityReference(
        feature_names=tuple(
            str(column)
            for column in X_reference.columns
        ),

        mean=mean,
        standard_deviation=standard_deviation,

        covariance=covariance,
        precision=precision,

        reference_mahalanobis=(
            reference_mahalanobis
        ),

        reference_max_abs_z=(
            reference_max_abs_z
        ),

        reference_mean_abs_z=(
            reference_mean_abs_z
        ),

        mahalanobis_borderline_threshold=(
            mahalanobis_borderline_threshold
        ),

        mahalanobis_ood_threshold=(
            mahalanobis_ood_threshold
        ),

        max_abs_z_borderline_threshold=(
            max_abs_z_borderline_threshold
        ),

        max_abs_z_ood_threshold=(
            max_abs_z_ood_threshold
        ),

        mean_abs_z_borderline_threshold=(
            mean_abs_z_borderline_threshold
        ),

        mean_abs_z_ood_threshold=(
            mean_abs_z_ood_threshold
        ),

        summary=summary,
    )


# =============================================================
# Patient applicability assessment
# =============================================================


def assess_applicability(
    reference: ApplicabilityReference,
    X_target: pd.DataFrame,
) -> ApplicabilityAssessment:
    """
    Assess biological applicability for one or more target patients.

    No treatment or outcome information is used.
    """

    if not isinstance(
        reference,
        ApplicabilityReference,
    ):
        raise TypeError(
            "reference must be an ApplicabilityReference."
        )

    X_target = _validate_matrix(
        X_target,
        name="X_target",
    )

    _validate_feature_compatibility(
        reference,
        X_target,
    )

    z_scores = standardized_feature_deviation(
        X_target,
        reference.mean,
        reference.standard_deviation,
    )

    mahalanobis = mahalanobis_distance(
        X_target,
        reference.mean,
        reference.precision,
    )

    max_abs_z = (
        z_scores
        .abs()
        .max(
            axis=1
        )
        .rename(
            "max_abs_z"
        )
    )

    mean_abs_z = (
        z_scores
        .abs()
        .mean(
            axis=1
        )
        .rename(
            "mean_abs_z"
        )
    )

    fraction_abs_z_gt_2 = (
        z_scores
        .abs()
        .gt(
            2.0
        )
        .mean(
            axis=1
        )
        .rename(
            "fraction_features_abs_z_gt_2"
        )
    )

    fraction_abs_z_gt_3 = (
        z_scores
        .abs()
        .gt(
            3.0
        )
        .mean(
            axis=1
        )
        .rename(
            "fraction_features_abs_z_gt_3"
        )
    )

    mahalanobis_percentile = (
        _empirical_percentile_series(
            mahalanobis,
            reference.reference_mahalanobis,
        )
        .rename(
            "mahalanobis_reference_percentile"
        )
    )

    max_abs_z_percentile = (
        _empirical_percentile_series(
            max_abs_z,
            reference.reference_max_abs_z,
        )
        .rename(
            "max_abs_z_reference_percentile"
        )
    )

    mean_abs_z_percentile = (
        _empirical_percentile_series(
            mean_abs_z,
            reference.reference_mean_abs_z,
        )
        .rename(
            "mean_abs_z_reference_percentile"
        )
    )

    table = pd.concat(
        [
            mahalanobis,
            mahalanobis_percentile,

            max_abs_z,
            max_abs_z_percentile,

            mean_abs_z,
            mean_abs_z_percentile,

            fraction_abs_z_gt_2,
            fraction_abs_z_gt_3,
        ],
        axis=1,
    )

    mahalanobis_ood = (
        table[
            "mahalanobis_distance"
        ]
        > reference.mahalanobis_ood_threshold
    )

    max_z_ood = (
        table[
            "max_abs_z"
        ]
        > reference.max_abs_z_ood_threshold
    )

    mean_z_ood = (
        table[
            "mean_abs_z"
        ]
        > reference.mean_abs_z_ood_threshold
    )

    mahalanobis_borderline = (
        table[
            "mahalanobis_distance"
        ]
        > reference.mahalanobis_borderline_threshold
    )

    max_z_borderline = (
        table[
            "max_abs_z"
        ]
        > reference.max_abs_z_borderline_threshold
    )

    mean_z_borderline = (
        table[
            "mean_abs_z"
        ]
        > reference.mean_abs_z_borderline_threshold
    )

    table[
        "n_ood_flags"
    ] = (
        mahalanobis_ood.astype(
            int
        )
        + max_z_ood.astype(
            int
        )
        + mean_z_ood.astype(
            int
        )
    )

    table[
        "n_borderline_flags"
    ] = (
        mahalanobis_borderline.astype(
            int
        )
        + max_z_borderline.astype(
            int
        )
        + mean_z_borderline.astype(
            int
        )
    )

    table[
        "applicability_state"
    ] = "in_distribution"

    table.loc[
        table[
            "n_borderline_flags"
        ]
        >= 1,
        "applicability_state",
    ] = "borderline"

    table.loc[
        table[
            "n_ood_flags"
        ]
        >= 1,
        "applicability_state",
    ] = "out_of_distribution"

    # A continuous 0-1 score where larger means more similar to
    # the reference population.
    combined_percentile = (
        table[
            [
                "mahalanobis_reference_percentile",
                "max_abs_z_reference_percentile",
                "mean_abs_z_reference_percentile",
            ]
        ]
        .max(
            axis=1
        )
    )

    table[
        "applicability_score"
    ] = (
        1.0
        - combined_percentile
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    state_counts = (
        table[
            "applicability_state"
        ]
        .value_counts()
        .to_dict()
    )

    summary: dict[
        str,
        Any,
    ] = {
        "patients_assessed":
            int(
                len(
                    table
                )
            ),

        "in_distribution_n":
            int(
                state_counts.get(
                    "in_distribution",
                    0,
                )
            ),

        "borderline_n":
            int(
                state_counts.get(
                    "borderline",
                    0,
                )
            ),

        "out_of_distribution_n":
            int(
                state_counts.get(
                    "out_of_distribution",
                    0,
                )
            ),

        "fraction_in_distribution":
            float(
                (
                    table[
                        "applicability_state"
                    ]
                    == "in_distribution"
                ).mean()
            ),

        "fraction_borderline":
            float(
                (
                    table[
                        "applicability_state"
                    ]
                    == "borderline"
                ).mean()
            ),

        "fraction_out_of_distribution":
            float(
                (
                    table[
                        "applicability_state"
                    ]
                    == "out_of_distribution"
                ).mean()
            ),

        "median_mahalanobis_distance":
            float(
                table[
                    "mahalanobis_distance"
                ].median()
            ),

        "maximum_mahalanobis_distance":
            float(
                table[
                    "mahalanobis_distance"
                ].max()
            ),

        "median_applicability_score":
            float(
                table[
                    "applicability_score"
                ].median()
            ),

        "minimum_applicability_score":
            float(
                table[
                    "applicability_score"
                ].min()
            ),
    }

    return ApplicabilityAssessment(
        patient_table=table,
        feature_z_scores=z_scores,
        summary=summary,
    )


# =============================================================
# Cohort-level shift
# =============================================================


def _pooled_standard_deviation(
    reference: pd.DataFrame,
    target: pd.DataFrame,
) -> pd.Series:
    """
    Pooled SD for standardized mean differences.
    """

    n_reference = len(
        reference
    )

    n_target = len(
        target
    )

    reference_variance = reference.var(
        axis=0,
        ddof=1,
    )

    target_variance = target.var(
        axis=0,
        ddof=1,
    )

    pooled_variance = (
        (
            (
                n_reference
                - 1
            )
            * reference_variance
        )
        +
        (
            (
                n_target
                - 1
            )
            * target_variance
        )
    ) / (
        n_reference
        + n_target
        - 2
    )

    pooled_sd = np.sqrt(
        pooled_variance
    )

    pooled_sd = pooled_sd.replace(
        0.0,
        np.nan,
    )

    return pooled_sd


def compare_cohort_shift(
    reference_matrix: pd.DataFrame,
    target_matrix: pd.DataFrame,
    *,
    applicability_reference: ApplicabilityReference | None = None,
) -> CohortShiftAssessment:
    """
    Compare an external or holdout biological cohort with the reference.

    Outputs feature-level standardized mean differences and patient-level
    applicability scores.
    """

    reference_matrix = _validate_matrix(
        reference_matrix,
        name="reference_matrix",
    )

    target_matrix = _validate_matrix(
        target_matrix,
        name="target_matrix",
    )

    if tuple(
        reference_matrix.columns
    ) != tuple(
        target_matrix.columns
    ):
        raise ValueError(
            "reference_matrix and target_matrix must contain identical "
            "features in identical order."
        )

    if len(
        target_matrix
    ) < 2:
        raise ValueError(
            "At least two target patients are required for cohort-shift analysis."
        )

    if applicability_reference is None:
        applicability_reference = (
            fit_applicability_reference(
                reference_matrix
            )
        )

    else:
        _validate_feature_compatibility(
            applicability_reference,
            reference_matrix,
        )

    target_assessment = assess_applicability(
        applicability_reference,
        target_matrix,
    )

    reference_mean = reference_matrix.mean(
        axis=0
    )

    target_mean = target_matrix.mean(
        axis=0
    )

    pooled_sd = _pooled_standard_deviation(
        reference_matrix,
        target_matrix,
    )

    standardized_mean_difference = (
        target_mean
        - reference_mean
    ).divide(
        pooled_sd
    )

    standardized_mean_difference = (
        standardized_mean_difference
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )

    reference_sd = reference_matrix.std(
        axis=0,
        ddof=1,
    )

    target_sd = target_matrix.std(
        axis=0,
        ddof=1,
    )

    variance_ratio = (
        target_sd.pow(
            2
        )
        / reference_sd.pow(
            2
        )
    )

    variance_ratio = variance_ratio.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    feature_table = pd.DataFrame(
        {
            "reference_mean":
                reference_mean,

            "target_mean":
                target_mean,

            "mean_difference":
                target_mean
                - reference_mean,

            "standardized_mean_difference":
                standardized_mean_difference,

            "absolute_standardized_mean_difference":
                standardized_mean_difference.abs(),

            "reference_sd":
                reference_sd,

            "target_sd":
                target_sd,

            "variance_ratio":
                variance_ratio,
        }
    )

    feature_table.index.name = "feature"

    feature_table = (
        feature_table
        .sort_values(
            "absolute_standardized_mean_difference",
            ascending=False,
        )
    )

    absolute_smd = feature_table[
        "absolute_standardized_mean_difference"
    ]

    summary: dict[
        str,
        Any,
    ] = {
        "reference_patients":
            int(
                len(
                    reference_matrix
                )
            ),

        "target_patients":
            int(
                len(
                    target_matrix
                )
            ),

        "biological_features":
            int(
                reference_matrix.shape[
                    1
                ]
            ),

        "mean_absolute_smd":
            float(
                absolute_smd.mean()
            ),

        "median_absolute_smd":
            float(
                absolute_smd.median()
            ),

        "maximum_absolute_smd":
            float(
                absolute_smd.max()
            ),

        "fraction_features_abs_smd_ge_0_10":
            float(
                (
                    absolute_smd
                    >= 0.10
                ).mean()
            ),

        "fraction_features_abs_smd_ge_0_20":
            float(
                (
                    absolute_smd
                    >= 0.20
                ).mean()
            ),

        "fraction_features_abs_smd_ge_0_50":
            float(
                (
                    absolute_smd
                    >= 0.50
                ).mean()
            ),

        "target_fraction_in_distribution":
            float(
                target_assessment.summary[
                    "fraction_in_distribution"
                ]
            ),

        "target_fraction_borderline":
            float(
                target_assessment.summary[
                    "fraction_borderline"
                ]
            ),

        "target_fraction_out_of_distribution":
            float(
                target_assessment.summary[
                    "fraction_out_of_distribution"
                ]
            ),
    }

    return CohortShiftAssessment(
        feature_shift_table=feature_table,
        target_patient_assessment=(
            target_assessment
        ),
        summary=summary,
    )


# =============================================================
# Synthetic perturbation experiment
# =============================================================


def generate_shifted_cohort(
    X_reference: pd.DataFrame,
    *,
    n_patients: int = 100,
    mean_shift: float = 0.0,
    scale_multiplier: float = 1.0,
    random_state: int = 2026,
) -> pd.DataFrame:
    """
    Generate a synthetic cohort in the reference biological feature space.

    This is used as a positive control for OOD detection.

    mean_shift:
        Shift expressed in reference SD units and applied to every feature.

    scale_multiplier:
        Multiplier applied to reference SD.
    """

    X_reference = _validate_matrix(
        X_reference,
        name="X_reference",
    )

    if n_patients < 2:
        raise ValueError(
            "n_patients must be at least 2."
        )

    if not np.isfinite(
        mean_shift
    ):
        raise ValueError(
            "mean_shift must be finite."
        )

    if (
        not np.isfinite(
            scale_multiplier
        )
        or scale_multiplier <= 0.0
    ):
        raise ValueError(
            "scale_multiplier must be positive and finite."
        )

    rng = np.random.default_rng(
        random_state
    )

    mean = X_reference.mean(
        axis=0
    ).to_numpy(
        dtype=float
    )

    sd = X_reference.std(
        axis=0,
        ddof=1,
    ).to_numpy(
        dtype=float
    )

    shifted_mean = (
        mean
        + mean_shift
        * sd
    )

    shifted_sd = (
        scale_multiplier
        * sd
    )

    generated = rng.normal(
        loc=shifted_mean,
        scale=shifted_sd,
        size=(
            n_patients,
            X_reference.shape[
                1
            ],
        ),
    )

    index = pd.Index(
        [
            f"SHIFTED_{i:05d}"
            for i in range(
                n_patients
            )
        ],
        name="Patient_ID",
    )

    return pd.DataFrame(
        generated,
        index=index,
        columns=X_reference.columns,
    )


# =============================================================
# Internal holdout applicability experiment
# =============================================================


def split_reference_holdout(
    X: pd.DataFrame,
    *,
    holdout_fraction: float = 0.20,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Deterministically split a biological cohort into reference and holdout
    subsets for internal software validation.

    This does NOT constitute external validation.
    """

    X = _validate_matrix(
        X,
        name="X",
    )

    if not (
        0.10
        <= holdout_fraction
        <= 0.50
    ):
        raise ValueError(
            "holdout_fraction must be between 0.10 and 0.50."
        )

    rng = np.random.default_rng(
        random_state
    )

    permutation = rng.permutation(
        len(
            X
        )
    )

    n_holdout = int(
        round(
            len(
                X
            )
            * holdout_fraction
        )
    )

    n_holdout = max(
        2,
        n_holdout,
    )

    holdout_positions = permutation[
        :n_holdout
    ]

    reference_positions = permutation[
        n_holdout:
    ]

    reference = X.iloc[
        reference_positions
    ].copy()

    holdout = X.iloc[
        holdout_positions
    ].copy()

    return (
        reference,
        holdout,
    )


# =============================================================
# NeoTRIP interface
# =============================================================


def run_neotrip_generalizability_demo(
    *,
    holdout_fraction: float = 0.20,
    random_state: int = 42,
    borderline_quantile: float = 0.95,
    ood_quantile: float = 0.99,
    synthetic_shift_sd: float = 2.0,
    synthetic_patients: int = 100,
) -> dict[str, Any]:
    """
    Demonstrate HERMES applicability behavior using NeoTRIP.

    Three groups are compared:

    1. NeoTRIP reference subset
    2. Held-out NeoTRIP patients
    3. Deliberately shifted synthetic patients

    The synthetic shifted group serves as an OOD positive control.

    This is internal validation only.
    """

    from backend.app.treatment_effects.feature_builder import (
        build_treatment_effect_dataset,
    )

    dataset = build_treatment_effect_dataset()

    reference_matrix, holdout_matrix = (
        split_reference_holdout(
            dataset.X,
            holdout_fraction=(
                holdout_fraction
            ),
            random_state=(
                random_state
            ),
        )
    )

    reference = fit_applicability_reference(
        reference_matrix,
        borderline_quantile=(
            borderline_quantile
        ),
        ood_quantile=(
            ood_quantile
        ),
    )

    holdout_assessment = assess_applicability(
        reference,
        holdout_matrix,
    )

    synthetic_shifted = generate_shifted_cohort(
        reference_matrix,
        n_patients=(
            synthetic_patients
        ),
        mean_shift=(
            synthetic_shift_sd
        ),
        scale_multiplier=1.0,
        random_state=2026,
    )

    shifted_assessment = assess_applicability(
        reference,
        synthetic_shifted,
    )

    holdout_shift = compare_cohort_shift(
        reference_matrix,
        holdout_matrix,
        applicability_reference=(
            reference
        ),
    )

    shifted_shift = compare_cohort_shift(
        reference_matrix,
        synthetic_shifted,
        applicability_reference=(
            reference
        ),
    )

    return {
        "dataset":
            dataset,

        "reference_matrix":
            reference_matrix,

        "holdout_matrix":
            holdout_matrix,

        "synthetic_shifted_matrix":
            synthetic_shifted,

        "reference":
            reference,

        "holdout_assessment":
            holdout_assessment,

        "shifted_assessment":
            shifted_assessment,

        "holdout_cohort_shift":
            holdout_shift,

        "shifted_cohort_shift":
            shifted_shift,
    }


# =============================================================
# CLI
# =============================================================


def main() -> None:
    """
    Run the HERMES 2.0 applicability/OOD demonstration.
    """

    print(
        "=== HERMES 2.0 "
        "GENERALIZABILITY / APPLICABILITY / OOD ==="
    )

    print()

    result = run_neotrip_generalizability_demo(
        holdout_fraction=0.20,
        random_state=42,
        borderline_quantile=0.95,
        ood_quantile=0.99,
        synthetic_shift_sd=2.0,
        synthetic_patients=100,
    )

    reference = result[
        "reference"
    ]

    holdout = result[
        "holdout_assessment"
    ]

    shifted = result[
        "shifted_assessment"
    ]

    holdout_shift = result[
        "holdout_cohort_shift"
    ]

    shifted_shift = result[
        "shifted_cohort_shift"
    ]

    print(
        "=== REFERENCE MODEL ==="
    )

    for key, value in reference.summary.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== HELD-OUT NEOTRIP APPLICABILITY ==="
    )

    for key, value in holdout.summary.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== SYNTHETIC SHIFTED-COHORT APPLICABILITY ==="
    )

    for key, value in shifted.summary.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== HELD-OUT NEOTRIP COHORT SHIFT ==="
    )

    for key, value in holdout_shift.summary.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== SYNTHETIC SHIFTED COHORT SHIFT ==="
    )

    for key, value in shifted_shift.summary.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== MOST SHIFTED SYNTHETIC FEATURES ==="
    )

    print(
        shifted_shift
        .feature_shift_table
        .head(
            15
        )
        .to_string()
    )

    print()

    print(
        "=== MOST OOD SYNTHETIC PATIENTS ==="
    )

    print(
        shifted
        .patient_table
        .sort_values(
            [
                "n_ood_flags",
                "mahalanobis_distance",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(
            15
        )
        .to_string()
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Applicability is assessed from biological inputs only."
    )

    print(
        "Treatment assignment and clinical outcome are not used "
        "to classify patients as in- or out-of-distribution."
    )

    print(
        "Internal NeoTRIP holdout performance is not external validation."
    )

    print(
        "True external deployment will additionally require a "
        "reference-fitted biological representation transform."
    )


if __name__ == "__main__":
    main()