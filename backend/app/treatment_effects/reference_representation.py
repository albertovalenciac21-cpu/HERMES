"""
HERMES 2.0
Reference-Fitted Biological Representation
============================================

Purpose
-------
Create a biological representation that is fitted once on a reference
training cohort and then frozen for application to:

    - internal holdout patients
    - external TNBC cohorts
    - future individual patients

This prevents data leakage caused by recomputing normalization parameters
from the cohort being predicted.

Core principle
--------------
Reference cohort:

    raw expression
        -> fit gene means / SDs
        -> standardize genes using HERMES-compatible semantics
        -> aggregate genes into pathway scores
        -> optionally fit pathway means / SDs
        -> freeze representation

New patient / external cohort:

    raw expression
        -> use FROZEN reference gene means / SDs
        -> aggregate using SAME pathway definitions
        -> optionally use FROZEN pathway means / SDs
        -> HERMES treatment-effect pipeline

The target cohort NEVER contributes to fitted preprocessing parameters.

Compatibility
-------------
The gene-level standardization implemented here intentionally reproduces
the existing HERMES representations.zscore_genes() convention:

    mean = pandas mean across patients
    SD   = pandas std(ddof=0)
    epsilon = 1e-12
    constant / near-constant genes -> z-score 0

This is important because the reference-fitted engine should reproduce the
existing HERMES Hallmark representation when both are fitted on exactly
the same cohort.

Optional pathway-level scaling is a separate second-stage transformation.

IMPORTANT
---------
This module establishes preprocessing transportability.

It does NOT itself establish:
    - external predictive performance
    - causal transportability
    - treatment-effect validity
    - predictive biomarker validity
    - clinical utility
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


# =============================================================
# Constants
# =============================================================


HERMES_GENE_DDOF = 0
HERMES_GENE_EPSILON = 1e-12

PATHWAY_SD_DDOF = 1
PATHWAY_MINIMUM_SD = 1e-8


# =============================================================
# Result containers
# =============================================================


@dataclass(frozen=True)
class ReferenceRepresentation:
    """
    Frozen HERMES biological representation.

    gene_mean and gene_sd contain the parameters learned ONLY from
    the reference cohort.

    gene_sd contains the safe denominator used by HERMES. Genes whose
    original SD was <= gene_epsilon receive safe SD = 1.0 and are
    explicitly assigned standardized value 0.

    pathway_mean and pathway_sd are optional second-stage normalization
    parameters fitted ONLY from the reference cohort.
    """

    gene_names: tuple[str, ...]
    pathway_names: tuple[str, ...]

    gene_mean: pd.Series
    gene_sd: pd.Series

    pathway_mean: pd.Series
    pathway_sd: pd.Series

    pathway_genes: dict[str, tuple[str, ...]]

    constant_genes: tuple[str, ...]

    minimum_genes_per_pathway: int
    standardize_pathways: bool

    gene_ddof: int
    gene_epsilon: float

    summary: dict[str, Any]


@dataclass
class RepresentationTransformResult:
    """
    Result of applying a frozen biological representation.
    """

    pathway_scores: pd.DataFrame
    standardized_gene_expression: pd.DataFrame

    missing_genes: tuple[str, ...]
    extra_genes: tuple[str, ...]

    pathway_coverage: pd.DataFrame

    summary: dict[str, Any]


# =============================================================
# Validation helpers
# =============================================================


def _validate_expression(
    expression: pd.DataFrame,
    *,
    name: str,
) -> pd.DataFrame:
    """
    Validate a patient x gene expression matrix.
    """

    if not isinstance(
        expression,
        pd.DataFrame,
    ):
        raise TypeError(
            f"{name} must be a pandas DataFrame."
        )

    if expression.empty:
        raise ValueError(
            f"{name} cannot be empty."
        )

    if expression.index.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate patient IDs."
        )

    if expression.columns.duplicated().any():
        raise ValueError(
            f"{name} contains duplicate gene names."
        )

    if expression.columns.isna().any():
        raise ValueError(
            f"{name} contains missing gene names."
        )

    numeric = expression.astype(
        float
    )

    if not np.isfinite(
        numeric.to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    numeric.columns = pd.Index(
        [
            str(column)
            for column in numeric.columns
        ]
    )

    return numeric


def _normalize_gene_sets(
    gene_sets: Mapping[
        str,
        Sequence[str],
    ],
) -> dict[str, tuple[str, ...]]:
    """
    Normalize pathway definitions while preserving pathway order.
    """

    if not isinstance(
        gene_sets,
        Mapping,
    ):
        raise TypeError(
            "gene_sets must be a mapping of pathway -> genes."
        )

    if len(
        gene_sets
    ) == 0:
        raise ValueError(
            "gene_sets cannot be empty."
        )

    normalized: dict[
        str,
        tuple[str, ...],
    ] = {}

    for pathway, genes in (
        gene_sets.items()
    ):

        pathway_name = str(
            pathway
        )

        if not pathway_name:
            raise ValueError(
                "Pathway names cannot be empty."
            )

        cleaned: list[str] = []
        seen: set[str] = set()

        for gene in genes:

            gene_name = str(
                gene
            )

            if not gene_name:
                continue

            if gene_name in seen:
                continue

            cleaned.append(
                gene_name
            )

            seen.add(
                gene_name
            )

        if len(
            cleaned
        ) == 0:
            raise ValueError(
                f"Pathway {pathway_name!r} contains no genes."
            )

        normalized[
            pathway_name
        ] = tuple(
            cleaned
        )

    return normalized


# =============================================================
# HERMES-compatible gene scaling
# =============================================================


def _fit_reference_gene_scaling(
    expression: pd.DataFrame,
    *,
    ddof: int = HERMES_GENE_DDOF,
    epsilon: float = HERMES_GENE_EPSILON,
) -> tuple[
    pd.Series,
    pd.Series,
    tuple[str, ...],
]:
    """
    Fit gene-level scaling parameters using the same mathematical
    convention as HERMES representations.zscore_genes().

    Returns
    -------
    mean:
        Reference gene means.

    safe_sd:
        SD denominator used during transformation. Near-constant genes
        receive 1.0 so division is safe.

    constant_genes:
        Genes whose original reference SD was <= epsilon.
    """

    if ddof < 0:
        raise ValueError(
            "ddof must be >= 0."
        )

    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be > 0."
        )

    matrix = expression.astype(
        float
    )

    mean = matrix.mean(
        axis=0
    )

    raw_sd = matrix.std(
        axis=0,
        ddof=ddof,
    )

    constant_mask = (
        raw_sd
        <= epsilon
    )

    safe_sd = raw_sd.copy()

    safe_sd.loc[
        constant_mask
    ] = 1.0

    constant_genes = tuple(
        str(gene)
        for gene in raw_sd.index[
            constant_mask
        ]
    )

    if not np.isfinite(
        mean.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Reference gene means contain non-finite values."
        )

    if not np.isfinite(
        safe_sd.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Reference gene SDs contain non-finite values."
        )

    return (
        mean.astype(
            float
        ),
        safe_sd.astype(
            float
        ),
        constant_genes,
    )


def _apply_reference_gene_scaling(
    expression: pd.DataFrame,
    *,
    gene_mean: pd.Series,
    gene_sd: pd.Series,
    constant_genes: Sequence[str],
) -> pd.DataFrame:
    """
    Apply frozen gene-level scaling.

    This reproduces HERMES zscore_genes() behavior except that the
    parameters come from the frozen reference cohort instead of the
    target cohort.
    """

    standardized = (
        expression
        - gene_mean
    ).divide(
        gene_sd,
        axis=1,
    )

    constant_present = [
        gene
        for gene in constant_genes
        if gene in standardized.columns
    ]

    if constant_present:

        standardized.loc[
            :,
            constant_present,
        ] = 0.0

    if not np.isfinite(
        standardized.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Reference gene standardization produced "
            "non-finite values."
        )

    return standardized


# =============================================================
# Optional pathway-level scaling
# =============================================================


def _safe_pathway_sd(
    values: pd.DataFrame,
    *,
    minimum_sd: float = PATHWAY_MINIMUM_SD,
) -> pd.Series:
    """
    Compute pathway-level SDs.

    This is deliberately separate from HERMES gene-level scaling.

    Pathway scaling is an optional second-stage transformation and uses
    ddof=1 by default.
    """

    if minimum_sd <= 0.0:
        raise ValueError(
            "minimum_sd must be positive."
        )

    sd = values.std(
        axis=0,
        ddof=PATHWAY_SD_DDOF,
    ).astype(
        float
    )

    safe_sd = sd.copy()

    safe_sd.loc[
        safe_sd
        < minimum_sd
    ] = 1.0

    return safe_sd


# =============================================================
# Core pathway aggregation
# =============================================================


def aggregate_pathways(
    standardized_expression: pd.DataFrame,
    pathway_genes: Mapping[
        str,
        Sequence[str],
    ],
) -> pd.DataFrame:
    """
    Aggregate standardized genes into pathway scores.

    Current HERMES pathway score:

        arithmetic mean of standardized member genes
    """

    scores: dict[
        str,
        pd.Series,
    ] = {}

    for pathway, genes in (
        pathway_genes.items()
    ):

        genes = list(
            genes
        )

        if len(
            genes
        ) == 0:
            raise ValueError(
                f"Pathway {pathway!r} has no genes."
            )

        missing = [
            gene
            for gene in genes
            if gene
            not in standardized_expression.columns
        ]

        if missing:
            raise ValueError(
                f"Frozen pathway {pathway!r} is missing "
                f"required genes: {missing[:10]}"
            )

        scores[
            pathway
        ] = (
            standardized_expression
            .loc[
                :,
                genes,
            ]
            .mean(
                axis=1
            )
        )

    result = pd.DataFrame(
        scores,
        index=standardized_expression.index,
    )

    if not np.isfinite(
        result.to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Pathway aggregation produced non-finite values."
        )

    return result


# =============================================================
# Fit reference representation
# =============================================================


def fit_reference_representation(
    reference_expression: pd.DataFrame,
    gene_sets: Mapping[
        str,
        Sequence[str],
    ],
    *,
    minimum_genes_per_pathway: int = 3,
    standardize_pathways: bool = True,
) -> ReferenceRepresentation:
    """
    Fit the HERMES biological representation on a reference cohort.

    Every normalization parameter in the returned representation is
    estimated ONLY from reference_expression.
    """

    expression = _validate_expression(
        reference_expression,
        name="reference_expression",
    )

    normalized_gene_sets = (
        _normalize_gene_sets(
            gene_sets
        )
    )

    if minimum_genes_per_pathway < 1:
        raise ValueError(
            "minimum_genes_per_pathway must be at least 1."
        )

    (
        gene_mean,
        gene_sd,
        constant_genes,
    ) = _fit_reference_gene_scaling(
        expression,
        ddof=HERMES_GENE_DDOF,
        epsilon=HERMES_GENE_EPSILON,
    )

    standardized_expression = (
        _apply_reference_gene_scaling(
            expression,
            gene_mean=gene_mean,
            gene_sd=gene_sd,
            constant_genes=constant_genes,
        )
    )

    frozen_pathways: dict[
        str,
        tuple[str, ...],
    ] = {}

    pathway_records: list[
        dict[str, Any]
    ] = []

    available_genes = set(
        expression.columns
    )

    for pathway, requested_genes in (
        normalized_gene_sets.items()
    ):

        usable_genes = tuple(
            gene
            for gene in requested_genes
            if gene in available_genes
        )

        coverage_fraction = float(
            len(
                usable_genes
            )
            / len(
                requested_genes
            )
        )

        included = bool(
            len(
                usable_genes
            )
            >= minimum_genes_per_pathway
        )

        pathway_records.append(
            {
                "pathway":
                    pathway,

                "requested_genes":
                    len(
                        requested_genes
                    ),

                "available_genes":
                    len(
                        usable_genes
                    ),

                "coverage_fraction":
                    coverage_fraction,

                "included":
                    included,
            }
        )

        if included:

            frozen_pathways[
                pathway
            ] = usable_genes

    if len(
        frozen_pathways
    ) == 0:
        raise ValueError(
            "No pathways passed the minimum gene requirement."
        )

    reference_pathway_scores = (
        aggregate_pathways(
            standardized_expression,
            frozen_pathways,
        )
    )

    pathway_mean = (
        reference_pathway_scores
        .mean(
            axis=0
        )
        .astype(
            float
        )
    )

    pathway_sd = (
        _safe_pathway_sd(
            reference_pathway_scores
        )
    )

    if standardize_pathways:

        final_reference_scores = (
            reference_pathway_scores
            - pathway_mean
        ).divide(
            pathway_sd,
            axis=1,
        )

    else:

        final_reference_scores = (
            reference_pathway_scores
        )

    if not np.isfinite(
        final_reference_scores
        .to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Reference pathway representation contains "
            "non-finite values."
        )

    coverage = pd.DataFrame(
        pathway_records
    )

    included_coverage = (
        coverage[
            coverage[
                "included"
            ]
        ]
    )

    summary: dict[
        str,
        Any,
    ] = {
        "reference_patients":
            int(
                len(
                    expression
                )
            ),

        "reference_genes":
            int(
                expression.shape[
                    1
                ]
            ),

        "constant_or_near_constant_genes":
            int(
                len(
                    constant_genes
                )
            ),

        "gene_standardization_ddof":
            int(
                HERMES_GENE_DDOF
            ),

        "gene_standardization_epsilon":
            float(
                HERMES_GENE_EPSILON
            ),

        "pathways_requested":
            int(
                len(
                    normalized_gene_sets
                )
            ),

        "pathways_retained":
            int(
                len(
                    frozen_pathways
                )
            ),

        "minimum_genes_per_pathway":
            int(
                minimum_genes_per_pathway
            ),

        "standardize_pathways":
            bool(
                standardize_pathways
            ),

        "mean_retained_pathway_gene_coverage":
            float(
                included_coverage[
                    "coverage_fraction"
                ].mean()
            ),

        "minimum_retained_pathway_gene_coverage":
            float(
                included_coverage[
                    "coverage_fraction"
                ].min()
            ),
    }

    return ReferenceRepresentation(
        gene_names=tuple(
            expression.columns
        ),

        pathway_names=tuple(
            frozen_pathways.keys()
        ),

        gene_mean=(
            gene_mean.copy()
        ),

        gene_sd=(
            gene_sd.copy()
        ),

        pathway_mean=(
            pathway_mean.copy()
        ),

        pathway_sd=(
            pathway_sd.copy()
        ),

        pathway_genes={
            pathway:
                tuple(
                    genes
                )

            for pathway, genes
            in frozen_pathways.items()
        },

        constant_genes=(
            constant_genes
        ),

        minimum_genes_per_pathway=int(
            minimum_genes_per_pathway
        ),

        standardize_pathways=bool(
            standardize_pathways
        ),

        gene_ddof=int(
            HERMES_GENE_DDOF
        ),

        gene_epsilon=float(
            HERMES_GENE_EPSILON
        ),

        summary=summary,
    )


# =============================================================
# Apply frozen representation
# =============================================================


def transform_with_reference(
    representation: ReferenceRepresentation,
    target_expression: pd.DataFrame,
    *,
    allow_extra_genes: bool = True,
    missing_gene_policy: str = "error",
) -> RepresentationTransformResult:
    """
    Transform unseen patients using ONLY frozen reference parameters.

    Parameters
    ----------
    representation:
        Previously fitted HERMES reference representation.

    target_expression:
        Patient x gene expression matrix.

    allow_extra_genes:
        Extra genes are ignored when True.

    missing_gene_policy:
        "error"
            Reject target data if a gene required by the frozen pathways
            is absent.

        "reference_mean"
            Impute a missing gene with the reference mean. Its resulting
            standardized value is therefore exactly 0.

    No normalization statistic is estimated from target_expression.
    """

    if not isinstance(
        representation,
        ReferenceRepresentation,
    ):
        raise TypeError(
            "representation must be a ReferenceRepresentation."
        )

    target = _validate_expression(
        target_expression,
        name="target_expression",
    )

    valid_missing_policies = {
        "error",
        "reference_mean",
    }

    if (
        missing_gene_policy
        not in valid_missing_policies
    ):
        raise ValueError(
            "missing_gene_policy must be either "
            "'error' or 'reference_mean'."
        )

    required_genes = tuple(
        dict.fromkeys(
            gene
            for pathway in representation.pathway_names
            for gene in representation.pathway_genes[
                pathway
            ]
        )
    )

    target_gene_set = set(
        target.columns
    )

    required_gene_set = set(
        required_genes
    )

    missing_genes = tuple(
        sorted(
            required_gene_set
            - target_gene_set
        )
    )

    extra_genes = tuple(
        sorted(
            target_gene_set
            - set(
                representation.gene_names
            )
        )
    )

    if (
        extra_genes
        and not allow_extra_genes
    ):
        raise ValueError(
            "Target expression contains unexpected genes: "
            f"{extra_genes[:10]}"
        )

    working = target.copy()

    if missing_genes:

        if missing_gene_policy == "error":

            raise ValueError(
                "Target expression is missing genes required "
                "by the frozen HERMES representation: "
                f"{missing_genes[:10]}"
            )

        for gene in missing_genes:

            working[
                gene
            ] = float(
                representation
                .gene_mean[
                    gene
                ]
            )

    working = working[
        list(
            required_genes
        )
    ]

    frozen_gene_mean = (
        representation
        .gene_mean
        .reindex(
            required_genes
        )
    )

    frozen_gene_sd = (
        representation
        .gene_sd
        .reindex(
            required_genes
        )
    )

    constant_required_genes = tuple(
        gene
        for gene in representation.constant_genes
        if gene in required_gene_set
    )

    standardized_expression = (
        _apply_reference_gene_scaling(
            working,
            gene_mean=(
                frozen_gene_mean
            ),
            gene_sd=(
                frozen_gene_sd
            ),
            constant_genes=(
                constant_required_genes
            ),
        )
    )

    raw_pathway_scores = aggregate_pathways(
        standardized_expression,
        representation.pathway_genes,
    )

    raw_pathway_scores = (
        raw_pathway_scores[
            list(
                representation.pathway_names
            )
        ]
    )

    if representation.standardize_pathways:

        pathway_scores = (
            raw_pathway_scores
            - representation.pathway_mean
        ).divide(
            representation.pathway_sd,
            axis=1,
        )

    else:

        pathway_scores = (
            raw_pathway_scores
        )

    pathway_scores = pathway_scores[
        list(
            representation.pathway_names
        )
    ]

    if not np.isfinite(
        pathway_scores
        .to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Frozen pathway transformation produced "
            "non-finite values."
        )

    coverage_records: list[
        dict[str, Any]
    ] = []

    missing_gene_set = set(
        missing_genes
    )

    for pathway in (
        representation.pathway_names
    ):

        genes = (
            representation
            .pathway_genes[
                pathway
            ]
        )

        pathway_missing = [
            gene
            for gene in genes
            if gene in missing_gene_set
        ]

        observed_gene_count = (
            len(
                genes
            )
            - len(
                pathway_missing
            )
        )

        coverage_records.append(
            {
                "pathway":
                    pathway,

                "reference_gene_count":
                    int(
                        len(
                            genes
                        )
                    ),

                "observed_target_gene_count":
                    int(
                        observed_gene_count
                    ),

                "imputed_gene_count":
                    int(
                        len(
                            pathway_missing
                        )
                    ),

                "observed_fraction":
                    float(
                        observed_gene_count
                        / len(
                            genes
                        )
                    ),
            }
        )

    pathway_coverage = (
        pd.DataFrame(
            coverage_records
        )
        .set_index(
            "pathway"
        )
    )

    summary: dict[
        str,
        Any,
    ] = {
        "patients_transformed":
            int(
                len(
                    target
                )
            ),

        "pathways_generated":
            int(
                pathway_scores.shape[
                    1
                ]
            ),

        "required_genes":
            int(
                len(
                    required_genes
                )
            ),

        "missing_required_genes":
            int(
                len(
                    missing_genes
                )
            ),

        "extra_target_genes":
            int(
                len(
                    extra_genes
                )
            ),

        "missing_gene_policy":
            str(
                missing_gene_policy
            ),

        "minimum_pathway_observed_fraction":
            float(
                pathway_coverage[
                    "observed_fraction"
                ].min()
            ),

        "mean_pathway_observed_fraction":
            float(
                pathway_coverage[
                    "observed_fraction"
                ].mean()
            ),

        "gene_standardization_ddof":
            int(
                representation.gene_ddof
            ),

        "gene_standardization_epsilon":
            float(
                representation.gene_epsilon
            ),

        "used_reference_gene_scaling":
            True,

        "used_reference_pathway_scaling":
            bool(
                representation
                .standardize_pathways
            ),
    }

    return RepresentationTransformResult(
        pathway_scores=(
            pathway_scores
        ),

        standardized_gene_expression=(
            standardized_expression
        ),

        missing_genes=(
            missing_genes
        ),

        extra_genes=(
            extra_genes
        ),

        pathway_coverage=(
            pathway_coverage
        ),

        summary=summary,
    )


# =============================================================
# Fit + transform convenience interface
# =============================================================


def fit_transform_reference(
    reference_expression: pd.DataFrame,
    gene_sets: Mapping[
        str,
        Sequence[str],
    ],
    *,
    minimum_genes_per_pathway: int = 3,
    standardize_pathways: bool = True,
) -> tuple[
    ReferenceRepresentation,
    RepresentationTransformResult,
]:
    """
    Fit the frozen representation and transform the reference cohort.

    Useful for constructing the training feature matrix.
    """

    representation = fit_reference_representation(
        reference_expression,
        gene_sets,
        minimum_genes_per_pathway=(
            minimum_genes_per_pathway
        ),
        standardize_pathways=(
            standardize_pathways
        ),
    )

    transformed = transform_with_reference(
        representation,
        reference_expression,
        allow_extra_genes=True,
        missing_gene_policy="error",
    )

    return (
        representation,
        transformed,
    )


# =============================================================
# Frozen vs independent scaling diagnostic
# =============================================================


def compare_reference_vs_independent_scaling(
    representation: ReferenceRepresentation,
    target_expression: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare correct frozen-reference transformation against incorrectly
    refitting normalization parameters on the target cohort.

    This diagnostic demonstrates why independently standardizing an
    external cohort can normalize away biological distribution shift.

    This function is NOT part of the deployment pathway.
    """

    target = _validate_expression(
        target_expression,
        name="target_expression",
    )

    required_genes = tuple(
        dict.fromkeys(
            gene
            for pathway in representation.pathway_names
            for gene in representation.pathway_genes[
                pathway
            ]
        )
    )

    missing = [
        gene
        for gene in required_genes
        if gene not in target.columns
    ]

    if missing:
        raise ValueError(
            "Target cohort lacks genes required for the "
            "scaling comparison."
        )

    target_required = target[
        list(
            required_genes
        )
    ]

    reference_transformed = (
        transform_with_reference(
            representation,
            target,
            missing_gene_policy="error",
        )
        .pathway_scores
    )

    (
        independent_gene_mean,
        independent_gene_sd,
        independent_constant_genes,
    ) = _fit_reference_gene_scaling(
        target_required,
        ddof=HERMES_GENE_DDOF,
        epsilon=HERMES_GENE_EPSILON,
    )

    independently_scaled_genes = (
        _apply_reference_gene_scaling(
            target_required,
            gene_mean=(
                independent_gene_mean
            ),
            gene_sd=(
                independent_gene_sd
            ),
            constant_genes=(
                independent_constant_genes
            ),
        )
    )

    independent_pathways = aggregate_pathways(
        independently_scaled_genes,
        representation.pathway_genes,
    )

    if representation.standardize_pathways:

        independent_pathway_mean = (
            independent_pathways
            .mean(
                axis=0
            )
        )

        independent_pathway_sd = (
            _safe_pathway_sd(
                independent_pathways
            )
        )

        independent_pathways = (
            independent_pathways
            - independent_pathway_mean
        ).divide(
            independent_pathway_sd,
            axis=1,
        )

    records: list[
        dict[str, Any]
    ] = []

    for pathway in (
        representation.pathway_names
    ):

        frozen = (
            reference_transformed[
                pathway
            ]
        )

        independent = (
            independent_pathways[
                pathway
            ]
        )

        difference = (
            frozen
            - independent
        )

        correlation = frozen.corr(
            independent,
            method="spearman",
        )

        records.append(
            {
                "pathway":
                    pathway,

                "mean_reference_scaled":
                    float(
                        frozen.mean()
                    ),

                "mean_independently_scaled":
                    float(
                        independent.mean()
                    ),

                "mean_absolute_difference":
                    float(
                        difference
                        .abs()
                        .mean()
                    ),

                "maximum_absolute_difference":
                    float(
                        difference
                        .abs()
                        .max()
                    ),

                "spearman_correlation":
                    float(
                        correlation
                    ),
            }
        )

    return (
        pd.DataFrame(
            records
        )
        .set_index(
            "pathway"
        )
        .sort_values(
            "mean_absolute_difference",
            ascending=False,
        )
    )


# =============================================================
# Synthetic demonstration
# =============================================================


def _generate_demo_expression(
    *,
    n_reference: int = 200,
    n_target: int = 60,
    n_genes: int = 30,
    target_shift: float = 1.25,
    random_state: int = 2026,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[str, ...]],
]:
    """
    Generate deterministic development data.
    """

    rng = np.random.default_rng(
        random_state
    )

    genes = [
        f"GENE_{i:03d}"
        for i in range(
            n_genes
        )
    ]

    reference = pd.DataFrame(
        rng.normal(
            loc=5.0,
            scale=2.0,
            size=(
                n_reference,
                n_genes,
            ),
        ),
        index=pd.Index(
            [
                f"REFERENCE_{i:04d}"
                for i in range(
                    n_reference
                )
            ],
            name="Patient_ID",
        ),
        columns=genes,
    )

    target = pd.DataFrame(
        rng.normal(
            loc=(
                5.0
                + target_shift
            ),
            scale=2.0,
            size=(
                n_target,
                n_genes,
            ),
        ),
        index=pd.Index(
            [
                f"TARGET_{i:04d}"
                for i in range(
                    n_target
                )
            ],
            name="Patient_ID",
        ),
        columns=genes,
    )

    gene_sets = {
        "PATHWAY_A":
            tuple(
                genes[
                    0:10
                ]
            ),

        "PATHWAY_B":
            tuple(
                genes[
                    10:20
                ]
            ),

        "PATHWAY_C":
            tuple(
                genes[
                    20:30
                ]
            ),
    }

    return (
        reference,
        target,
        gene_sets,
    )


# =============================================================
# CLI
# =============================================================


def main() -> None:
    """
    Demonstrate frozen-reference representation behavior.
    """

    print(
        "=== HERMES 2.0 "
        "REFERENCE-FITTED BIOLOGICAL REPRESENTATION ==="
    )

    print()

    (
        reference_expression,
        target_expression,
        gene_sets,
    ) = _generate_demo_expression()

    (
        representation,
        reference_result,
    ) = fit_transform_reference(
        reference_expression,
        gene_sets,
        minimum_genes_per_pathway=3,
        standardize_pathways=True,
    )

    target_result = transform_with_reference(
        representation,
        target_expression,
        missing_gene_policy="error",
    )

    scaling_comparison = (
        compare_reference_vs_independent_scaling(
            representation,
            target_expression,
        )
    )

    print(
        "=== FROZEN REFERENCE ==="
    )

    for key, value in (
        representation
        .summary
        .items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== REFERENCE TRANSFORMATION ==="
    )

    for key, value in (
        reference_result
        .summary
        .items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "=== UNSEEN TARGET TRANSFORMATION ==="
    )

    for key, value in (
        target_result
        .summary
        .items()
    ):
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "Reference pathway means after frozen scaling:"
    )

    print(
        reference_result
        .pathway_scores
        .mean()
        .to_string()
    )

    print()

    print(
        "Shifted target pathway means after FROZEN reference scaling:"
    )

    print(
        target_result
        .pathway_scores
        .mean()
        .to_string()
    )

    print()

    print(
        "=== FROZEN VS INDEPENDENT TARGET SCALING ==="
    )

    print(
        scaling_comparison
        .to_string()
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Gene standardization uses the same ddof=0 and "
        "epsilon=1e-12 convention as the established "
        "HERMES representation engine."
    )

    print(
        "Target-cohort statistics were never used to fit "
        "the frozen HERMES representation."
    )

    print(
        "Biological shifts therefore remain visible instead "
        "of being normalized away."
    )

    print(
        "External clinical validation is still required before "
        "making claims about treatment-effect generalizability."
    )


if __name__ == "__main__":
    main()