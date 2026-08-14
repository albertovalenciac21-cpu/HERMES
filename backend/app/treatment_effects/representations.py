"""
HERMES 2.0 — Biological Representation Engine
==============================================

Transforms gene-level transcriptomic measurements into biologically
interpretable pathway/program-level representations.

Design principles
-----------------
1. Gene sets are external inputs rather than silently hard-coded.
2. Scoring is deterministic and label-independent.
3. Gene-set coverage is explicitly measured.
4. Poorly represented pathways can be excluded using predefined rules.
5. Patient/sample ordering is preserved.
6. No treatment or outcome information is used during representation
   construction.
7. The implementation is designed to support later cross-cohort validation.

Initial scoring method
----------------------
For each gene:
    z_ij = (x_ij - mean_j) / sd_j

For each gene set:
    score_i = mean(z_ij for genes j in the set)

This produces a simple, transparent pathway activity representation.
More sophisticated methods (e.g. ssGSEA/GSVA-like approaches) can later
be evaluated as sensitivity analyses without changing the public interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneSetCoverage:
    """Coverage information for one gene set."""

    name: str
    requested_genes: int
    matched_genes: int
    coverage_fraction: float
    matched_gene_names: tuple[str, ...]
    missing_gene_names: tuple[str, ...]
    retained: bool


@dataclass
class RepresentationResult:
    """Container returned by the biological representation pipeline."""

    scores: pd.DataFrame
    coverage: pd.DataFrame
    retained_gene_sets: dict[str, tuple[str, ...]]
    dropped_gene_sets: dict[str, tuple[str, ...]]

    @property
    def n_patients(self) -> int:
        return int(self.scores.shape[0])

    @property
    def n_representations(self) -> int:
        return int(self.scores.shape[1])


def _normalize_gene_symbol(value: object) -> str:
    """
    Normalize a gene symbol for matching.

    Gene symbols are matched case-insensitively, but the expression
    matrix's original column names are retained in outputs.
    """
    return str(value).strip().upper()


def validate_expression_matrix(expression: pd.DataFrame) -> None:
    """
    Validate the expression matrix before biological scoring.

    Expected layout:
        rows    = patients/samples
        columns = genes
    """
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a pandas DataFrame")

    if expression.empty:
        raise ValueError("expression matrix is empty")

    if expression.index.has_duplicates:
        raise ValueError("expression matrix contains duplicate patient/sample IDs")

    if expression.columns.has_duplicates:
        raise ValueError("expression matrix contains duplicate gene columns")

    if expression.columns.isna().any():
        raise ValueError("expression matrix contains missing gene names")

    normalized = [_normalize_gene_symbol(x) for x in expression.columns]

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "expression matrix contains gene symbols that become duplicated "
            "after case normalization"
        )

    non_numeric = [
        str(column)
        for column in expression.columns
        if not pd.api.types.is_numeric_dtype(expression[column])
    ]

    if non_numeric:
        preview = ", ".join(non_numeric[:10])
        raise TypeError(
            "expression matrix contains non-numeric gene columns: "
            f"{preview}"
        )

    values = expression.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("expression matrix contains NaN or infinite values")


def validate_gene_sets(
    gene_sets: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """
    Validate and normalize externally supplied gene sets.

    Duplicate genes inside a gene set are removed while preserving order.
    """
    if not isinstance(gene_sets, Mapping):
        raise TypeError("gene_sets must be a mapping of name -> genes")

    if not gene_sets:
        raise ValueError("gene_sets is empty")

    cleaned: dict[str, tuple[str, ...]] = {}

    for raw_name, raw_genes in gene_sets.items():
        name = str(raw_name).strip()

        if not name:
            raise ValueError("gene-set names cannot be empty")

        if isinstance(raw_genes, str):
            raise TypeError(
                f"gene set '{name}' must contain an iterable of genes, "
                "not a single string"
            )

        seen: set[str] = set()
        genes: list[str] = []

        for gene in raw_genes:
            normalized = _normalize_gene_symbol(gene)

            if not normalized:
                continue

            if normalized not in seen:
                seen.add(normalized)
                genes.append(normalized)

        if not genes:
            raise ValueError(f"gene set '{name}' contains no valid genes")

        cleaned[name] = tuple(genes)

    return cleaned


def compute_gene_set_coverage(
    expression: pd.DataFrame,
    gene_sets: Mapping[str, Iterable[str]],
    *,
    min_genes: int = 3,
    min_coverage: float = 0.20,
) -> tuple[
    pd.DataFrame,
    dict[str, tuple[str, ...]],
    dict[str, tuple[str, ...]],
]:
    """
    Measure gene-set coverage against the expression matrix.

    A gene set is retained when BOTH conditions are satisfied:
        matched genes >= min_genes
        matched/requested >= min_coverage
    """
    validate_expression_matrix(expression)
    cleaned_gene_sets = validate_gene_sets(gene_sets)

    if min_genes < 1:
        raise ValueError("min_genes must be >= 1")

    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be between 0 and 1")

    expression_lookup = {
        _normalize_gene_symbol(column): str(column)
        for column in expression.columns
    }

    records: list[GeneSetCoverage] = []
    retained: dict[str, tuple[str, ...]] = {}
    dropped: dict[str, tuple[str, ...]] = {}

    for name, genes in cleaned_gene_sets.items():
        matched_normalized = [
            gene for gene in genes if gene in expression_lookup
        ]
        missing = tuple(
            gene for gene in genes if gene not in expression_lookup
        )

        matched_original = tuple(
            expression_lookup[gene] for gene in matched_normalized
        )

        requested_count = len(genes)
        matched_count = len(matched_original)
        coverage_fraction = matched_count / requested_count

        keep = (
            matched_count >= min_genes
            and coverage_fraction >= min_coverage
        )

        records.append(
            GeneSetCoverage(
                name=name,
                requested_genes=requested_count,
                matched_genes=matched_count,
                coverage_fraction=coverage_fraction,
                matched_gene_names=matched_original,
                missing_gene_names=missing,
                retained=keep,
            )
        )

        if keep:
            retained[name] = matched_original
        else:
            dropped[name] = matched_original

    coverage = pd.DataFrame(
        {
            "gene_set": [record.name for record in records],
            "requested_genes": [
                record.requested_genes for record in records
            ],
            "matched_genes": [
                record.matched_genes for record in records
            ],
            "coverage_fraction": [
                record.coverage_fraction for record in records
            ],
            "retained": [record.retained for record in records],
            "matched_gene_names": [
                record.matched_gene_names for record in records
            ],
            "missing_gene_names": [
                record.missing_gene_names for record in records
            ],
        }
    ).set_index("gene_set")

    return coverage, retained, dropped


def zscore_genes(
    expression: pd.DataFrame,
    *,
    ddof: int = 0,
    epsilon: float = 1e-12,
) -> pd.DataFrame:
    """
    Standardize each gene across patients/samples.

    Constant or numerically near-constant genes receive a z-score of zero
    rather than generating NaN/Inf values.
    """
    validate_expression_matrix(expression)

    if ddof < 0:
        raise ValueError("ddof must be >= 0")

    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")

    matrix = expression.astype(float)

    means = matrix.mean(axis=0)
    standard_deviations = matrix.std(axis=0, ddof=ddof)

    safe_standard_deviations = standard_deviations.copy()
    constant_mask = safe_standard_deviations <= epsilon
    safe_standard_deviations.loc[constant_mask] = 1.0

    standardized = (matrix - means) / safe_standard_deviations

    if constant_mask.any():
        standardized.loc[:, constant_mask] = 0.0

    if not np.isfinite(standardized.to_numpy(dtype=float)).all():
        raise ValueError(
            "gene standardization unexpectedly produced non-finite values"
        )

    return standardized


def score_gene_sets(
    expression: pd.DataFrame,
    gene_sets: Mapping[str, Iterable[str]],
    *,
    min_genes: int = 3,
    min_coverage: float = 0.20,
    standardize_genes: bool = True,
) -> RepresentationResult:
    """
    Convert gene-level expression into pathway/program-level scores.

    Scores are calculated as the arithmetic mean of the matched,
    standardized genes belonging to each retained gene set.

    No treatment or outcome labels are accepted by this function.
    """
    coverage, retained, dropped = compute_gene_set_coverage(
        expression,
        gene_sets,
        min_genes=min_genes,
        min_coverage=min_coverage,
    )

    if not retained:
        raise ValueError(
            "no gene sets passed the requested coverage thresholds"
        )

    if standardize_genes:
        scoring_matrix = zscore_genes(expression)
    else:
        scoring_matrix = expression.astype(float).copy()

    scores = pd.DataFrame(index=expression.index.copy())

    for name, matched_genes in retained.items():
        scores[name] = scoring_matrix.loc[:, list(matched_genes)].mean(
            axis=1
        )

    if scores.index.tolist() != expression.index.tolist():
        raise RuntimeError(
            "patient/sample ordering changed during representation scoring"
        )

    if not np.isfinite(scores.to_numpy(dtype=float)).all():
        raise RuntimeError(
            "biological representation matrix contains non-finite values"
        )

    return RepresentationResult(
        scores=scores,
        coverage=coverage,
        retained_gene_sets=retained,
        dropped_gene_sets=dropped,
    )


def load_gmt(path: str | Path) -> dict[str, tuple[str, ...]]:
    """
    Load gene sets from a standard GMT file.

    GMT layout:
        gene_set_name <TAB> description <TAB> gene1 <TAB> gene2 ...

    The description field is intentionally ignored for scoring.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"GMT file not found: {path}")

    gene_sets: dict[str, tuple[str, ...]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n\r")

            if not line:
                continue

            fields = line.split("\t")

            if len(fields) < 3:
                raise ValueError(
                    f"invalid GMT record on line {line_number}: "
                    "expected at least 3 tab-separated fields"
                )

            name = fields[0].strip()

            if not name:
                raise ValueError(
                    f"empty gene-set name on GMT line {line_number}"
                )

            if name in gene_sets:
                raise ValueError(
                    f"duplicate gene-set name in GMT file: {name}"
                )

            genes = tuple(
                gene
                for gene in (
                    _normalize_gene_symbol(value)
                    for value in fields[2:]
                )
                if gene
            )

            if not genes:
                raise ValueError(
                    f"gene set '{name}' contains no genes"
                )

            # Remove duplicate genes while preserving GMT order.
            genes = tuple(dict.fromkeys(genes))

            gene_sets[name] = genes

    if not gene_sets:
        raise ValueError("GMT file contains no gene sets")

    return gene_sets


def summarize_representations(
    result: RepresentationResult,
) -> dict[str, object]:
    """Generate a compact reproducible summary of a representation run."""
    retained_coverage = result.coverage.loc[
        result.coverage["retained"]
    ]

    return {
        "patients": result.n_patients,
        "representations": result.n_representations,
        "gene_sets_evaluated": int(result.coverage.shape[0]),
        "gene_sets_retained": int(
            result.coverage["retained"].sum()
        ),
        "gene_sets_dropped": int(
            (~result.coverage["retained"]).sum()
        ),
        "median_retained_coverage": (
            float(retained_coverage["coverage_fraction"].median())
            if not retained_coverage.empty
            else np.nan
        ),
        "minimum_retained_coverage": (
            float(retained_coverage["coverage_fraction"].min())
            if not retained_coverage.empty
            else np.nan
        ),
    }


def _demo() -> None:
    """Small deterministic smoke test independent of NeoTRIP data."""
    expression = pd.DataFrame(
        {
            "CD8A": [1.0, 2.0, 3.0, 4.0],
            "CD8B": [2.0, 3.0, 4.0, 5.0],
            "GZMB": [0.5, 1.0, 2.0, 4.0],
            "MKI67": [8.0, 7.0, 6.0, 5.0],
            "TOP2A": [4.0, 5.0, 6.0, 7.0],
            "ESR1": [0.1, 0.1, 0.1, 0.1],
        },
        index=["P001", "P002", "P003", "P004"],
    )

    gene_sets = {
        "CYTOTOXIC_PROGRAM": ["CD8A", "CD8B", "GZMB"],
        "PROLIFERATION_PROGRAM": ["MKI67", "TOP2A", "MISSING1"],
        "INSUFFICIENT_PROGRAM": ["NOT_PRESENT_1", "NOT_PRESENT_2"],
    }

    result = score_gene_sets(
        expression,
        gene_sets,
        min_genes=2,
        min_coverage=0.50,
    )

    summary = summarize_representations(result)

    print("=== HERMES 2.0 Biological Representation Engine ===")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print()
    print("Coverage:")
    print(result.coverage)

    print()
    print("Scores:")
    print(result.scores)


if __name__ == "__main__":
    _demo()