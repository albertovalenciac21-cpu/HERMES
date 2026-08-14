"""
Tests for the HERMES 2.0 biological representation engine.

These tests verify:
- expression validation
- gene-set validation
- case-insensitive matching
- gene-set coverage
- pathway scoring
- patient-order preservation
- deterministic behavior
- label-independent interface
- GMT parsing
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from backend.app.treatment_effects.representations import (
    compute_gene_set_coverage,
    load_gmt,
    score_gene_sets,
    summarize_representations,
    validate_expression_matrix,
    validate_gene_sets,
    zscore_genes,
)


def make_expression() -> pd.DataFrame:
    return pd.DataFrame(
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


def make_gene_sets():
    return {
        "CYTOTOXIC": ["CD8A", "CD8B", "GZMB"],
        "PROLIFERATION": ["MKI67", "TOP2A", "MISSING"],
        "ABSENT": ["FAKE1", "FAKE2"],
    }


def test_expression_validation():
    expression = make_expression()
    validate_expression_matrix(expression)

    try:
        validate_expression_matrix(pd.DataFrame())
        raise AssertionError("empty matrix should fail")
    except ValueError:
        pass

    bad = expression.copy()
    bad.loc["P001", "CD8A"] = np.nan

    try:
        validate_expression_matrix(bad)
        raise AssertionError("NaN-containing matrix should fail")
    except ValueError:
        pass


def test_gene_set_validation():
    cleaned = validate_gene_sets(
        {
            "immune": ["cd8a", "CD8A", " gzmb "],
        }
    )

    assert cleaned["immune"] == ("CD8A", "GZMB")


def test_case_insensitive_matching():
    expression = make_expression()

    gene_sets = {
        "immune": ["cd8a", "cd8b", "gzmb"],
    }

    coverage, retained, dropped = compute_gene_set_coverage(
        expression,
        gene_sets,
        min_genes=3,
        min_coverage=1.0,
    )

    assert coverage.loc["immune", "matched_genes"] == 3
    assert bool(coverage.loc["immune", "retained"])
    assert retained["immune"] == ("CD8A", "CD8B", "GZMB")
    assert dropped == {}


def test_coverage_filtering():
    expression = make_expression()

    coverage, retained, dropped = compute_gene_set_coverage(
        expression,
        make_gene_sets(),
        min_genes=2,
        min_coverage=0.50,
    )

    assert bool(coverage.loc["CYTOTOXIC", "retained"])
    assert bool(coverage.loc["PROLIFERATION", "retained"])
    assert not bool(coverage.loc["ABSENT", "retained"])

    assert "CYTOTOXIC" in retained
    assert "PROLIFERATION" in retained
    assert "ABSENT" in dropped


def test_zscore_behavior():
    expression = make_expression()
    standardized = zscore_genes(expression)

    nonconstant = [
        "CD8A",
        "CD8B",
        "GZMB",
        "MKI67",
        "TOP2A",
    ]

    means = standardized[nonconstant].mean(axis=0)

    assert np.allclose(
        means.to_numpy(),
        np.zeros(len(nonconstant)),
        atol=1e-12,
    )

    assert np.allclose(
        standardized["ESR1"].to_numpy(),
        np.zeros(expression.shape[0]),
    )


def test_scoring_and_patient_order():
    expression = make_expression()

    result = score_gene_sets(
        expression,
        make_gene_sets(),
        min_genes=2,
        min_coverage=0.50,
    )

    assert result.scores.shape == (4, 2)

    assert result.scores.index.tolist() == [
        "P001",
        "P002",
        "P003",
        "P004",
    ]

    assert result.scores.columns.tolist() == [
        "CYTOTOXIC",
        "PROLIFERATION",
    ]

    assert np.isfinite(
        result.scores.to_numpy(dtype=float)
    ).all()


def test_expected_cytotoxic_direction():
    expression = make_expression()

    result = score_gene_sets(
        expression,
        {
            "CYTOTOXIC": ["CD8A", "CD8B", "GZMB"],
        },
        min_genes=3,
        min_coverage=1.0,
    )

    scores = result.scores["CYTOTOXIC"]

    assert scores.loc["P001"] < scores.loc["P002"]
    assert scores.loc["P002"] < scores.loc["P003"]
    assert scores.loc["P003"] < scores.loc["P004"]


def test_deterministic_output():
    expression = make_expression()
    gene_sets = make_gene_sets()

    first = score_gene_sets(
        expression,
        gene_sets,
        min_genes=2,
        min_coverage=0.50,
    )

    second = score_gene_sets(
        expression,
        gene_sets,
        min_genes=2,
        min_coverage=0.50,
    )

    pd.testing.assert_frame_equal(
        first.scores,
        second.scores,
    )

    pd.testing.assert_frame_equal(
        first.coverage,
        second.coverage,
    )


def test_label_independent_interface():
    signature = inspect.signature(score_gene_sets)

    forbidden = {
        "treatment",
        "arm",
        "outcome",
        "pcr",
        "response",
        "label",
        "y",
    }

    parameters = {
        name.lower()
        for name in signature.parameters
    }

    assert parameters.isdisjoint(forbidden)


def test_summary():
    result = score_gene_sets(
        make_expression(),
        make_gene_sets(),
        min_genes=2,
        min_coverage=0.50,
    )

    summary = summarize_representations(result)

    assert summary["patients"] == 4
    assert summary["representations"] == 2
    assert summary["gene_sets_evaluated"] == 3
    assert summary["gene_sets_retained"] == 2
    assert summary["gene_sets_dropped"] == 1


def test_gmt_loader():
    content = (
        "SET_A\tdescription\tCD8A\tCD8B\tGZMB\n"
        "SET_B\tdescription\tMKI67\tTOP2A\tMKI67\n"
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.gmt"
        path.write_text(content, encoding="utf-8")

        gene_sets = load_gmt(path)

    assert gene_sets["SET_A"] == (
        "CD8A",
        "CD8B",
        "GZMB",
    )

    assert gene_sets["SET_B"] == (
        "MKI67",
        "TOP2A",
    )


def run_all_tests():
    tests = [
        ("expression validation", test_expression_validation),
        ("gene-set validation", test_gene_set_validation),
        ("case-insensitive matching", test_case_insensitive_matching),
        ("coverage filtering", test_coverage_filtering),
        ("gene z-scoring", test_zscore_behavior),
        ("scoring + patient order", test_scoring_and_patient_order),
        ("biological score direction", test_expected_cytotoxic_direction),
        ("deterministic output", test_deterministic_output),
        ("label-independent interface", test_label_independent_interface),
        ("representation summary", test_summary),
        ("GMT loading", test_gmt_loader),
    ]

    print("=== HERMES 2.0 Biological Representation Tests ===")

    for name, test in tests:
        test()
        print(f"PASS: {name}")

    print()
    print("==========================================")
    print("ALL BIOLOGICAL REPRESENTATION TESTS PASSED")
    print("==========================================")


if __name__ == "__main__":
    run_all_tests()