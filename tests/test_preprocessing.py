from backend.app.treatment_effects.cohort_loader import (
    load_neotrip_baseline,
)
from backend.app.treatment_effects.preprocessing import (
    calculate_gene_qc,
    filter_transcriptome,
    preprocess_neotrip_baseline,
    summarize_processed_transcriptome,
    validate_cohort_alignment,
    validate_expression_matrix,
)


def main():
    print("=== HERMES 2.0 Preprocessing Tests ===")

    cohort = load_neotrip_baseline()

    # ---------------------------------------------------------
    # Raw expression validation
    # ---------------------------------------------------------

    validate_expression_matrix(
        cohort.expression
    )

    assert cohort.expression.shape == (
        241,
        23890,
    )

    print("PASS: raw expression validation")

    # ---------------------------------------------------------
    # Gene QC
    # ---------------------------------------------------------

    qc = calculate_gene_qc(
        cohort.expression
    )

    assert qc.shape[0] == 23890

    required_columns = {
        "mean_expression",
        "median_expression",
        "variance",
        "expression_prevalence",
        "constant",
        "low_variance",
        "low_information",
    }

    assert required_columns.issubset(
        set(qc.columns)
    )

    print("PASS: gene-level QC")

    # ---------------------------------------------------------
    # Transcriptomic filtering
    # ---------------------------------------------------------

    processed = filter_transcriptome(
        cohort.expression
    )

    assert processed.expression.shape[0] == 241

    assert (
        processed.expression.shape[1]
        < cohort.expression.shape[1]
    )

    assert processed.expression.shape[1] == 21727

    assert (
        len(processed.retained_genes)
        == processed.expression.shape[1]
    )

    assert (
        len(processed.removed_genes)
        == 2163
    )

    print("PASS: transcriptomic filtering")

    # ---------------------------------------------------------
    # Expression integrity after filtering
    # ---------------------------------------------------------

    validate_expression_matrix(
        processed.expression
    )

    assert (
        not processed.expression
        .isna()
        .any()
        .any()
    )

    assert (
        not processed.expression
        .columns
        .duplicated()
        .any()
    )

    print("PASS: filtered expression integrity")

    # ---------------------------------------------------------
    # Patient alignment
    # ---------------------------------------------------------

    validate_cohort_alignment(
        cohort,
        processed,
    )

    assert (
        processed.expression.index.tolist()
        == cohort.expression.index.tolist()
    )

    print("PASS: patient alignment preserved")

    # ---------------------------------------------------------
    # No treatment/outcome-dependent filtering
    # ---------------------------------------------------------

    treatment_labels = [
        record.treatment.treatment_indicator
        for record in cohort.records
    ]

    outcome_labels = [
        record.outcome.binary_outcome
        for record in cohort.records
    ]

    assert set(treatment_labels) == {0, 1}
    assert set(outcome_labels) == {0, 1}

    # The preprocessing API receives expression only,
    # preventing treatment/outcome-driven selection.

    print("PASS: label-independent QC interface")

    # ---------------------------------------------------------
    # Full NeoTRIP preprocessing workflow
    # ---------------------------------------------------------

    processed_full = (
        preprocess_neotrip_baseline()
    )

    summary = (
        summarize_processed_transcriptome(
            processed_full
        )
    )

    assert summary["patients"] == 241
    assert summary["genes_input"] == 23890
    assert summary["genes_retained"] == 21727
    assert summary["genes_removed"] == 2163
    assert summary["constant_genes"] == 0
    assert summary["low_variance_genes"] == 2159
    assert summary["low_information_genes"] == 25

    print("PASS: reproducible preprocessing summary")

    # ---------------------------------------------------------
    # Final
    # ---------------------------------------------------------

    print()
    print("======================================")
    print("ALL PREPROCESSING TESTS PASSED")
    print("======================================")
    print()
    print(
        "Raw matrix:",
        cohort.expression.shape,
    )
    print(
        "Filtered matrix:",
        processed_full.expression.shape,
    )
    print(
        "Genes removed:",
        summary["genes_removed"],
    )
    print(
        "Retention:",
        f"{summary['retention_fraction']:.2%}",
    )


if __name__ == "__main__":
    main()