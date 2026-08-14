from backend.app.treatment_effects.cohort_loader import (
    load_neotrip_baseline,
    summarize_neotrip_cohort,
)


def main():
    print("=== HERMES 2.0 NeoTRIP Loader Tests ===")

    cohort = load_neotrip_baseline()
    summary = summarize_neotrip_cohort(cohort)

    # ---------------------------------------------------------
    # Cohort dimensions
    # ---------------------------------------------------------

    assert cohort.n_patients == 241
    assert cohort.expression.shape[0] == 241
    assert cohort.n_genes == 23890

    print("PASS: cohort dimensions")

    # ---------------------------------------------------------
    # Patient-expression alignment
    # ---------------------------------------------------------

    record_ids = [
        record.patient_id
        for record in cohort.records
    ]

    expression_ids = cohort.expression.index.tolist()

    assert record_ids == expression_ids
    assert len(record_ids) == len(set(record_ids))

    print("PASS: patient-expression alignment")

    # ---------------------------------------------------------
    # Treatment randomization
    # ---------------------------------------------------------

    treatment = [
        record.treatment.treatment_indicator
        for record in cohort.records
    ]

    assert set(treatment) == {0, 1}
    assert treatment.count(0) == 122
    assert treatment.count(1) == 119

    print("PASS: randomized treatment encoding")

    # ---------------------------------------------------------
    # Clinical outcomes
    # ---------------------------------------------------------

    outcomes = [
        record.outcome.binary_outcome
        for record in cohort.records
    ]

    assert set(outcomes) == {0, 1}
    assert outcomes.count(1) == 121
    assert outcomes.count(0) == 120

    print("PASS: pCR outcome encoding")

    # ---------------------------------------------------------
    # Expression integrity
    # ---------------------------------------------------------

    assert not cohort.expression.index.duplicated().any()
    assert not cohort.expression.columns.duplicated().any()

    assert not cohort.expression.isna().any().any()

    print("PASS: expression integrity")

    # ---------------------------------------------------------
    # Summary consistency
    # ---------------------------------------------------------

    assert summary["patients"] == 241
    assert summary["genes"] == 23890

    assert summary["treatment_counts"]["CT"] == 122
    assert summary["treatment_counts"]["CT/A"] == 119

    assert summary["outcome_counts"]["pCR"] == 121
    assert summary["outcome_counts"]["RD"] == 120

    print("PASS: cohort summary")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("======================================")
    print("ALL NEOTRIP LOADER TESTS PASSED")
    print("======================================")
    print()
    print(f"Patients: {cohort.n_patients}")
    print(f"Genes: {cohort.n_genes}")
    print(
        "Expression matrix:",
        cohort.expression.shape,
    )


if __name__ == "__main__":
    main()