from backend.app.treatment_effects.schema import (
    ClinicalOutcome,
    DataProvenance,
    DiseaseSetting,
    HermesPatientRecord,
    MolecularState,
    OutcomeType,
    SampleTimepoint,
    Treatment,
    TreatmentEffectEstimate,
    TreatmentGroup,
)


def test_hermes_patient_record():
    treatment = Treatment(
        treatment_indicator=1,
        group=TreatmentGroup.ICI,
        ici_agent="atezolizumab",
        chemotherapy=["carboplatin"],
        treatment_arm_label="carboplatin + atezolizumab",
    )

    outcome = ClinicalOutcome(
        outcome_type=OutcomeType.RESPONSE,
        binary_outcome=1,
        response_category="response",
    )

    molecular_state = MolecularState(
        pathway_features={
            "interferon_gamma": 1.25,
            "antigen_presentation": 0.84,
        },
        immune_features={
            "cd8_t_cell_score": 0.91,
        },
        tumor_features={
            "proliferation_score": 0.63,
        },
    )

    provenance = DataProvenance(
        cohort="example_randomized_tnbc_trial",
        platform="RNA-seq",
        source="test",
    )

    patient = HermesPatientRecord(
        patient_id="HERMES_TEST_001",
        sample_id="HERMES_SAMPLE_001",
        disease_setting=DiseaseSetting.METASTATIC,
        sample_timepoint=SampleTimepoint.PRETREATMENT,
        treatment=treatment,
        outcome=outcome,
        molecular_state=molecular_state,
        provenance=provenance,
    )

    patient.validate()

    assert patient.patient_id == "HERMES_TEST_001"
    assert patient.treatment.treatment_indicator == 1
    assert patient.outcome.binary_outcome == 1


def test_treatment_effect_calculation():
    estimate = TreatmentEffectEstimate(
        patient_id="HERMES_TEST_001",
        predicted_outcome_treated=0.72,
        predicted_outcome_control=0.41,
        treatment_effect=0.31,
        uncertainty=0.08,
        lower_bound=0.15,
        upper_bound=0.47,
        out_of_distribution_score=0.12,
        model_version="HERMES-2.0-test",
    )

    estimate.validate()

    assert abs(estimate.treatment_effect - 0.31) < 1e-8


def test_invalid_treatment_effect_fails():
    estimate = TreatmentEffectEstimate(
        patient_id="HERMES_TEST_002",
        predicted_outcome_treated=0.80,
        predicted_outcome_control=0.50,
        treatment_effect=0.10,
    )

    try:
        estimate.validate()
    except ValueError:
        return

    raise AssertionError(
        "Validation should fail when treatment_effect "
        "does not equal treated minus control prediction."
    )


if __name__ == "__main__":
    test_hermes_patient_record()
    test_treatment_effect_calculation()
    test_invalid_treatment_effect_fails()

    print("HERMES 2.0 treatment-effect schema tests passed.")