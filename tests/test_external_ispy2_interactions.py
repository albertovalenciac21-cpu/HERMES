"""
HERMES 2.0
Locked I-SPY2 Pathway × Pembrolizumab Interaction Tests
=======================================================
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.treatment_effects.external_ispy2_interactions import (
    export_locked_ispy2_interactions,
    run_locked_ispy2_interactions,
)
from backend.app.treatment_effects.external_ispy2_pathways import (
    ISPY2PathwayRepresentation,
)
from backend.app.treatment_effects.external_validation_plan import (
    LOCKED_EXTERNAL_VALIDATION_PLAN,
)


LOCKED = tuple(
    LOCKED_EXTERNAL_VALIDATION_PLAN.locked_negative_pathway_hypotheses
)


def _fixture_representation(
    *,
    seed: int = 1234,
) -> ISPY2PathwayRepresentation:
    rng = np.random.default_rng(seed)

    n_control = 90
    n_treated = 60
    n = n_control + n_treated

    ids = pd.Index(
        [f"ISPY2_TEST_{i:03d}" for i in range(n)],
        name="Patient_ID",
    )

    treatment = pd.Series(
        np.r_[
            np.zeros(n_control, dtype=int),
            np.ones(n_treated, dtype=int),
        ],
        index=ids,
        name="T",
    )

    scores = pd.DataFrame(
        rng.normal(size=(n, len(LOCKED))),
        index=ids,
        columns=list(LOCKED),
    )

    # Generate an outcome with a negative treatment interaction for the first
    # locked pathway while preserving overlap in all treatment/outcome strata.
    linear = (
        -0.8
        + 1.2 * treatment.to_numpy(dtype=float)
        + 0.25 * scores[LOCKED[0]].to_numpy(dtype=float)
        - 0.75
        * treatment.to_numpy(dtype=float)
        * scores[LOCKED[0]].to_numpy(dtype=float)
    )

    probability = 1.0 / (1.0 + np.exp(-linear))

    outcome = pd.Series(
        rng.binomial(1, probability),
        index=ids,
        name="Y",
        dtype=int,
    )

    # Make the fixture deterministic with all four T x Y strata.
    for treatment_value in (0, 1):
        subset = ids[treatment.eq(treatment_value)]
        outcome.loc[subset[:4]] = 0
        outcome.loc[subset[4:8]] = 1

    clinical = pd.DataFrame(
        {
            "Patient_ID": ids,
            "HR": 0,
            "HER2": 0,
            "pCR": outcome.to_numpy(dtype=int),
            "Arm": np.where(
                treatment.to_numpy(dtype=int) == 1,
                "Paclitaxel + Pembrolizumab",
                "Paclitaxel",
            ),
        },
        index=ids,
    )

    coverage = pd.DataFrame(
        {
            "requested_genes": 100,
            "matched_genes": 90,
            "coverage_fraction": 0.90,
            "retained": True,
        },
        index=pd.Index(
            LOCKED,
            name="gene_set",
        ),
    )

    representation = ISPY2PathwayRepresentation(
        scores=scores,
        coverage=coverage,
        treatment=treatment,
        outcome=outcome,
        clinical=clinical,
        locked_pathways=LOCKED,
        summary={
            "patients": n,
            "locked_pathways_retained": True,
        },
    )
    representation.validate()

    return representation


def test_external_interaction_analysis_uses_exactly_locked_family() -> None:
    result = run_locked_ispy2_interactions(
        representation=_fixture_representation()
    )

    assert len(result.interaction_table) == 7
    assert set(result.interaction_table["feature"]) == set(LOCKED)
    assert tuple(result.locked_pathways) == LOCKED
    assert result.summary["locked_hypotheses_requested"] == 7
    assert result.summary["locked_hypotheses_analyzed"] == 7


def test_external_interaction_fdr_is_computed_over_seven_locked_pathways() -> None:
    result = run_locked_ispy2_interactions(
        representation=_fixture_representation()
    )

    table = result.interaction_table

    assert table["interaction_fdr"].between(0.0, 1.0).all()
    assert np.all(
        table["interaction_fdr"].to_numpy(dtype=float)
        + 1e-12
        >= table["interaction_p_value"].to_numpy(dtype=float)
    )

    assert np.isclose(
        result.summary["fdr_threshold"],
        LOCKED_EXTERNAL_VALIDATION_PLAN.hypothesis_fdr_threshold,
    )


def test_external_interaction_reports_locked_direction_concordance() -> None:
    result = run_locked_ispy2_interactions(
        representation=_fixture_representation()
    )

    table = result.interaction_table.set_index("feature")

    assert (
        table.loc[
            LOCKED[0],
            "interaction_coefficient",
        ]
        < 0.0
    )

    assert bool(
        table.loc[
            LOCKED[0],
            "direction_concordant_with_locked_hypothesis",
        ]
    )

    assert set(
        result.interaction_table[
            "expected_interaction_direction"
        ]
    ) == {
        "greater_benefit_with_lower_pathway"
    }


def test_external_interaction_rejects_mismatched_pathway_lock() -> None:
    representation = _fixture_representation()

    representation.locked_pathways = tuple(
        reversed(LOCKED)
    )

    try:
        run_locked_ispy2_interactions(
            representation=representation
        )
    except ValueError as exc:
        assert "lock" in str(exc).lower()
    else:
        raise AssertionError(
            "Mismatched external pathway lock was not rejected."
        )


def test_external_interaction_summary_preserves_scientific_guardrails() -> None:
    result = run_locked_ispy2_interactions(
        representation=_fixture_representation()
    )

    assert result.summary["posthoc_pathway_selection_used"] is False
    assert (
        result.summary["direct_neotrip_raw_scale_transport_used"]
        is False
    )
    assert result.summary["clinical_claims_allowed"] is False
    assert len(
        result.summary["external_validation_plan_sha256"]
    ) == 64


def test_external_interaction_export_contract(tmp_path) -> None:
    result = run_locked_ispy2_interactions(
        representation=_fixture_representation()
    )

    generated = export_locked_ispy2_interactions(
        result,
        output_dir=tmp_path,
    )

    assert set(generated) == {
        "ranked_results",
        "hypothesis_order_results",
        "compact_results",
        "summary",
        "manifest",
    }

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0

    exported = pd.read_csv(
        generated["hypothesis_order_results"]
    )

    assert exported["feature"].tolist() == list(LOCKED)
    assert len(exported) == 7
    assert exported["locked_external_hypothesis"].all()
    assert not exported["posthoc_pathway_selection_used"].any()

    manifest = json.loads(
        generated["manifest"].read_text(
            encoding="utf-8"
        )
    )

    assert manifest["locked_pathway_count"] == 7
    assert manifest[
        "all_results_exported_regardless_of_significance"
    ] is True
    assert manifest["posthoc_pathway_selection_used"] is False