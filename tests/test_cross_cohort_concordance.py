"""
HERMES 2.0
NeoTRIP ↔ I-SPY2 Cross-Cohort Concordance Tests
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from backend.app.treatment_effects.cross_cohort_concordance import (
    _exact_sign_concordance_p_value,
    export_cross_cohort_concordance,
    run_cross_cohort_concordance,
)
from backend.app.treatment_effects.external_validation_plan import (
    LOCKED_EXTERNAL_VALIDATION_PLAN,
)


LOCKED = tuple(
    LOCKED_EXTERNAL_VALIDATION_PLAN.locked_negative_pathway_hypotheses
)


def _write_artifact(
    path,
    coefficients,
) -> None:
    rows = []

    for pathway, beta in zip(
        LOCKED,
        coefficients,
        strict=True,
    ):
        se = 0.25
        rows.append(
            {
                "feature": pathway,
                "interaction_coefficient": beta,
                "interaction_standard_error": se,
                "interaction_odds_ratio": np.exp(beta),
                "interaction_or_ci_lower": np.exp(
                    beta - 1.96 * se
                ),
                "interaction_or_ci_upper": np.exp(
                    beta + 1.96 * se
                ),
                "interaction_p_value": 0.20,
                "interaction_fdr": 0.40,
            }
        )

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
    )


def test_exact_sign_concordance_probability() -> None:
    # P[X >= 5] for X ~ Binomial(7, 0.5) = 29 / 128.
    assert np.isclose(
        _exact_sign_concordance_p_value(5, 7),
        29.0 / 128.0,
    )


def test_cross_cohort_uses_exactly_locked_seven_pathways(
    tmp_path,
) -> None:
    neo = tmp_path / "neo.csv"
    ext = tmp_path / "ext.csv"

    _write_artifact(
        neo,
        [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2],
    )
    _write_artifact(
        ext,
        [-0.7, -0.6, -0.5, -0.4, -0.3, 0.2, 0.1],
    )

    result = run_cross_cohort_concordance(
        neotrip_modifier_path=neo,
        ispy2_interaction_path=ext,
    )

    assert len(result.comparison_table) == 7
    assert result.comparison_table["feature"].tolist() == list(LOCKED)
    assert result.summary["locked_pathways"] == 7
    assert (
        result.summary["cross_cohort_sign_concordant_count"]
        == 5
    )


def test_cross_cohort_reports_coefficient_correlations(
    tmp_path,
) -> None:
    neo = tmp_path / "neo.csv"
    ext = tmp_path / "ext.csv"

    beta = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2]

    _write_artifact(neo, beta)
    _write_artifact(ext, beta)

    result = run_cross_cohort_concordance(
        neotrip_modifier_path=neo,
        ispy2_interaction_path=ext,
    )

    assert np.isclose(
        result.summary[
            "pearson_interaction_coefficient_correlation"
        ],
        1.0,
    )
    assert np.isclose(
        result.summary[
            "spearman_interaction_coefficient_correlation"
        ],
        1.0,
    )


def test_cross_cohort_rejects_missing_locked_pathway(
    tmp_path,
) -> None:
    neo = tmp_path / "neo.csv"
    ext = tmp_path / "ext.csv"

    _write_artifact(
        neo,
        [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2],
    )
    _write_artifact(
        ext,
        [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
    )

    table = pd.read_csv(ext)
    table = table.iloc[:-1].copy()
    table.to_csv(ext, index=False)

    try:
        run_cross_cohort_concordance(
            neotrip_modifier_path=neo,
            ispy2_interaction_path=ext,
        )
    except ValueError as exc:
        assert "missing locked pathways" in str(exc).lower()
    else:
        raise AssertionError(
            "Missing locked pathway was not rejected."
        )


def test_cross_cohort_summary_preserves_guardrails(
    tmp_path,
) -> None:
    neo = tmp_path / "neo.csv"
    ext = tmp_path / "ext.csv"

    _write_artifact(
        neo,
        [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2],
    )
    _write_artifact(
        ext,
        [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
    )

    result = run_cross_cohort_concordance(
        neotrip_modifier_path=neo,
        ispy2_interaction_path=ext,
    )

    assert result.summary["posthoc_pathway_selection_used"] is False
    assert (
        result.summary["direct_raw_scale_coefficient_transport_claimed"]
        is False
    )
    assert result.summary["clinical_claims_allowed"] is False
    assert len(
        result.summary["external_validation_plan_sha256"]
    ) == 64


def test_cross_cohort_export_contract(
    tmp_path,
) -> None:
    neo = tmp_path / "neo.csv"
    ext = tmp_path / "ext.csv"

    _write_artifact(
        neo,
        [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2],
    )
    _write_artifact(
        ext,
        [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1],
    )

    result = run_cross_cohort_concordance(
        neotrip_modifier_path=neo,
        ispy2_interaction_path=ext,
    )

    generated = export_cross_cohort_concordance(
        result,
        output_dir=tmp_path / "out",
    )

    assert set(generated) == {
        "comparison",
        "summary",
        "manifest",
    }

    for path in generated.values():
        assert path.exists()
        assert path.stat().st_size > 0

    manifest = json.loads(
        generated["manifest"].read_text(
            encoding="utf-8"
        )
    )

    assert manifest["locked_pathway_count"] == 7
    assert manifest["all_locked_pathways_reported"] is True
    assert manifest["posthoc_pathway_selection_used"] is False