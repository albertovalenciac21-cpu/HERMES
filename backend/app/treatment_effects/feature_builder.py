"""
HERMES 2.0 — Modeling Feature Builder
=====================================

Builds the canonical patient-level dataset for treatment-effect modeling:

    X = baseline biological state
    T = randomized treatment assignment
    Y = pathologic complete response

NeoTRIP encoding
----------------
Treatment:
    0 = CT
    1 = CT/A

Outcome:
    0 = residual disease
    1 = pCR

The biological representation uses the MSigDB Hallmark collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.treatment_effects.cohort_loader import (
    load_neotrip_baseline,
)
from backend.app.treatment_effects.preprocessing import (
    preprocess_neotrip_baseline,
)
from backend.app.treatment_effects.representations import (
    load_gmt,
    score_gene_sets,
)


DEFAULT_HALLMARK_GMT = Path(
    "data/hermes2/gene_sets/"
    "h.all.v2026.1.Hs.symbols.gmt"
)


@dataclass
class TreatmentEffectDataset:
    X: pd.DataFrame
    T: pd.Series
    Y: pd.Series
    metadata: pd.DataFrame
    summary: dict[str, Any]

    @property
    def n_patients(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    def validate(self) -> None:
        if self.X.empty:
            raise ValueError(
                "Feature matrix X is empty."
            )

        if self.X.index.duplicated().any():
            raise ValueError(
                "Duplicate patients detected in X."
            )

        if self.X.columns.duplicated().any():
            raise ValueError(
                "Duplicate biological features detected."
            )

        if self.X.isna().any().any():
            raise ValueError(
                "Missing values detected in X."
            )

        if not np.isfinite(
            self.X.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Non-finite values detected in X."
            )

        if not self.X.index.equals(self.T.index):
            raise ValueError(
                "Treatment vector is not aligned with X."
            )

        if not self.X.index.equals(self.Y.index):
            raise ValueError(
                "Outcome vector is not aligned with X."
            )

        if not self.X.index.equals(
            self.metadata.index
        ):
            raise ValueError(
                "Metadata is not aligned with X."
            )

        if set(self.T.unique()) != {0, 1}:
            raise ValueError(
                "Treatment vector must contain "
                "both 0 and 1."
            )

        if set(self.Y.unique()) != {0, 1}:
            raise ValueError(
                "Outcome vector must contain "
                "both 0 and 1."
            )


def _build_treatment_vector(
    cohort,
) -> pd.Series:
    patient_ids = [
        record.patient_id
        for record in cohort.records
    ]

    treatment_values = [
        int(
            record.treatment
            .treatment_indicator
        )
        for record in cohort.records
    ]

    return pd.Series(
        treatment_values,
        index=pd.Index(
            patient_ids,
            name="Patient_ID",
        ),
        name="T",
        dtype=int,
    )


def _build_outcome_vector(
    cohort,
) -> pd.Series:
    patient_ids = [
        record.patient_id
        for record in cohort.records
    ]

    outcome_values = [
        int(
            record.outcome.binary_outcome
        )
        for record in cohort.records
    ]

    return pd.Series(
        outcome_values,
        index=pd.Index(
            patient_ids,
            name="Patient_ID",
        ),
        name="Y",
        dtype=int,
    )


def _build_metadata(
    cohort,
    T: pd.Series,
    Y: pd.Series,
) -> pd.DataFrame:
    patient_ids = T.index

    metadata = pd.DataFrame(
        index=patient_ids,
    )

    metadata["treatment_label"] = T.map(
        {
            0: "CT",
            1: "CT/A",
        }
    )

    metadata["outcome_label"] = Y.map(
        {
            0: "RD",
            1: "pCR",
        }
    )

    metadata["tnbc_type"] = [
        str(
            record.metadata.get(
                "tnbc_type",
                "",
            )
        )
        for record in cohort.records
    ]

    return metadata


def build_treatment_effect_dataset(
    hallmark_gmt_path: Path = DEFAULT_HALLMARK_GMT,
    min_genes: int = 3,
    min_coverage: float = 0.50,
) -> TreatmentEffectDataset:
    """
    Construct the canonical HERMES 2.0 NeoTRIP dataset.
    """

    cohort = load_neotrip_baseline()

    processed = (
        preprocess_neotrip_baseline()
    )

    gene_sets = load_gmt(
        hallmark_gmt_path
    )

    representations = score_gene_sets(
        processed.expression,
        gene_sets,
        min_genes=min_genes,
        min_coverage=min_coverage,
    )

    X = representations.scores.copy()

    X.index = pd.Index(
        X.index.astype(str),
        name="Patient_ID",
    )

    T = _build_treatment_vector(
        cohort
    )

    Y = _build_outcome_vector(
        cohort
    )

    expected_patient_order = pd.Index(
        [
            record.patient_id
            for record in cohort.records
        ],
        name="Patient_ID",
    )

    if not X.index.equals(
        expected_patient_order
    ):
        raise ValueError(
            "Biological representation matrix "
            "does not match canonical NeoTRIP "
            "patient ordering."
        )

    T = T.loc[X.index]
    Y = Y.loc[X.index]

    metadata = _build_metadata(
        cohort,
        T,
        Y,
    )

    treatment_counts = (
        metadata["treatment_label"]
        .value_counts()
        .to_dict()
    )

    outcome_counts = (
        metadata["outcome_label"]
        .value_counts()
        .to_dict()
    )

    pcr_rate_by_arm = (
        pd.DataFrame(
            {
                "T": T,
                "Y": Y,
            }
        )
        .groupby("T")["Y"]
        .mean()
        .rename(
            index={
                0: "CT",
                1: "CT/A",
            }
        )
        .to_dict()
    )

    summary = {
        "patients": int(
            X.shape[0]
        ),
        "features": int(
            X.shape[1]
        ),
        "treatment_counts": (
            treatment_counts
        ),
        "outcome_counts": (
            outcome_counts
        ),
        "pcr_rate_by_arm": {
            key: float(value)
            for key, value
            in pcr_rate_by_arm.items()
        },
        "hallmark_sets_loaded": (
            len(gene_sets)
        ),
        "hallmark_sets_retained": (
            representations
            .n_representations
        ),
        "minimum_gene_set_coverage": (
            float(min_coverage)
        ),
    }

    dataset = TreatmentEffectDataset(
        X=X,
        T=T,
        Y=Y,
        metadata=metadata,
        summary=summary,
    )

    dataset.validate()

    return dataset


def summarize_dataset(
    dataset: TreatmentEffectDataset,
) -> None:
    print(
        "=== HERMES 2.0 "
        "Treatment-Effect Dataset ==="
    )

    print(
        f"Patients: "
        f"{dataset.n_patients}"
    )

    print(
        f"Biological features: "
        f"{dataset.n_features}"
    )

    print(
        f"Feature matrix: "
        f"{dataset.X.shape}"
    )

    print()

    print("Treatment counts:")

    print(
        dataset.metadata[
            "treatment_label"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print("Outcome counts:")

    print(
        dataset.metadata[
            "outcome_label"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        "pCR rate by treatment arm:"
    )

    rates = (
        pd.DataFrame(
            {
                "Treatment": (
                    dataset.metadata[
                        "treatment_label"
                    ]
                ),
                "pCR": dataset.Y,
            }
        )
        .groupby(
            "Treatment"
        )["pCR"]
        .agg(
            [
                "count",
                "sum",
                "mean",
            ]
        )
    )

    print(rates)

    print()

    print(
        "Hallmark sets loaded:",
        dataset.summary[
            "hallmark_sets_loaded"
        ],
    )

    print(
        "Hallmark sets retained:",
        dataset.summary[
            "hallmark_sets_retained"
        ],
    )


def main() -> None:
    dataset = (
        build_treatment_effect_dataset()
    )

    summarize_dataset(
        dataset
    )


if __name__ == "__main__":
    main()