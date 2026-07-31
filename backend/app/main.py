from fastapi import FastAPI

from backend.app.api import (
    ai,
    breast,
    cohort,
    gdc,
    health,
    mutations,
    patient,
    rna,
    tnbc,
)


app = FastAPI(
    title="Project HERMES",
    description=(
        "HERMES — High-throughput Engine for Research in Multi-omic "
        "Evaluation and Stratification. An AI-assisted precision oncology "
        "platform for automated molecular cohort construction, multi-omic "
        "data integration, machine learning, explainability, and "
        "research-oriented prediction in triple-negative breast cancer."
    ),
    version="1.1.0",
)


app.include_router(health.router)
app.include_router(gdc.router)
app.include_router(breast.router)
app.include_router(rna.router)
app.include_router(mutations.router)
app.include_router(patient.router)
app.include_router(cohort.router)
app.include_router(ai.router)
app.include_router(tnbc.router)