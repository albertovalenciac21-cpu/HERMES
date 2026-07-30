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
    title="Project Trojan Horse",
    description=(
        "AI-assisted precision oncology platform "
        "for Triple-Negative Breast Cancer"
    ),
    version="0.6.0",
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
