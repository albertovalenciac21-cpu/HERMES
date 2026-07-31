from fastapi import APIRouter


router = APIRouter(tags=["System"])


@router.get("/")
def home():
    return {
        "project": "Project HERMES",
        "acronym": (
            "High-throughput Engine for Research in "
            "Multi-omic Evaluation and Stratification"
        ),
        "version": "1.1.0",
        "status": "Running",
        "message": "Welcome to the Project HERMES API!",
        "research_use_only": True,
    }


@router.get("/health")
def health():
    return {
        "project": "Project HERMES",
        "version": "1.1.0",
        "status": "healthy",
    }