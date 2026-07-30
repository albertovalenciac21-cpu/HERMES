from fastapi import APIRouter

router = APIRouter(tags=["System"])


@router.get("/")
def home():
    return {
        "project": "Trojan Horse",
        "status": "Running",
        "message": "Welcome to the Trojan Horse API!",
    }


@router.get("/health")
def health():
    return {
        "status": "healthy",
    }