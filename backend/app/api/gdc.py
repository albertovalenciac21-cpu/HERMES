from fastapi import APIRouter

from backend.app.services.gdc import get_projects

router = APIRouter(
    prefix="/gdc",
    tags=["GDC"],
)


@router.get("/projects")
def gdc_projects():
    return get_projects()