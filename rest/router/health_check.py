
from fastapi import APIRouter, Response


router = APIRouter()


@router.get("/inference" + "/monitor/health_check", tags=["healthcheck"])
async def health_check():
    return Response(status_code=200)
