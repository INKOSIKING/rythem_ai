from fastapi import APIRouter, Request, Depends

router = APIRouter(prefix="/ads", tags=["ads"])

@router.get("/status/")
async def ads_status(user_id: str):
    # Query DB or cache: has user paid? (stubbed)
    if user_id in {"user_with_no_ads"}:
        return {"show_ads": False}
    return {"show_ads": True}