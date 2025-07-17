from fastapi import APIRouter, Request
from src.services.billing_service import create_checkout_session
from src.services.crypto_service import get_crypto_payment_link

router = APIRouter(prefix="/billing", tags=["billing"])

@router.post("/remove-ads/")
async def remove_ads(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    success_url = data.get("success_url")
    cancel_url = data.get("cancel_url")
    url = create_checkout_session(user_id, success_url, cancel_url)
    return {"checkout_url": url}

@router.post("/crypto-link/")
async def crypto_link(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    url = get_crypto_payment_link(user_id)
    return {"crypto_url": url}