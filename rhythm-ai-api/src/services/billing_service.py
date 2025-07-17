import stripe
from fastapi import HTTPException

stripe.api_key = "sk_live_your_stripe_key_here"

AD_REMOVAL_PRICE = 100  # $1 in cents

def create_checkout_session(user_id: str, success_url: str, cancel_url: str):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {"name": "Remove Ads"},
                    "unit_amount": AD_REMOVAL_PRICE,
                },
                "quantity": 1,
            }],
            mode="payment",
            metadata={"user_id": user_id, "feature": "remove_ads"},
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return session.url
    except Exception as e:
        raise HTTPException(500, str(e))