import requests

CRYPTO_ADDRESS = "your_wallet_address_here"
CRYPTO_AMOUNT_USD = 1.00

def get_crypto_payment_link(user_id: str):
    # Example: Use Coingate, Coinbase Commerce, or NowPayments API
    payload = {
        "price_amount": CRYPTO_AMOUNT_USD,
        "price_currency": "USD",
        "pay_currency": "USDT",  # or BTC/ETH, etc.
        "order_id": user_id,
        "receive_address": CRYPTO_ADDRESS,
        "title": "Remove Ads",
        "callback_url": "https://yourapi.com/billing/crypto-callback"
    }
    # Replace with real API call (documented in their docs)
    # resp = requests.post("https://api.nowpayments.io/v1/invoice", json=payload, headers=...)
    # return resp.json()["pay_url"]
    return "https://your-crypto-payments.com/pay?order=" + user_id