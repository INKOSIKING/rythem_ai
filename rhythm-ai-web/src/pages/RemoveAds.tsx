import React from "react";

export default function RemoveAds() {
  const handleStripe = async () => {
    const res = await fetch("/api/billing/remove-ads/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: "user123",
        success_url: window.location.origin + "/ads-removed",
        cancel_url: window.location.origin + "/remove-ads",
      }),
    });
    const data = await res.json();
    window.location.href = data.checkout_url;
  };

  const handleCrypto = async () => {
    const res = await fetch("/api/billing/crypto-link/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: "user123" }),
    });
    const data = await res.json();
    window.open(data.crypto_url, "_blank");
  };

  return (
    <div>
      <h2 className="text-xl font-bold mb-2">Remove Ads ($1 one-time)</h2>
      <button onClick={handleStripe} className="btn btn-primary mr-4">
        Pay with Credit Card
      </button>
      <button onClick={handleCrypto} className="btn btn-secondary">
        Pay with Crypto
      </button>
    </div>
  );
}