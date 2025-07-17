import React from "react";

export default function AdsBanner() {
  // Embed Google AdSense or another network here
  return (
    <div className="my-4">
      {/* Example: Replace with your ad code or use a script loader */}
      <ins className="adsbygoogle"
           style={{ display: "block" }}
           data-ad-client="ca-pub-xxxxxxxxxxxxxxxx"
           data-ad-slot="xxxxxxx"
           data-ad-format="auto"></ins>
      <script>
        {`(adsbygoogle = window.adsbygoogle || []).push({});`}
      </script>
    </div>
  );
}