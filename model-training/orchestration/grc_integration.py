import json
import requests
from utils.audit_logging import log_event

def send_compliance_report(report_file, grc_api_url, api_key=None):
    """
    Send the compliance export file to a GRC (Governance, Risk, Compliance) system.
    """
    with open(report_file, "rb") as f:
        files = {"file": (report_file, f, "text/csv")}
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            resp = requests.post(grc_api_url, files=files, headers=headers, timeout=30)
            resp.raise_for_status()
            log_event("grc_report_sent", user="system", extra=report_file)
            print(f"Compliance report sent to GRC: {grc_api_url}")
            return True
        except Exception as e:
            log_event("grc_report_failed", user="system", extra=f"{report_file} {e}")
            print(f"GRC integration failed: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    report_file = "compliance_exports/audit_log_2025-07-17.csv"
    grc_api_url = "https://grc.example.com/api/upload"
    send_compliance_report(report_file, grc_api_url, api_key="demo-token")