import os
import json
from typing import Dict, Any, Optional

APPROVAL_PATH = "model_registry/approvals/"

def ensure_approval_path():
    if not os.path.exists(APPROVAL_PATH):
        os.makedirs(APPROVAL_PATH)

def submit_for_approval(model_name: str, version: str, reviewer: str, notes: Optional[str] = None):
    """
    Mark a model as pending human approval.
    """
    ensure_approval_path()
    approval_entry = {
        "model_name": model_name,
        "version": version,
        "status": "pending",
        "submitted_at": __import__("datetime").datetime.utcnow().isoformat(),
        "reviewer": reviewer,
        "notes": notes
    }
    fpath = os.path.join(APPROVAL_PATH, f"{model_name}_{version}_approval.json")
    with open(fpath, "w") as f:
        json.dump(approval_entry, f, indent=2)
    print(f"Approval requested for {model_name}:{version} to reviewer {reviewer}.")

def approve_model(model_name: str, version: str, reviewer: str, decision: str, comments: Optional[str] = None):
    """
    Approve or reject a model for production.
    """
    fpath = os.path.join(APPROVAL_PATH, f"{model_name}_{version}_approval.json")
    if not os.path.exists(fpath):
        raise FileNotFoundError("Approval request not found.")
    with open(fpath) as f:
        approval_entry = json.load(f)
    approval_entry["reviewed_at"] = __import__("datetime").datetime.utcnow().isoformat()
    approval_entry["status"] = decision
    approval_entry["reviewer"] = reviewer
    approval_entry["comments"] = comments
    with open(fpath, "w") as f:
        json.dump(approval_entry, f, indent=2)
    print(f"Model {model_name}:{version} {decision} by {reviewer}.")

def approval_status(model_name: str, version: str):
    fpath = os.path.join(APPROVAL_PATH, f"{model_name}_{version}_approval.json")
    if not os.path.exists(fpath):
        return "not requested"
    with open(fpath) as f:
        entry = json.load(f)
    return entry["status"]

if __name__ == "__main__":
    # Example usage
    submit_for_approval("melodygen", "1.0.1", reviewer="lead_data_scientist", notes="Review for production.")
    approve_model("melodygen", "1.0.1", reviewer="lead_data_scientist", decision="approved", comments="Meets all criteria.")
    print("Status:", approval_status("melodygen", "1.0.1"))