import os
import json
import datetime
from typing import Dict, Any

REGISTRY_PATH = "model_registry/"

def ensure_registry_path():
    if not os.path.exists(REGISTRY_PATH):
        os.makedirs(REGISTRY_PATH)

def register_model(model_name: str, version: str, metadata: Dict[str, Any], artifact_path: str):
    """
    Register a trained model with metadata and artifact path.
    """
    ensure_registry_path()
    model_entry = {
        "model_name": model_name,
        "version": version,
        "registered_at": datetime.datetime.utcnow().isoformat(),
        "metadata": metadata,
        "artifact_path": artifact_path
    }
    reg_file = os.path.join(REGISTRY_PATH, f"{model_name}_{version}.json")
    with open(reg_file, "w") as f:
        json.dump(model_entry, f, indent=2)
    print(f"Model {model_name}:{version} registered.")

def list_models():
    ensure_registry_path()
    return [f for f in os.listdir(REGISTRY_PATH) if f.endswith(".json")]

def get_model_info(model_name: str, version: str):
    reg_file = os.path.join(REGISTRY_PATH, f"{model_name}_{version}.json")
    if not os.path.exists(reg_file):
        raise FileNotFoundError(f"Model {model_name}:{version} not found in registry.")
    with open(reg_file) as f:
        return json.load(f)

if __name__ == "__main__":
    # Example Usage
    register_model(
        model_name="melodygen",
        version="1.0.0",
        metadata={"accuracy": 0.98, "trained_on": "2025-07-12"},
        artifact_path="models/checkpoints/melodygen_v1.pt"
    )
    print("All registered models:", list_models())
    print("Info:", get_model_info("melodygen", "1.0.0"))