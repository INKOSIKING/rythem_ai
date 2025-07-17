import os
import json
from typing import List, Dict, Any, Optional

REGISTRY_PATH = "model_registry/"

def search_models(
    name_contains: Optional[str] = None,
    metric_gte: Optional[Dict[str, float]] = None,
    tag: Optional[str] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Advanced search for models in the file-based registry.
    Filters:
        - name_contains: substring in model_name
        - metric_gte: dict of metric_name: value (e.g. {"accuracy": 0.98})
        - tag: must be in metadata["tags"] (if present)
        - date_after, date_before: ISO 8601 UTC string
    Returns:
        - List of model metadata dicts
    """
    results = []
    for fname in os.listdir(REGISTRY_PATH):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(REGISTRY_PATH, fname)) as f:
            meta = json.load(f)
        if name_contains and name_contains.lower() not in meta["model_name"].lower():
            continue
        if date_after and meta["registered_at"] < date_after:
            continue
        if date_before and meta["registered_at"] > date_before:
            continue
        if metric_gte:
            match = True
            for m, v in metric_gte.items():
                if m not in meta["metadata"]:
                    match = False
                    break
                try:
                    if float(meta["metadata"][m]) < float(v):
                        match = False
                        break
                except (ValueError, TypeError):
                    match = False
                    break
            if not match:
                continue
        if tag:
            tags = meta["metadata"].get("tags", [])
            if tag not in tags:
                continue
        results.append(meta)
    return results

if __name__ == "__main__":
    # Example: search for all models with accuracy >= 0.98 after 2025-07-01
    found = search_models(metric_gte={"accuracy": 0.98}, date_after="2025-07-01T00:00:00")
    print("Found models:", found)