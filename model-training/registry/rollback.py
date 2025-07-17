import os
import shutil
from typing import Optional

REGISTRY_PATH = "model_registry/"
CHECKPOINTS_PATH = "models/checkpoints/"
BACKUP_PATH = "model_registry/rollback_backups/"

def ensure_backup_path():
    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH)

def backup_model_artifact(artifact_path: str):
    ensure_backup_path()
    base_name = os.path.basename(artifact_path)
    backup_file = os.path.join(BACKUP_PATH, base_name)
    shutil.copy2(artifact_path, backup_file)
    print(f"Backup of {artifact_path} created at {backup_file}")
    return backup_file

def rollback_model(model_name: str, version: str, target_version: Optional[str] = None):
    """
    Rolls back a model to a specified previous version.
    - If target_version is None, will rollback to the previous version by timestamp order.
    - Backs up current artifact before replacing.
    - Updates registry and optionally approval status.
    """
    import json
    registry_files = sorted([f for f in os.listdir(REGISTRY_PATH) if f.startswith(model_name + "_") and f.endswith(".json")])
    if target_version is None:
        # Find the previous version
        idx = [f"{model_name}_{version}.json" == f for f in registry_files]
        if not any(idx):
            raise FileNotFoundError("Current version not found in registry.")
        cur_idx = idx.index(True)
        if cur_idx == 0:
            raise Exception("No previous version to rollback to.")
        prev_file = registry_files[cur_idx - 1]
    else:
        prev_file = f"{model_name}_{target_version}.json"
        if prev_file not in registry_files:
            raise FileNotFoundError("Target rollback version not found in registry.")
    # Load previous version
    with open(os.path.join(REGISTRY_PATH, prev_file)) as f:
        prev_meta = json.load(f)
    prev_artifact = prev_meta["artifact_path"]
    cur_file = f"{model_name}_{version}.json"
    with open(os.path.join(REGISTRY_PATH, cur_file)) as f:
        cur_meta = json.load(f)
    cur_artifact = cur_meta["artifact_path"]
    backup_model_artifact(cur_artifact)
    # Rollback artifact
    shutil.copy2(prev_artifact, cur_artifact)
    print(f"Rolled back {model_name} from {version} to {prev_meta['version']} (artifact replaced).")
    # Optionally update registry with rollback note
    cur_meta["rollback_to"] = prev_meta["version"]
    with open(os.path.join(REGISTRY_PATH, cur_file), "w") as f:
        json.dump(cur_meta, f, indent=2)
    # Optionally update approval status
    try:
        from registry.human_in_the_loop_approval import submit_for_approval
        submit_for_approval(model_name, prev_meta["version"], reviewer="auto", notes="Rolled-back version, requires re-approval.")
    except Exception as e:
        print("Approval status not updated:", e)

if __name__ == "__main__":
    # Example: rollback melodygen 1.0.2 to previous version
    rollback_model("melodygen", "1.0.2")