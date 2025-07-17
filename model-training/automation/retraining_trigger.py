import os
import time
from utils.audit_logging import log_event

DATA_DIR = "datasets/labeled_new/"
TRIGGER_FILE = "retrain.trigger"
TRIGGER_INTERVAL_SECONDS = 300  # 5 minutes

def check_for_new_data():
    # In production, use event-driven (S3, GCS, etc)
    return any(os.path.getmtime(os.path.join(DATA_DIR, f)) > (time.time() - TRIGGER_INTERVAL_SECONDS)
               for f in os.listdir(DATA_DIR) if not f.startswith('.'))

def trigger_retraining():
    log_event("retraining_triggered", user="system", extra="New data detected")
    print("Retraining triggered due to new labeled data.")
    # Touch a file for CI/CD or orchestrator to pick up
    with open(TRIGGER_FILE, "w") as f:
        f.write(str(time.time()))

def main():
    while True:
        if check_for_new_data():
            trigger_retraining()
        time.sleep(TRIGGER_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()