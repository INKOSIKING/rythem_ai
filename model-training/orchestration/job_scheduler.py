import os
import time
import subprocess
from utils.audit_logging import log_event

JOBS_DIR = "jobs/"
CHECK_INTERVAL = 60  # seconds

def discover_jobs():
    """
    Discover pending job files in JOBS_DIR.
    Each job is a shell script or command file.
    """
    if not os.path.exists(JOBS_DIR):
        os.makedirs(JOBS_DIR)
    return [os.path.join(JOBS_DIR, f) for f in os.listdir(JOBS_DIR) if f.endswith(".job") and not f.endswith(".done")]

def run_job(job_file):
    """
    Run a job file (shell script or command), mark as done, and log.
    """
    try:
        log_event("job_started", user="system", extra=job_file)
        print(f"Running job: {job_file}")
        # For security: Only allow trusted scripts in production.
        result = subprocess.run(["bash", job_file], capture_output=True, text=True, timeout=1800)
        print(f"Job output:\n{result.stdout}")
        if result.returncode == 0:
            log_event("job_success", user="system", extra=job_file)
        else:
            log_event("job_failed", user="system", extra=f"{job_file}: rc={result.returncode}")
        # Mark as done
        os.rename(job_file, job_file + ".done")
    except Exception as e:
        log_event("job_exception", user="system", extra=f"{job_file}: {e}")

def main():
    while True:
        jobs = discover_jobs()
        if jobs:
            for job in jobs:
                run_job(job)
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()