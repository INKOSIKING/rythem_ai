import os
import datetime

AUDIT_LOG_PATH = "audit.log"

def log_event(event, user="system", extra=""):
    ts = datetime.datetime.utcnow().isoformat()
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(f"{ts},{event},{user},{extra}\n")

def read_audit_log(start_date=None, end_date=None):
    """
    Reads the audit log and returns entries as list of dicts.
    Optionally filter by date (YYYY-MM-DD).
    """
    entries = []
    if not os.path.exists(AUDIT_LOG_PATH):
        return entries
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            ts, event, user, *extra = line.strip().split(",", 3)
            date_str = ts.split("T")[0]
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            entries.append({
                "timestamp": ts,
                "event": event,
                "user": user,
                "extra": extra[0] if extra else ""
            })
    return entries

if __name__ == "__main__":
    log_event("test_event", user="system", extra="testing log")
    print(read_audit_log())