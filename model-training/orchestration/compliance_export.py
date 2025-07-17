import os
import datetime
from utils.audit_logging import read_audit_log

EXPORT_DIR = "compliance_exports/"
EXPORT_FORMAT = "csv"

def export_audit_log(start_date=None, end_date=None):
    """
    Export audit log entries for compliance (regulatory, enterprise).
    Exports to CSV by default.
    """
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
    today = datetime.date.today().isoformat()
    export_file = os.path.join(EXPORT_DIR, f"audit_log_{today}.{EXPORT_FORMAT}")
    log_entries = read_audit_log(start_date, end_date)
    with open(export_file, "w") as f:
        # CSV header
        f.write("timestamp,event,user,extra\n")
        for entry in log_entries:
            line = f"{entry['timestamp']},{entry['event']},{entry['user']},{entry.get('extra','')}\n"
            f.write(line)
    print(f"Audit log exported to {export_file}")
    return export_file

if __name__ == "__main__":
    export_audit_log()