import os
import datetime

def export_audit_log_to_csv(audit_log="audit.log", out_csv="audit_report.csv"):
    import csv
    if not os.path.exists(audit_log):
        raise FileNotFoundError(f"{audit_log} not found")
    with open(audit_log, "r") as f, open(out_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "user", "event", "extra"])
        for line in f:
            if "|" not in line:
                continue
            parts = [x.strip() for x in line.split("|")]
            if len(parts) < 4:
                continue
            writer.writerow([
                parts[0],
                parts[1].replace("USER: ", ""),
                parts[2].replace("EVENT: ", ""),
                parts[3].replace("EXTRA: ", ""),
            ])
    print(f"Audit log exported to {out_csv}")

def compliance_report_summary(audit_csv="audit_report.csv"):
    import pandas as pd
    df = pd.read_csv(audit_csv)
    print("=== Compliance Report Summary ===")
    print("Total events:", len(df))
    print("Events by user:")
    print(df["user"].value_counts())
    print("Events by type:")
    print(df["event"].value_counts())
    # Add more compliance checks as needed

if __name__ == "__main__":
    export_audit_log_to_csv()
    compliance_report_summary()