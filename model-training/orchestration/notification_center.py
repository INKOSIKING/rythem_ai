import smtplib
from email.mime.text import MIMEText
import requests
from utils.audit_logging import log_event

def notify_slack(message, webhook_url):
    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=10)
        resp.raise_for_status()
        log_event("notify_slack", user="system", extra=message)
        print(f"Slack notification sent.")
        return True
    except Exception as e:
        log_event("notify_slack_failed", user="system", extra=str(e))
        print(f"Slack notification failed: {e}")
        return False

def notify_email(subject, body, to_email, smtp_server, smtp_port, smtp_user, smtp_pass):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        log_event("notify_email", user="system", extra=f"{to_email} {subject}")
        print(f"Email sent to {to_email}.")
        return True
    except Exception as e:
        log_event("notify_email_failed", user="system", extra=str(e))
        print(f"Email notification failed: {e}")
        return False

def notify_teams(message, webhook_url):
    try:
        resp = requests.post(webhook_url, json={"text": message}, timeout=10)
        resp.raise_for_status()
        log_event("notify_teams", user="system", extra=message)
        print(f"Teams notification sent.")
        return True
    except Exception as e:
        log_event("notify_teams_failed", user="system", extra=str(e))
        print(f"Teams notification failed: {e}")
        return False

def notify_sms(body, to_number, twilio_sid, twilio_token, from_number):
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
        data = {
            "From": from_number,
            "To": to_number,
            "Body": body,
        }
        resp = requests.post(url, data=data, auth=(twilio_sid, twilio_token), timeout=10)
        resp.raise_for_status()
        log_event("notify_sms", user="system", extra=f"{to_number} {body}")
        print(f"SMS sent to {to_number}.")
        return True
    except Exception as e:
        log_event("notify_sms_failed", user="system", extra=str(e))
        print(f"SMS notification failed: {e}")
        return False

if __name__ == "__main__":
    # Example: send notifications (replace with real credentials!)
    notify_slack("Test notification from Rhythm AI", "https://hooks.slack.com/services/example")
    notify_email("Rhythm AI Alert", "This is a test notification.", "user@example.com", "smtp.example.com", 465, "robot@example.com", "password")
    notify_teams("Test notification from Rhythm AI", "https://outlook.office.com/webhook/example")
    notify_sms("Test SMS from Rhythm AI", "+1234567890", "twilio_sid", "twilio_token", "+10987654321")