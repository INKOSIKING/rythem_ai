from utils.audit_logging import log_event

def audited_submit_for_approval(*args, **kwargs):
    user = kwargs.pop("user", None) or "system"
    from registry.human_in_the_loop_approval import submit_for_approval
    submit_for_approval(*args, **kwargs)
    log_event("model_approval_requested", user=user, extra=f"{args[0]}:{args[1]} reviewer={args[2]}")

def audited_approve_model(*args, **kwargs):
    user = kwargs.pop("user", None) or "system"
    from registry.human_in_the_loop_approval import approve_model
    approve_model(*args, **kwargs)
    log_event("model_approval_decision", user=user, extra=f"{args[0]}:{args[1]} decision={args[3]} reviewer={args[2]}")

if __name__ == "__main__":
    # Example usage with audit
    audited_submit_for_approval("melodygen", "1.0.1", reviewer="lead_data_scientist", notes="Ready", user="RythemAI")
    audited_approve_model("melodygen", "1.0.1", reviewer="lead_data_scientist", decision="approved", comments="OK", user="lead_data_scientist")