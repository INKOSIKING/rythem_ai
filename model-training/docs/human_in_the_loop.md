# Rhythm AI – Human-in-the-Loop Approval Workflow

## Why

- Enforce critical model deployments to be approved by a human reviewer.
- Add notes, comments, and status for compliance and accountability.
- All approvals/rejections are audit logged.

## How it works

1. **Submit for Approval**
   - After training, a new model is registered and submitted for approval by a reviewer (data scientist, product owner, etc).
2. **Review**
   - Reviewer inspects metrics, metadata, and attached notes.
   - Approves or rejects the model, optionally adding comments.
3. **Status Tracking**
   - Status is "pending", "approved", or "rejected".
   - Status can be queried by automation or UI.
4. **Audit Logging**
   - All actions are logged for compliance (who approved what, when, and why).
   - Approval/rejection status is included in compliance reports.

## Example

```python
from registry.human_in_the_loop_approval import submit_for_approval, approve_model, approval_status

submit_for_approval("melodygen", "1.0.1", reviewer="lead_data_scientist", notes="Ready for production.")
approve_model("melodygen", "1.0.1", reviewer="lead_data_scientist", decision="approved", comments="Meets criteria.")

print(approval_status("melodygen", "1.0.1"))  # "approved"
```

---

*Human-in-the-loop approval is a best practice for responsible, safe, and compliant enterprise AI.*