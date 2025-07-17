# Rhythm AI – Model Registry UI & Self-Service Promotion

## Goals

- Provide a web UI for searching, reviewing, and promoting models to production.
- Display metrics, artifacts, approval status, audit history.
- Allow reviewers to approve/reject models and add comments.

## Open Source Tools

- [MLflow Model Registry UI](https://mlflow.org/docs/latest/model-registry.html)
- [DVC Studio](https://studio.iterative.ai/)
- [Streamlit](https://streamlit.io/) or [Gradio](https://gradio.app/) for custom internal dashboards.

## Example: Minimal Streamlit UI

```python
import streamlit as st
from registry.model_registry import list_models, get_model_info
from registry.human_in_the_loop_approval import approval_status

st.title("Rhythm AI Model Registry")

for model_file in list_models():
    model_info = get_model_info(model_file.split("_")[0], model_file.split("_")[1].replace(".json", ""))
    st.header(f"{model_info['model_name']}:{model_info['version']}")
    st.write(model_info["metadata"])
    st.write("Approval Status:", approval_status(model_info['model_name'], model_info['version']))
```

---

*For advanced features, integrate with MLflow or DVC Studio, or build out the above Streamlit with approval buttons and audit trails!*