import mlflow

def setup_mlflow(cfg):
    mlflow.set_experiment(cfg.get("mlflow_experiment", "rhythm-ai"))
    mlflow.start_run(run_name=cfg.get("run_name", "experiment"))

def log_metrics_mlflow(metrics: dict, step=None):
    mlflow.log_metrics(metrics, step=step)

def log_artifact_mlflow(path):
    mlflow.log_artifact(path)