import yaml
from registry.model_registry import register_model

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    # Example: fetch metrics from evaluation output
    with open("eval_results.json") as f:
        eval_results = f.read()
    # Register latest model
    register_model(
        model_name=cfg["model"]["type"],
        version="auto-" + cfg["training"]["checkpoint_path"].split("/")[-1],
        metadata={"eval": eval_results, "config": cfg},
        artifact_path=cfg["training"]["checkpoint_path"] + "best_model.pt"
    )