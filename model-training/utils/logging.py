import logging
import wandb

def setup_logging(cfg):
    logging.basicConfig(level=logging.INFO)
    wandb.init(project=cfg.get("wandb_project", "rhythm-ai"), config=cfg)

def log_metrics(metrics: dict):
    logging.info(metrics)
    wandb.log(metrics)