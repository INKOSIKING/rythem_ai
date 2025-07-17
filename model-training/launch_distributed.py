import torch
import torch.multiprocessing as mp
from utils.distributed import setup_ddp, cleanup_ddp
from train import main as train_main
import yaml
import os

def run(rank, world_size, cfg):
    setup_ddp(rank, world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    train_main()  # Should read LOCAL_RANK for DDP setup
    cleanup_ddp()

if __name__ == "__main__":
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, cfg), nprocs=world_size, join=True)