import os
import random
import numpy as np
import torch

def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def log_env_info():
    import platform
    import sys
    import torch
    print("Python version:", sys.version)
    print("Platform:", platform.platform())
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA devices:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))