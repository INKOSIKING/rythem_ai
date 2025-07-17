import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name: str, log_file: str, level: str = "INFO", max_bytes: int = 50*1024*1024, backup_count: int = 10):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger