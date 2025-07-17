import time
import psutil
import logging

def log_resource_usage(interval=60):
    while True:
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        gpu = "N/A"
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu = gpus[0].load * 100 if gpus else "N/A"
        except ImportError:
            pass
        logging.info(f"CPU: {cpu}% | RAM: {mem}% | GPU: {gpu}%")
        time.sleep(interval)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_resource_usage()