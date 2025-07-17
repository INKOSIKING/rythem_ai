"""
DVC pipeline for dataset versioning and experiment tracking.
"""

import os
import subprocess

def dvc_init():
    if not os.path.exists('.dvc'):
        subprocess.run(['dvc', 'init'], check=True)
        print("Initialized DVC repository.")

def dvc_add(path):
    subprocess.run(['dvc', 'add', path], check=True)
    print(f"Added {path} to DVC tracking.")

def dvc_push(remote="myremote"):
    subprocess.run(['dvc', 'push', '-r', remote], check=True)
    print(f"Pushed data to DVC remote: {remote}")

def dvc_pull(remote="myremote"):
    subprocess.run(['dvc', 'pull', '-r', remote], check=True)
    print(f"Pulled data from DVC remote: {remote}")

if __name__ == "__main__":
    dvc_init()
    dvc_add("datasets/processed")
    dvc_push()