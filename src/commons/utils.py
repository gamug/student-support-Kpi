import os

from src.config import paths

def check_paths():
    for path in paths.values():
        os.makedirs(path, exist_ok=True)