import os
import shutil

CACHE_DIR = "cache"

def cache_exists(filename="filtered_docs.json"):
    return os.path.exists(os.path.join(CACHE_DIR, filename))

def clear_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("🧹 Cache cleared.")
    os.makedirs(CACHE_DIR, exist_ok=True)

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)
