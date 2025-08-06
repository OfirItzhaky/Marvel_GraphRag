import os
import shutil

CACHE_DIR = "cache"

def cache_exists(filename="filtered_docs.json"):
    """
    Check if a specific cache file exists in the cache directory.

    @param filename: The name of the file to check for (default is "filtered_docs.json").
    @return: True if the file exists, False otherwise.
    """
    return os.path.exists(os.path.join(CACHE_DIR, filename))

def clear_cache():
    """
    Delete the entire cache directory and recreate it as empty.

    @return: None
    """
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print("🧹 Cache cleared.")
    os.makedirs(CACHE_DIR, exist_ok=True)

def ensure_cache_dir():
    """
    Ensure that the cache directory exists. If not, create it.

    @return: None
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
