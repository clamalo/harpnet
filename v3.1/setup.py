from config import RAW_DIR, PROCESSED_DIR

def setup():
    """
    Create all the necessary directories in config.py if they don't already exist.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directories ensured:\n  RAW_DIR: {RAW_DIR}\n  PROCESSED_DIR: {PROCESSED_DIR}")