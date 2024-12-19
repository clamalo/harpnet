import logging
import sys

def setup_logger(name: str = "harpnet", level: int = logging.INFO) -> logging.Logger:
    """
    Setup a logger with a given name and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger