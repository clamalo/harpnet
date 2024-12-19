"""
Setup script to ensure required directories exist.
"""

import os
from src.config import Config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def setup() -> None:
    """
    Setup base directories for processing data.
    """
    for d in [Config.PROCESSED_DIR, Config.ZIP_DIR, Config.CHECKPOINTS_DIR, Config.FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")