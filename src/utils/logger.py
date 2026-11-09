"""
Logging Utility for the Healthcare AI Assistant.
Provides standardized logging to console and persistent log files.
"""

import logging
import sys
from pathlib import Path
from config.settings import LOGS_DIR, APP_LOG_PATH

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str = "healthcare_ai", log_file: Path = APP_LOG_PATH, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance with console and file handlers.
    
    Args:
        name: Name of the logger.
        log_file: Path to destination log file.
        level: Logging verbosity level.
        
    Returns:
        logging.Logger configured instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Global default logger
logger = setup_logger()

