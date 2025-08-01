"""logging utility"""

import logging
import sys


def setup_logger(level=logging.INFO):
    """
    Configure logging with consistent format across all pipeline files.

    Args:
        level: Logging level (default: INFO)
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logging.info("Logging configured successfully")


def get_logger(name):
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    setup_logger()
    return logging.getLogger(name)
