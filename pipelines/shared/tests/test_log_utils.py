"""Tests for logging utilities."""

from pipelines.shared.log_utils import get_logger


def test_get_logger_returns_logger():
    """Test that get_logger returns a logger instance."""
    logger = get_logger("test_module")
    assert logger is not None
    assert logger.name == "test_module"


def test_get_logger_different_names():
    """Test that get_logger returns different loggers for different names."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    assert logger1.name == "module1"
    assert logger2.name == "module2"
    assert logger1 != logger2
