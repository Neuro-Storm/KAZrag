"""Centralized logging configuration using loguru."""

import logging
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    rotation: str = "100 MB",
    retention: str = "30 days",
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    serialize: bool = False
) -> None:
    """Set up centralized logging configuration with loguru.
    
    Args:
        level: Logging level (default: "INFO")
        log_file: Path to log file (default: logs/app.log)
        rotation: Log file rotation policy (default: "100 MB")
        retention: Log file retention policy (default: "30 days")
        format: Log message format
        serialize: Whether to serialize logs as JSON (default: False)
    """
    # Remove default logger to avoid duplicate logs
    logger.remove()
    
    # Create logs directory
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir / "app.log")
    
    # Add file sink with rotation and retention
    logger.add(
        log_file,
        level=level,
        format=format,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        serialize=serialize
    )
    
    # Add console sink
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        format=format,
        serialize=serialize
    )
    
    # Reduce verbosity of the file watcher to avoid log-change loops
    try:
        logger.level("watchfiles", no=30)  # Set to WARNING level
    except ValueError:
        # Level already exists, that's fine
        pass


# Convenience function for getting logger instance
def get_logger(name: str = None):
    """Get a logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        loguru logger instance
    """
    return logger


# Intercept standard logging messages and redirect to loguru
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# Setup intercept handler for standard logging
def setup_intercept_handler():
    """Setup intercept handler to redirect standard logging to loguru."""
    logging.basicConfig(handlers=[InterceptHandler()], level=0)