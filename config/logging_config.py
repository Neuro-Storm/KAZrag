"""Module for centralized logging configuration."""

import logging
from pathlib import Path
from typing import Optional

# Import JSON formatter
from pythonjsonlogger import jsonlogger

# Create logs directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(
    level: int = logging.INFO,
    use_json_format: bool = False,
    log_file: Optional[str] = None
) -> None:
    """Set up centralized logging configuration.
    
    Args:
        level: Logging level (default: logging.INFO)
        use_json_format: Whether to use JSON format for logs (default: False)
        log_file: Path to log file (default: logs/app.log)
    """
    # Set up formatters
    if use_json_format:
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Set up handlers
    handlers = []
    
    # File handler
    if log_file is None:
        log_file = str(LOG_DIR / 'app.log')
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
    # Reduce verbosity of the file watcher to avoid log-change loops
    logging.getLogger('watchfiles').setLevel(logging.WARNING)