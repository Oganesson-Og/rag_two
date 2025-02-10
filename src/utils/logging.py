import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

@dataclass
class LogConfig:
    log_file: Path
    level: int = logging.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_bytes: int = 1024 * 1024  # 1MB
    backup_count: int = 3

def setup_logging(config: LogConfig) -> logging.Logger:
    """Setup logging with the given configuration."""
    logger = logging.getLogger('rag')
    logger.setLevel(config.level)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count
    )
    
    # Create formatters
    formatter = logging.Formatter(config.format)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    
    return logger 