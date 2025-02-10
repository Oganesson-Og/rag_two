import logging
import json
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import time
from typing import List, Dict, Any, Optional
from .models import LogConfig, LogEvent, LogLevel

class RagLogger:
    def __init__(self, config_manager, log_config: LogConfig):
        self.config_manager = config_manager
        self.config = log_config
        self._setup_logger()
        
    def _setup_logger(self):
        self.logger = logging.getLogger('rag_logger')
        self.logger.setLevel(self.config.level.value)
        
        handler = logging.FileHandler(Path(self.config.log_dir) / self.config.file_name)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def log(self, level: LogLevel, message: str, **kwargs):
        self.logger.log(level.value, message, extra=kwargs)
        
    def info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)
        
    def error(self, message: str, exc_info=None, **kwargs):
        if exc_info:
            kwargs['exc_info'] = str(exc_info)
        self.log(LogLevel.ERROR, message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)
        
    @contextmanager
    def track_duration(self):
        start_time = time.time()
        try:
            yield lambda: time.time() - start_time
        finally:
            duration = time.time() - start_time
            self.info("Operation completed", duration_ms=duration * 1000) 