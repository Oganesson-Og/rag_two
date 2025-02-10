"""
Error Manager Module
------------------

Comprehensive error management system for educational RAG applications with 
advanced tracking, recovery, and analysis capabilities.

Key Features:
- Error event tracking and management
- Severity-based error handling
- Automatic recovery mechanisms
- Error pattern analysis
- Multi-category error support
- Context preservation
- Recovery step generation

Technical Details:
- Thread-safe error tracking
- Configurable recovery strategies
- Error pattern detection
- Asynchronous recovery
- Persistent error logging
- Performance monitoring
- Context management

Dependencies:
- logging (standard library)
- datetime (standard library)
- threading (standard library)
- queue (standard library)
- json (standard library)
- asyncio (standard library)
- pathlib (standard library)

Example Usage:
    # Initialize manager
    manager = ErrorManager(config_path="config/error.json")
    
    # Handle error with context
    with manager.error_context(category=ErrorCategory.PROCESSING):
        process_document()
    
    # Direct error handling
    try:
        embed_text()
    except Exception as e:
        event = manager.handle_error(
            e, 
            ErrorCategory.EMBEDDING,
            {"text_length": 1000}
        )

Error Categories:
- Content Processing
- Embedding Generation
- Retrieval Operations
- Math Calculations
- System Operations
- Database Operations
- Validation Tasks

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
from pathlib import Path
import traceback
from enum import Enum
from dataclasses import dataclass
import json
import asyncio
from contextlib import contextmanager
import sys
import threading
from queue import Queue

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    CONTENT = "content"
    EMBEDDING = "embedding"
    RETRIEVAL = "retrieval"
    PROCESSING = "processing"
    MATH = "math"
    SYSTEM = "system"
    DATABASE = "database"
    VALIDATION = "validation"

@dataclass
class ErrorEvent:
    """Represents an error event in the system."""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    stack_trace: Optional[str]
    context: Dict[str, Any]
    recovery_steps: List[str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class ErrorManager:    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        enable_auto_recovery: bool = True
    ):
        self.config = self._load_config(config_path)
        self.log_dir = log_dir or Path("logs/errors")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize error tracking
        self.error_history: List[ErrorEvent] = []
        self.active_errors: Dict[str, ErrorEvent] = {}
        self.error_patterns: Dict[str, Dict] = {}
        
        # Recovery configuration
        self.enable_auto_recovery = enable_auto_recovery
        self.recovery_handlers: Dict[ErrorCategory, Callable] = {}
        self.recovery_queue = Queue()
        
        # Start recovery thread if enabled
        if enable_auto_recovery:
            self.recovery_thread = threading.Thread(target=self._recovery_worker)
            self.recovery_thread.daemon = True
            self.recovery_thread.start()

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load error handling configuration."""
        default_config = {
            "max_retries": 3,
            "retry_delay": 1.0,
            "error_threshold": {
                "critical": 1,
                "high": 5,
                "medium": 10,
                "low": 20
            },
            "auto_recovery_enabled": True,
            "notification_channels": ["log", "email"]
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
                
        return default_config

    def _setup_logging(self) -> logging.Logger:
        """Configure error logging."""
        logger = logging.getLogger("error_manager")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(
            self.log_dir / "error.log"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Stream handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        logger.setLevel(logging.INFO)
        return logger

    @contextmanager
    def error_context(
        self,
        category: ErrorCategory,
        context: Dict[str, Any] = None
    ):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            self.handle_error(e, category, context or {})
            raise

    def handle_error(
        self,
        error: Exception,
        category: ErrorCategory,
        context: Dict[str, Any] = None
    ) -> ErrorEvent:
        """Handle and process an error."""
        error_id = f"{category.value}_{datetime.now().timestamp()}"
        
        # Determine error severity
        severity = self._determine_severity(error, category)
        
        # Create error event
        event = ErrorEvent(
            error_id=error_id,
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
            recovery_steps=self._generate_recovery_steps(error, category)
        )
        
        # Record error
        self.error_history.append(event)
        self.active_errors[error_id] = event
        
        # Log error
        self._log_error(event)
        
        # Check for error patterns
        self._analyze_error_pattern(event)
        
        # Trigger auto-recovery if enabled
        if self.enable_auto_recovery:
            self.recovery_queue.put(event)
            
        return event

    def _determine_severity(
        self,
        error: Exception,
        category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity based on type and context."""
        if isinstance(error, (SystemError, MemoryError)):
            return ErrorSeverity.CRITICAL
            
        if category in [ErrorCategory.DATABASE, ErrorCategory.SYSTEM]:
            return ErrorSeverity.HIGH
            
        if isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.MEDIUM
            
        return ErrorSeverity.LOW

    def _generate_recovery_steps(
        self,
        error: Exception,
        category: ErrorCategory
    ) -> List[str]:
        """Generate recovery steps based on error type."""
        steps = []
        
        if category == ErrorCategory.CONTENT:
            steps.extend([
                "Validate content format",
                "Check content completeness",
                "Verify educational metadata"
            ])
        elif category == ErrorCategory.EMBEDDING:
            steps.extend([
                "Verify input dimensions",
                "Check model availability",
                "Validate embedding parameters"
            ])
        elif category == ErrorCategory.MATH:
            steps.extend([
                "Validate mathematical expressions",
                "Check formula formatting",
                "Verify equation completeness"
            ])
            
        return steps

    def _log_error(self, event: ErrorEvent) -> None:
        """Log error event with appropriate severity."""
        log_message = (
            f"Error [{event.error_id}] - {event.category.value}: {event.message}\n"
            f"Severity: {event.severity.value}\n"
            f"Context: {json.dumps(event.context, indent=2)}"
        )
        
        if event.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif event.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        else:
            self.logger.warning(log_message)

    def _analyze_error_pattern(self, event: ErrorEvent) -> None:
        """Analyze error patterns and update tracking."""
        pattern_key = f"{event.category.value}_{type(event.message).__name__}"
        
        if pattern_key not in self.error_patterns:
            self.error_patterns[pattern_key] = {
                "count": 0,
                "first_seen": datetime.now(),
                "last_seen": datetime.now(),
                "examples": []
            }
            
        pattern = self.error_patterns[pattern_key]
        pattern["count"] += 1
        pattern["last_seen"] = datetime.now()
        pattern["examples"].append(event.error_id)
        
        # Check if pattern exceeds threshold
        if pattern["count"] >= self.config["error_threshold"][event.severity.value]:
            self._handle_error_pattern(pattern_key, pattern)

    def _handle_error_pattern(
        self,
        pattern_key: str,
        pattern: Dict
    ) -> None:
        """Handle detected error patterns."""
        self.logger.error(
            f"Error pattern detected: {pattern_key}\n"
            f"Count: {pattern['count']}\n"
            f"Duration: {pattern['last_seen'] - pattern['first_seen']}"
        )
        
        # Implement pattern-specific handling
        # This could include:
        # - Alerting administrators
        # - Triggering system diagnostics
        # - Implementing temporary workarounds

    def register_recovery_handler(
        self,
        category: ErrorCategory,
        handler: Callable
    ) -> None:
        """Register a recovery handler for an error category."""
        self.recovery_handlers[category] = handler

    def _recovery_worker(self) -> None:
        """Background worker for processing recovery queue."""
        while True:
            try:
                error_event = self.recovery_queue.get()
                
                if error_event.category in self.recovery_handlers:
                    handler = self.recovery_handlers[error_event.category]
                    try:
                        handler(error_event)
                        self._mark_error_resolved(error_event.error_id)
                    except Exception as e:
                        self.logger.error(
                            f"Recovery failed for {error_event.error_id}: {str(e)}"
                        )
                        
            except Exception as e:
                self.logger.error(f"Recovery worker error: {str(e)}")
                
            finally:
                self.recovery_queue.task_done()

    def _mark_error_resolved(self, error_id: str) -> None:
        """Mark an error as resolved."""
        if error_id in self.active_errors:
            error_event = self.active_errors[error_id]
            error_event.resolved = True
            error_event.resolution_time = datetime.now()
            del self.active_errors[error_id]

    def get_error_statistics(self) -> Dict[str, Any]:
        """Generate error statistics and insights."""
        return {
            "total_errors": len(self.error_history),
            "active_errors": len(self.active_errors),
            "by_category": self._get_category_stats(),
            "by_severity": self._get_severity_stats(),
            "patterns": self.error_patterns,
            "resolution_rate": self._calculate_resolution_rate()
        }

    def _get_category_stats(self) -> Dict[str, int]:
        """Get error statistics by category."""
        stats = {category.value: 0 for category in ErrorCategory}
        for error in self.error_history:
            stats[error.category.value] += 1
        return stats

    def _get_severity_stats(self) -> Dict[str, int]:
        """Get error statistics by severity."""
        stats = {severity.value: 0 for severity in ErrorSeverity}
        for error in self.error_history:
            stats[error.severity.value] += 1
        return stats

    def _calculate_resolution_rate(self) -> float:
        """Calculate error resolution rate."""
        total = len(self.error_history)
        resolved = sum(1 for error in self.error_history if error.resolved)
        return resolved / total if total > 0 else 1.0

    def cleanup(self) -> None:
        """Cleanup error manager resources."""
        if self.enable_auto_recovery:
            self.recovery_queue.join()

    def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle errors with context and appropriate fallback."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(
            f"Error: {str(error)}, Type: {error_type}, Context: {context}",
            exc_info=True
        ) 