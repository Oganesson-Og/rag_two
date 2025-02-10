from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional

class RAGError(Exception):
    """Base class for RAG system errors."""
    def __init__(self, message: str, severity: str = "error"):
        self.message = message
        self.severity = severity
        super().__init__(message)

class ProcessingError(RAGError):
    """Error during document processing."""
    pass

class EmbeddingError(RAGError):
    """Error during embedding generation."""
    pass

class SearchError(RAGError):
    """Error during search operations."""
    pass

class ValidationError(RAGError):
    """Error during data validation."""
    pass

@dataclass
class ErrorInfo:
    error_type: str
    message: str
    timestamp: datetime
    context: Optional[Dict[str, Any]] = None
    is_recoverable: bool = True

@dataclass
class ErrorReport:
    total_errors: int
    error_types: Dict[str, int]
    timestamp: datetime = datetime.now() 