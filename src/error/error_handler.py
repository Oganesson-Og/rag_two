from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from .models import RAGError, ErrorInfo, ErrorReport

class ErrorHandler:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.error_logs = []
        
    def handle_error(self, error: Exception) -> ErrorInfo:
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            message=str(error),
            timestamp=datetime.now()
        )
        self.error_logs.append(error_info)
        return error_info
        
    def handle_rag_error(self, error: RAGError) -> ErrorInfo:
        return self.handle_error(error)
        
    def execute_with_fallback(self, main_operation: Callable, fallback_operation: Callable):
        try:
            return main_operation()
        except Exception as e:
            self.handle_error(e)
            return fallback_operation()
            
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        error_info = self.handle_error(error)
        error_info.context = context
        
    def get_error_logs(self) -> List[ErrorInfo]:
        return self.error_logs
        
    def categorize_errors(self, errors: List[Exception]) -> Dict[str, List[Exception]]:
        categorized = {}
        for error in errors:
            category = error.__class__.__name__.lower().replace('error', '')
            categorized.setdefault(category, []).append(error)
        return categorized 