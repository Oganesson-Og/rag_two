"""
Validation Utilities
-----------------
"""

import re
from typing import Any, Dict, Union, Optional, List
from pathlib import Path

class ValidationUtils:
    def is_valid_email(self, email: str) -> bool:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
        
    def is_valid_url(self, url: str) -> bool:
        pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        return bool(re.match(pattern, url))
        
    def is_in_range(self, value: float, min_val: float, max_val: float) -> bool:
        return min_val <= value <= max_val
        
    def is_valid_json(self, text: str) -> bool:
        try:
            import json
            json.loads(text)
            return True
        except ValueError:
            return False
        
    def validate_file_type(self, file_path: Union[str, Path], 
                          allowed_types: Optional[List[str]] = None) -> bool:
        """Validate file type against allowed types."""
        if allowed_types is None:
            allowed_types = ['.txt', '.pdf', '.doc', '.docx']
            
        file_path = Path(file_path)
        return file_path.suffix.lower() in allowed_types
        
    def validate_text_length(self, text: str, 
                           min_length: int = 1, 
                           max_length: Optional[int] = None) -> bool:
        """Validate text length."""
        if len(text) < min_length:
            return False
        if max_length and len(text) > max_length:
            return False
        return True
        
    def validate_metadata(self, metadata: Dict[str, Any], 
                         required_fields: Optional[List[str]] = None) -> bool:
        """Validate metadata contains required fields."""
        if required_fields is None:
            return True
        return all(field in metadata for field in required_fields) 