"""
Base Document Extractor Module
---------------------------

Base classes and utilities for document content extraction.

Key Features:
- Common extraction interfaces
- Shared utilities
- Error handling
- Type definitions
- Metadata management
- Validation tools

Technical Details:
- Abstract base classes
- Type annotations
- Error management
- Metadata handling
- Content validation
- Logging integration

Dependencies:
- abc (standard library)
- typing (standard library)
- logging (standard library)
- datetime (standard library)
- numpy>=1.24.0

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import numpy as np
from numpy.typing import NDArray
import logging
from datetime import datetime

# Type aliases
ImageArray = NDArray[np.uint8]
ExtractorResult = Dict[str, Any]
DocumentContent = Union[str, bytes, NDArray[np.uint8]]

class BaseExtractor(ABC):
    """Base class for all document extractors."""
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def extract(
        self,
        content: DocumentContent,
        options: Optional[Dict[str, bool]] = None
    ) -> ExtractorResult:
        """Extract content from document."""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning shared by all extractors."""
        if not text:
            return ""
        return " ".join(text.split())

    def get_metadata(self) -> Dict[str, Any]:
        """Get extractor metadata."""
        return {
            'extractor': self.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }

    def validate_content(self, content: DocumentContent) -> bool:
        """Validate input content."""
        if content is None:
            return False
        if isinstance(content, (str, bytes)):
            return len(content) > 0
        if isinstance(content, np.ndarray):
            return content.size > 0
        return False 