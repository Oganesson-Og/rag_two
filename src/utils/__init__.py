"""
Utils Package
------------

Utility functions and classes for the RAG pipeline.

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Optional, Any
import logging
from datetime import datetime

# Import utilities
from .text_cleaner import TextCleaner
from .token_counter import TokenCounter
from .image_utils import ImageUtils
from .rag_tokenizer import RagTokenizer

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Type aliases
ConfigDict = Dict[str, Any]

def init_utils(config: Optional[ConfigDict] = None) -> Dict[str, Any]:
    """Initialize utility components."""
    try:
        config = config or {}
        
        return {
            'text_cleaner': TextCleaner(config.get('text_cleaner')),
            'token_counter': TokenCounter(config.get('token_counter')),
            'image_utils': ImageUtils(config.get('image_utils')),
            'rag_tokenizer': RagTokenizer(config.get('rag_tokenizer'))
        }
        
    except Exception as e:
        logger.error(f"Utils initialization error: {str(e)}")
        raise

# Export utilities
__all__ = [
    'TextCleaner',
    'TokenCounter',
    'ImageUtils',
    'RagTokenizer',
    'init_utils'
] 