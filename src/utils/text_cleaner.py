"""
Text Cleaning Module
------------------

Text cleaning and preprocessing utilities.

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import re
import unicodedata
from typing import List, Optional

class TextCleaner:
    """Text cleaning and normalization utilities."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        return text.strip()
    
    @staticmethod
    def remove_special_characters(text: str, keep_chars: Optional[str] = None) -> str:
        """Remove special characters except those specified."""
        if keep_chars:
            pattern = f'[^a-zA-Z0-9\s{re.escape(keep_chars)}]'
        else:
            pattern = '[^a-zA-Z0-9\s]'
        return re.sub(pattern, '', text)
    
    @staticmethod
    def clean_latex(text: str) -> str:
        """Clean LaTeX commands and environments."""
        # Remove inline math
        text = re.sub(r'\$[^$]+\$', '', text)
        # Remove display math
        text = re.sub(r'\\\[[^\]]+\\\]', '', text)
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        return text
    
    @staticmethod
    def clean_html(text: str) -> str:
        """Remove HTML tags."""
        return re.sub(r'<[^>]+>', '', text)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces at start/end of lines
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def fix_line_breaks(text):
        """Fix incorrect line breaks"""
        # Remove breaks in middle of sentences
        text = re.sub(r'(?<=[a-z])\n(?=[a-z])', ' ', text)
        
        # Keep paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text