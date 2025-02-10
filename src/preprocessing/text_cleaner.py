"""
Text Cleaner Module
---------------
"""

from typing import Dict, List, Union, Optional
import re
from dataclasses import dataclass

@dataclass
class CleaningConfig:
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = True

class TextCleaner:
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        
    def clean_text(self, text: str) -> str:
        """Clean text according to configuration."""
        if not text:
            return ""
            
        if self.config.lowercase:
            text = text.lower()
            
        if self.config.remove_urls:
            text = self._remove_urls(text)
            
        if self.config.remove_emails:
            text = self._remove_emails(text)
            
        if self.config.remove_phone_numbers:
            text = self._remove_phone_numbers(text)
            
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)
            
        return text.strip()
        
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        
    def _remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        return re.sub(r'\+?[\d\-\(\)\s]{10,}', '', text)
        
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return re.sub(r'\s+', ' ', text) 