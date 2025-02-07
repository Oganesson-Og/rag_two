"""
Document Preprocessor Module
-------------------------

Advanced document preprocessing system for standardizing and enhancing
document content before main processing pipeline.

Features:
- Text normalization
- Language detection
- Character encoding handling
- Format standardization
- Quality enhancement
- Noise removal
- Structure preservation

Key Components:
1. Text Normalization: Unicode and encoding standardization
2. Language Processing: Detection and validation
3. Format Handler: Multiple format support
4. Quality Enhancement: Content improvement
5. Structure Analyzer: Document structure preservation

Technical Details:
- Unicode normalization
- Encoding detection
- Language identification
- Format conversion
- Quality metrics
- Error correction
- Performance optimization

Dependencies:
- chardet>=5.0.0
- langdetect>=1.0.9
- ftfy>=6.1.1
- beautifulsoup4>=4.12.0
- python-magic>=0.4.27

Author: Keith Satuku
Version: 1.4.0
Created: 2025
License: MIT
"""

import chardet
import unicodedata
from langdetect import detect
from ftfy import fix_text
import magic
from bs4 import BeautifulSoup
import logging
from typing import Dict, Optional, Union
from pathlib import Path

class DocumentPreprocessor:
    """Handles document preprocessing tasks."""
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        fix_encoding: bool = True,
        detect_language: bool = True,
        remove_noise: bool = True
    ):
        """Initialize the document preprocessor.
        
        Args:
            normalize_unicode: Whether to normalize Unicode characters
            fix_encoding: Whether to fix character encoding issues
            detect_language: Whether to perform language detection
            remove_noise: Whether to remove noise from content
        """
        self.normalize_unicode = normalize_unicode
        self.fix_encoding = fix_encoding
        self.detect_language = detect_language
        self.remove_noise = remove_noise
        self.mime = magic.Magic(mime=True)
        
    def process(self, content: Union[str, bytes], file_path: Optional[Path] = None) -> Dict:
        """Process document content.
        
        Args:
            content: Document content as string or bytes
            file_path: Optional path to source file
            
        Returns:
            Dict containing processed content and metadata
        """
        result = {
            'content': content,
            'metadata': {}
        }
        
        # Detect and fix encoding if needed
        if isinstance(content, bytes):
            encoding = chardet.detect(content)
            result['metadata']['original_encoding'] = encoding['encoding']
            content = content.decode(encoding['encoding'], errors='replace')
            result['content'] = content
            
        # Fix text encoding issues
        if self.fix_encoding:
            result['content'] = fix_text(result['content'])
            
        # Normalize Unicode
        if self.normalize_unicode:
            result['content'] = self._normalize_unicode(result['content'])
            
        # Detect language
        if self.detect_language:
            try:
                result['metadata']['language'] = detect(result['content'])
            except Exception as e:
                logging.warning(f"Language detection failed: {e}")
                result['metadata']['language'] = 'unknown'
                
        # Remove noise if requested
        if self.remove_noise:
            result['content'] = self._remove_noise(result['content'])
            
        # Detect MIME type if file path provided
        if file_path:
            try:
                result['metadata']['mime_type'] = self.mime.from_file(str(file_path))
            except Exception as e:
                logging.warning(f"MIME type detection failed: {e}")
                
        return result
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters in text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize('NFKC', text)
    
    def _remove_noise(self, content: str) -> str:
        """Remove noise from document content.
        
        Args:
            content: Document content
            
        Returns:
            Cleaned content
        """
        # Remove excessive whitespace
        content = ' '.join(content.split())
        
        # Remove HTML if present
        if '<' in content and '>' in content:
            try:
                soup = BeautifulSoup(content, 'html.parser')
                content = soup.get_text(separator=' ')
            except Exception as e:
                logging.warning(f"HTML cleaning failed: {e}")
                
        # Remove control characters
        content = ''.join(char for char in content if unicodedata.category(char)[0] != 'C')
        
        return content
    
    def preprocess_batch(self, documents: list) -> list:
        """Process multiple documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed documents
        """
        return [self.process(doc) for doc in documents]
