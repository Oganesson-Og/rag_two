"""
Text Document Extractor Module
---------------------------

Specialized extractor for plain text documents with advanced processing.

Key Features:
- Text extraction
- Encoding detection
- Format detection
- Structure preservation
- Metadata parsing
- Character set handling
- Line ending normalization

Technical Details:
- Multiple encoding support
- Unicode handling
- Line ending management
- Whitespace normalization
- Character set detection
- Error handling
- Performance optimization

Dependencies:
- chardet>=4.0.0
- typing-extensions>=4.7.0

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import chardet
from .base import BaseExtractor, ExtractorResult, DocumentContent

class TextExtractor(BaseExtractor):
    """Handles plain text documents with advanced processing capabilities."""
    
    def extract(
        self,
        content: Union[str, Path, bytes],
        options: Optional[Dict[str, bool]] = None
    ) -> ExtractorResult:
        """
        Extract and process text content.
        
        Args:
            content: Input text content
            options: Processing options
            
        Returns:
            Processed text content and metadata
        """
        try:
            options = options or {}
            
            # Handle different input types
            if isinstance(content, (str, Path)):
                with open(content, 'rb') as f:
                    raw_content = f.read()
            else:
                raw_content = content if isinstance(content, bytes) else content.encode()

            # Detect encoding
            encoding_info = self._detect_encoding(raw_content)
            
            # Decode content
            text_content = raw_content.decode(encoding_info['encoding'])
            
            # Process text
            processed_text = self._process_text(
                text_content,
                normalize_whitespace=options.get('normalize_whitespace', True),
                normalize_endings=options.get('normalize_endings', True)
            )

            return {
                'content': processed_text,
                'metadata': {
                    **self.get_metadata(),
                    **encoding_info,
                    'content_type': 'text',
                    'length': len(processed_text),
                    'line_count': processed_text.count('\n') + 1
                }
            }
            
        except Exception as e:
            self.logger.error(f"Text extraction error: {str(e)}")
            raise

    def _detect_encoding(self, content: bytes) -> Dict[str, Any]:
        """Detect text encoding."""
        try:
            result = chardet.detect(content)
            return {
                'encoding': result['encoding'] or 'utf-8',
                'confidence': result['confidence'],
                'language': result.get('language')
            }
        except Exception:
            return {
                'encoding': 'utf-8',
                'confidence': 1.0,
                'language': None
            }

    def _process_text(
        self,
        text: str,
        normalize_whitespace: bool = True,
        normalize_endings: bool = True
    ) -> str:
        """Process text content."""
        if normalize_endings:
            # Normalize line endings to \n
            text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        if normalize_whitespace:
            # Normalize whitespace while preserving line breaks
            lines = text.split('\n')
            lines = [self._clean_text(line) for line in lines]
            text = '\n'.join(lines)
        
        return text.strip() 