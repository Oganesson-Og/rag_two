"""
Document Extractors Module
------------------------

Handles the extraction of text and metadata from different document formats.
Each extractor includes both extraction and necessary preprocessing steps.
"""

from abc import ABC, abstractmethod
import fitz  # PyMuPDF
from docx import Document
import pandas as pd
from PIL import Image
import pytesseract
from pathlib import Path
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
    
    def _clean_text(self, text):
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

class PDFExtractor(BaseExtractor):
    """Handles PDF documents including preprocessing steps."""
    
    def extract(
        self,
        content: bytes,
        options: Optional[Dict[str, bool]] = None
    ) -> ExtractorResult:
        try:
            options = options or {}
            
            # Extract text and metadata
            result = {
                'content': self._extract_text(content),
                'metadata': {
                    **self.get_metadata(),
                    'content_type': 'pdf'
                }
            }
            
            # Extract additional elements if requested
            if options.get('extract_images', True):
                result['images'] = self._extract_images(content)
                
            if options.get('extract_tables', True):
                result['tables'] = self._extract_tables(content)
                
            return result
            
        except Exception as e:
            self.logger.error(f"PDF extraction error: {str(e)}")
            raise

    def _extract_text(self, content: bytes) -> str:
        """Extract text from PDF."""
        # Implement text extraction
        return ""

    def _extract_images(self, content: bytes) -> List[Dict[str, Any]]:
        """Extract images from PDF."""
        # Implement image extraction
        return []

    def _extract_tables(self, content: bytes) -> List[Dict[str, Any]]:
        """Extract tables from PDF."""
        # Implement table extraction
        return []

class DocxExtractor(BaseExtractor):
    """Handles DOCX documents including preprocessing steps."""
    
    def extract(self, file_path):
        doc = Document(file_path)
        text = ""
        metadata = {
            'file_type': 'docx',
            'metadata': self._extract_docx_metadata(doc)
        }
        
        for paragraph in doc.paragraphs:
            processed_text = self._preprocess_paragraph(paragraph.text)
            text += processed_text
            
        # Handle tables
        tables = self._extract_tables(doc)
        
        # Handle images
        images = self._extract_images(doc)
        
        return {
            'text': self._clean_text(text),
            'metadata': metadata,
            'tables': tables,
            'images': images
        }

class ExcelExtractor:
    def extract(self, file_path: Path) -> Dict[str, Any]:
        df = pd.read_excel(file_path)
        return {"text": df.to_string(), "metadata": {"rows": len(df)}}

class CSVExtractor:
    def extract(self, file_path: Path) -> Dict[str, Any]:
        df = pd.read_csv(file_path)
        return {"text": df.to_string(), "metadata": {"rows": len(df)}}

class TextExtractor(BaseExtractor):
    """Extract text content."""
    
    def extract(
        self,
        content: Union[str, bytes],
        options: Optional[Dict[str, Any]] = None
    ) -> ExtractorResult:
        try:
            if isinstance(content, bytes):
                content = content.decode('utf-8')
                
            return {
                'text': content,
                'metadata': {
                    'extractor': self.__class__.__name__,
                    'timestamp': datetime.now().isoformat(),
                    'length': len(content)
                }
            }
        except Exception as e:
            self.logger.error(f"Text extraction error: {str(e)}")
            raise

class ImageExtractor(BaseExtractor):
    """Extract information from images."""
    
    def extract(
        self,
        content: ImageArray,
        options: Optional[Dict[str, Any]] = None
    ) -> ExtractorResult:
        try:
            options = options or {}
            
            return {
                'image_shape': content.shape,
                'metadata': {
                    'extractor': self.__class__.__name__,
                    'timestamp': datetime.now().isoformat(),
                    'options': options
                }
            }
        except Exception as e:
            self.logger.error(f"Image extraction error: {str(e)}")
            raise 