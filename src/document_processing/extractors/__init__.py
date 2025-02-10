"""
Document Extractors Package
------------------------

Collection of specialized document content extractors.

Available Extractors:
- PDFExtractor: PDF document processing
- DocxExtractor: Word document processing
- ExcelExtractor: Excel workbook processing
- CSVExtractor: CSV file processing
- BaseExtractor: Abstract base class

Usage:
    from document_processing.extractors import PDFExtractor
    
    extractor = PDFExtractor()
    result = extractor.extract('document.pdf')

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from .base import BaseExtractor, ExtractorResult, DocumentContent
from .pdf import PDFExtractor
from .docx import DocxExtractor
from .spreadsheet import ExcelExtractor, CSVExtractor

__all__ = [
    'BaseExtractor',
    'PDFExtractor',
    'DocxExtractor',
    'ExcelExtractor',
    'CSVExtractor',
    'ExtractorResult',
    'DocumentContent'
] 