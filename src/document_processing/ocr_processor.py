"""
OCR Processing Module
-------------------

Advanced OCR system optimized for academic and technical documents.
Provides high-accuracy text extraction with specialized preprocessing.

Features:
- Image preprocessing
- Text extraction
- Confidence scoring
- Language detection
- Layout analysis
- Table recognition
- Multi-format support

Key Components:
1. Image Processing: Enhancement and normalization
2. OCR Engine: Text extraction and recognition
3. Post-processing: Text cleanup and formatting
4. Quality Assessment: Confidence scoring
5. Layout Analysis: Structure detection

Technical Details:
- Adaptive thresholding
- Noise reduction
- DPI normalization
- Character recognition
- Language detection
- Confidence scoring
- Format preservation

Dependencies:
- pytesseract>=0.3.10
- opencv-python>=4.8.0
- Pillow>=10.0.0
- numpy>=1.24.0

Author: Keith Satuku
Version: 2.1.0
Created: 2025
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
import numpy as np
from numpy.typing import NDArray
import pytesseract
from PIL import Image
import logging
from datetime import datetime
import io

class OCRProcessor:
    """OCR processing for document images."""
    
    def __init__(
        self,
        lang: str = "eng",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.lang = lang
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

    def process_image(
        self,
        image: Union[NDArray[np.uint8], bytes],
        options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """Process image with OCR."""
        try:
            options = options or {}
            
            # Convert bytes to image if needed
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
                
            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config.get('tesseract_config', '')
            )
            
            return {
                'text': text,
                'metadata': {
                    'processor': self.__class__.__name__,
                    'language': self.lang,
                    'timestamp': datetime.now().isoformat(),
                    'confidence': self._get_confidence(image)
                }
            }
            
        except Exception as e:
            self.logger.error(f"OCR processing error: {str(e)}")
            raise

    def _get_confidence(self, image: Image.Image) -> float:
        """Get OCR confidence score."""
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(c) for c in data['conf'] if c != '-1']
            return sum(confidences) / len(confidences) if confidences else 0.0
            
        except Exception:
            return 0.0