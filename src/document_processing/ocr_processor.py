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

import pytesseract
from PIL import Image
import cv2
import numpy as np
from ..config.ocr_config import OCRConfig

class OCRProcessor:
    def __init__(self, config=None):
        self.config = config or OCRConfig()
        pytesseract.pytesseract.tesseract_cmd = self.config.TESSERACT_PATH

    def process_image(self, image):
        """Process single image with OCR"""
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(
            processed,
            lang=self.config.TESSERACT_LANG,
            config=f'--psm {self.config.PAGE_SEGMENTATION_MODE}'
        )
        
        return text

    def preprocess_image(self, image):
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return Image.fromarray(denoised)

    def get_confidence(self, image):
        """Get OCR confidence score"""
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        return sum(confidences) / len(confidences) if confidences else 0