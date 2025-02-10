from typing import Optional, Dict, Any, List, Union
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
from enum import Enum

class DiagramType(Enum):
    FLOWCHART = "flowchart"
    TECHNICAL = "technical"
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    UNKNOWN = "unknown"

@dataclass
class DiagramElement:
    """Represents an element in a diagram."""
    element_type: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    text: Optional[str] = None
    relationships: List[str] = None

class DiagramAnalyzerV2:
    """Enhanced diagram analyzer with multiple processing options."""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        use_basic: bool = False
    ):
        self.logger = logging.getLogger(__name__)
        self.use_basic = use_basic
        
        # Initialize OCR
        try:
            import pytesseract
            self.ocr = pytesseract
        except ImportError:
            self.logger.warning("Tesseract OCR not available")
            self.ocr = None
            
        # Initialize basic OpenCV
        self.cv2 = cv2
        
        # Try to initialize advanced models
        if not use_basic:
            try:
                from transformers import DetrImageProcessor, DetrForObjectDetection
                self.processor = DetrImageProcessor.from_pretrained(
                    "facebook/detr-resnet-50"
                )
                self.model = DetrForObjectDetection.from_pretrained(
                    "facebook/detr-resnet-50"
                )
                self.use_basic = False
            except Exception as e:
                self.logger.warning(f"Advanced models not available: {str(e)}")
                self.use_basic = True

    def process_diagram(
        self,
        image_path: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """Process diagram with available methods."""
        try:
            # Load image if path provided
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path)
            else:
                image = image_path
                
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Process based on available methods
            if self.use_basic:
                return self._basic_processing(image)
            else:
                return self._advanced_processing(image)
                
        except Exception as e:
            self.logger.error(f"Error processing diagram: {str(e)}")
            return {
                "error": str(e),
                "type": DiagramType.UNKNOWN.value,
                "elements": [],
                "confidence": 0.0
            }

    def _basic_processing(self, image: Image.Image) -> Dict[str, Any]:
        """Basic diagram processing using OpenCV."""
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Basic shape detection
        elements = []
        
        # Detect edges
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Process contours
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Determine shape type
            approx = cv2.approxPolyDP(
                contour,
                0.04 * cv2.arcLength(contour, True),
                True
            )
            
            if len(approx) == 4:
                element_type = "rectangle"
            elif len(approx) == 3:
                element_type = "triangle"
            else:
                element_type = "other"
                
            elements.append(DiagramElement(
                element_type=element_type,
                confidence=0.7,
                bbox=[x, y, x + w, y + h]
            ))
            
        # Extract text if OCR available
        if self.ocr:
            try:
                text = self.ocr.image_to_string(image)
                if text.strip():
                    elements.append(DiagramElement(
                        element_type="text",
                        confidence=0.8,
                        bbox=[0, 0, image.width, image.height],
                        text=text.strip()
                    ))
            except Exception as e:
                self.logger.warning(f"OCR failed: {str(e)}")
                
        return {
            "type": self._determine_diagram_type(elements),
            "elements": [vars(elem) for elem in elements],
            "confidence": 0.7
        }

    def _advanced_processing(self, image: Image.Image) -> Dict[str, Any]:
        """Advanced diagram processing using DETR."""
        # Prepare image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Get predictions
        outputs = self.model(**inputs)
        
        # Process results
        elements = []
        for score, label, box in zip(
            outputs.logits[0].softmax(-1),
            outputs.pred_boxes[0]
        ):
            if score.max() > 0.7:  # Confidence threshold
                elements.append(DiagramElement(
                    element_type=self.model.config.id2label[label.argmax().item()],
                    confidence=score.max().item(),
                    bbox=box.tolist()
                ))
                
        return {
            "type": self._determine_diagram_type(elements),
            "elements": [vars(elem) for elem in elements],
            "confidence": float(outputs.logits.softmax(-1).max())
        }

    def _determine_diagram_type(
        self,
        elements: List[DiagramElement]
    ) -> str:
        """Determine diagram type based on elements."""
        # Simple heuristic for diagram type
        element_types = [elem.element_type for elem in elements]
        
        if any("arrow" in et.lower() for et in element_types):
            return DiagramType.FLOWCHART.value
        elif any("equation" in et.lower() for et in element_types):
            return DiagramType.MATHEMATICAL.value
        elif any("measurement" in et.lower() for et in element_types):
            return DiagramType.TECHNICAL.value
        elif any("graph" in et.lower() for et in element_types):
            return DiagramType.SCIENTIFIC.value
            
        return DiagramType.UNKNOWN.value

    def get_capabilities(self) -> Dict[str, bool]:
        """Get current capabilities of the analyzer."""
        return {
            "advanced_processing": not self.use_basic,
            "ocr_available": self.ocr is not None,
            "shape_detection": True,
            "text_extraction": self.ocr is not None
        } 
    