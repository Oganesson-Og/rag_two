"""
Diagram Analysis Module
---------------------

Advanced diagram analysis system for scientific and educational content.
Processes technical diagrams, charts, and scientific illustrations.

Features:
- Scientific notation detection
- Relationship mapping
- Label identification
- Chemical structure recognition
- Equation extraction
- Component analysis
- Spatial relationship detection

Key Components:
1. Scientific Notation: Mathematical and technical symbols
2. Relationship Analysis: Component connections and flows
3. Label Detection: Text and annotation processing
4. Chemical Structure: Molecular and compound recognition
5. Equation Processing: Mathematical formula extraction

Technical Details:
- Computer vision algorithms
- OCR integration
- Graph-based analysis
- Pattern recognition
- Spatial analysis
- Symbol classification
- Neural network models

Dependencies:
- opencv-python>=4.8.0
- numpy>=1.24.0
- torch>=2.0.0
- scikit-image>=0.21.0

Author: Keith Satuku
Version: 1.5.0
Created: 2025
License: MIT
"""

import cv2
import numpy as np
import torch
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import pytesseract
from rdkit import Chem
from rdkit.Chem import Draw
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import re

@dataclass
class AnalysisResult:
    """Structure for analysis results"""
    notation: List[str]
    relationships: List[Dict]
    labels: List[Dict]
    chemical_structures: List[Dict]
    equations: List[Dict]
    confidence: float
    metadata: Dict

class DiagramAnalyzer:
    """Advanced diagram analysis system."""

    def __init__(self, model_config: Optional[Dict] = None):
        """Initialize the analyzer with models."""
        self.logger = logging.getLogger(__name__)
        self.model_config = model_config or {}
        
        try:
            self.scientific_notation_model = self._load_notation_model()
            self.relationship_analyzer = self._load_relationship_model()
            self.label_detector = self._load_label_model()
            self.chemical_structure_model = self._load_chemical_model()
            self.equation_extractor = self._load_equation_model()
        except Exception as e:
            self.logger.error(f"Failed to initialize DiagramAnalyzer: {str(e)}")
            raise

    def analyze_diagram(self, image: np.ndarray) -> AnalysisResult:
        """Main entry point for diagram analysis."""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Perform analysis
            results = {
                'notation': self.detect_scientific_notation(processed_image),
                'relationships': self.analyze_relationships(processed_image),
                'labels': self.identify_labels(processed_image),
                'chemical_structures': self.detect_chemical_structures(processed_image),
                'equations': self.extract_equations(processed_image)
            }
            
            # Calculate confidence
            confidence = self._calculate_confidence(results)
            
            # Add metadata
            metadata = self._generate_metadata(image, results)
            
            return AnalysisResult(
                **results,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing diagram: {str(e)}")
            raise

    def _load_notation_model(self):
        """
        Loads and configures Tesseract OCR with specific settings for scientific notation
        """
        try:
            # Configure Tesseract path if needed (especially on Windows)
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            
            # Test if Tesseract is working
            pytesseract.get_tesseract_version()
            
            # Configure custom parameters for scientific notation
            custom_config = {
                'oem': 3,  # OCR Engine Mode
                'psm': 6,  # Page Segmentation Mode (assume uniform block of text)
                'char_whitelist': '0123456789.×+-Ee',
                'language': 'eng'
            }
            
            return {
                'engine': pytesseract,
                'config': custom_config
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tesseract OCR: {str(e)}")

    def _load_relationship_model(self):
        """
        Loads DETR model for detecting arrows and relationships
        """
        try:
            # Initialize DETR model and processor
            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            
            # Move model to GPU if available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            
            # Define relationship categories
            categories = {
                0: "arrow",
                1: "double_arrow",
                2: "dashed_arrow",
                3: "connection_line"
            }
            
            return {
                "processor": processor,
                "model": model,
                "device": device,
                "categories": categories
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load DETR model: {str(e)}")

    def _load_label_model(self):
        """
        Loads and configures Detectron2 model for label detection
        """
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            
            # Configure model parameters
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only detect text regions
            
            predictor = DefaultPredictor(cfg)
            
            return {
                "predictor": predictor,
                "config": cfg,
                "ocr_engine": pytesseract
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load Detectron2 model: {str(e)}")

    def _load_chemical_model(self):
        """
        Loads and configures RDKit for chemical structure recognition
        """
        try:
            # Configure RDKit parameters
            Draw.DrawingOptions.bondLineWidth = 1.2
            Draw.DrawingOptions.atomLabelFontSize = 12
            
            return {
                "rdkit": Chem,
                "draw_utils": Draw,
                "supported_formats": ['.mol', '.sdf', '.png', '.jpg'],
                "min_confidence": 0.7
            }
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chemical structure recognition: {str(e)}")

    def _load_equation_model(self):
        """
        Loads models for mathematical equation recognition
        """
        try:
            config = {
                "model_type": "transformer",
                "max_sequence_length": 512,
                "confidence_threshold": 0.8,
                "device": 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            return {
                "config": config,
                "preprocessor": self._preprocess_for_math,
                "postprocessor": self._convert_to_latex,
                "validator": self._validate_equation
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load equation recognition model: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for analysis."""
        try:
            # Convert to RGB if needed
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                
            # Resize if too large
            max_dim = 1024
            height, width = image.shape[:2]
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                image = cv2.resize(image, None, fx=scale, fy=scale)
                
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {str(e)}")
            raise

    def detect_scientific_notation(self, image: np.ndarray) -> List[str]:
        """Handles scientific notation recognition"""
        try:
            # Convert image to format suitable for Tesseract
            img_pil = Image.fromarray(image)
            
            # Configure Tesseract for scientific notation
            config = self.scientific_notation_model['config']
            custom_config = f'--oem {config["oem"]} --psm {config["psm"]} -c tessedit_char_whitelist={config["char_whitelist"]}'
            
            # Extract text
            text = self.scientific_notation_model['engine'].image_to_string(
                img_pil, config=custom_config)
            
            # Process and validate scientific notation
            notations = []
            for potential_notation in text.split('\n'):
                if self._validate_scientific_notation(potential_notation):
                    notations.append(potential_notation)
            
            return notations
        except Exception as e:
            self.logger.error(f"Error detecting scientific notation: {str(e)}")
            return []

    def analyze_relationships(self, image: np.ndarray) -> List[Dict]:
        """Analyzes arrows and relationships between components"""
        try:
            # Prepare image for DETR
            processor = self.relationship_analyzer["processor"]
            model = self.relationship_analyzer["model"]
            
            # Process image
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            
            # Post-process to detect arrows and connections
            target_sizes = torch.tensor([image.shape[:2]])
            results = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes)[0]
            
            relationships = []
            categories = self.relationship_analyzer["categories"]
            
            for score, label, box in zip(results["scores"], results["labels"], 
                                       results["boxes"]):
                if score > 0.7:  # Confidence threshold
                    relationships.append({
                        "type": categories.get(label.item(), "unknown"),
                        "box": box.tolist(),
                        "confidence": score.item()
                    })
            
            return relationships
        except Exception as e:
            self.logger.error(f"Error analyzing relationships: {str(e)}")
            return []

    def identify_labels(self, image: np.ndarray) -> List[Dict]:
        """Detects and extracts labeled parts"""
        try:
            # Use Detectron2 for label detection
            outputs = self.label_detector["predictor"](image)
            
            # Process predictions
            instances = outputs["instances"].to("cpu")
            labels = []
            
            for i in range(len(instances)):
                box = instances.pred_boxes[i].tensor.numpy()[0]
                score = instances.scores[i].item()
                
                if score > 0.5:  # Confidence threshold
                    # Extract text within the detected box
                    cropped = image[int(box[1]):int(box[3]), 
                                  int(box[0]):int(box[2])]
                    text = self.label_detector["ocr_engine"].image_to_string(cropped)
                    
                    labels.append({
                        "text": text.strip(),
                        "box": box.tolist(),
                        "confidence": score
                    })
            
            return labels
        except Exception as e:
            self.logger.error(f"Error identifying labels: {str(e)}")
            return []

    def detect_chemical_structures(self, image: np.ndarray) -> List[Dict]:
        """Recognizes chemical structures and formulas"""
        try:
            # Preprocess image for chemical structure detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours that might represent chemical structures
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            chemical_structures = []
            rdkit = self.chemical_structure_model["rdkit"]
            min_confidence = self.chemical_structure_model["min_confidence"]
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Size threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    structure_img = image[y:y+h, x:x+w]
                    
                    # Try to convert to SMILES using RDKit
                    try:
                        mol = rdkit.MolFromImage(structure_img)
                        if mol:
                            smiles = rdkit.MolToSmiles(mol)
                            chemical_structures.append({
                                "smiles": smiles,
                                "box": [x, y, w, h],
                                "confidence": min_confidence
                            })
                    except:
                        continue
            
            return chemical_structures
        except Exception as e:
            self.logger.error(f"Error detecting chemical structures: {str(e)}")
            return []

    def extract_equations(self, image: np.ndarray) -> List[Dict]:
        """Extracts mathematical equations from the diagram"""
        try:
            # Preprocess image for equation detection
            processed_image = self.equation_extractor["preprocessor"](image)
            
            # Use specialized math OCR model
            config = self.equation_extractor["config"]
            validator = self.equation_extractor["validator"]
            postprocessor = self.equation_extractor["postprocessor"]
            
            # Placeholder for actual equation recognition
            # You would need to implement or import actual model processing here
            equations = []
            
            # Post-process and validate equations
            for eq in equations:
                if validator(eq):
                    latex = postprocessor(eq)
                    equations.append({
                        "text": eq,
                        "latex": latex,
                        "confidence": config["confidence_threshold"]
                    })
            
            return equations
        except Exception as e:
            self.logger.error(f"Error extracting equations: {str(e)}")
            return []

    def _calculate_confidence(self, results: Dict) -> float:
        """Calculate overall confidence score."""
        confidence_weights = {
            'notation': 0.2,
            'relationships': 0.25,
            'labels': 0.25,
            'chemical_structures': 0.15,
            'equations': 0.15
        }
        
        confidence_scores = []
        for key, weight in confidence_weights.items():
            if results[key]:
                if isinstance(results[key][0], dict) and 'confidence' in results[key][0]:
                    scores = [r['confidence'] for r in results[key]]
                    confidence_scores.append(np.mean(scores) * weight)
                else:
                    confidence_scores.append(weight)
                    
        return sum(confidence_scores)

    def _generate_metadata(self, image: np.ndarray, results: Dict) -> Dict:
        """Generate metadata about the analysis."""
        return {
            'image_size': image.shape,
            'component_counts': {
                k: len(v) for k, v in results.items()
            },
            'processing_info': {
                'models_used': [
                    'scientific_notation',
                    'relationship_analyzer',
                    'label_detector',
                    'chemical_structure',
                    'equation_extractor'
                ],
                'device': str(next(self.relationship_analyzer['model'].parameters()).device)
            }
        }

    def _validate_scientific_notation(self, text: str) -> bool:
        """Helper method to validate scientific notation"""
        pattern = r'^[+-]?\d*\.?\d+[Ee][+-]?\d+$'
        return bool(re.match(pattern, text.strip()))

    def _preprocess_for_math(self, image: np.ndarray) -> np.ndarray:
        """Helper method to preprocess image for math detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        return binary

    def _validate_equation(self, equation: str) -> bool:
        """
        Helper method to validate extracted equations.
        Checks for basic mathematical equation validity.
        """
        if not equation or equation.isspace():
            return False
            
        try:
            # Remove whitespace and check if empty
            eq = equation.strip()
            if not eq:
                return False
                
            # Basic validation checks
            # Check for balanced parentheses/brackets
            brackets = {'(': ')', '[': ']', '{': '}'}
            stack = []
            
            for char in eq:
                if char in brackets.keys():
                    stack.append(char)
                elif char in brackets.values():
                    if not stack:
                        return False
                    if char != brackets[stack.pop()]:
                        return False
                        
            if stack:  # Unmatched opening brackets
                return False
                
            # Check for invalid sequences of operators
            invalid_sequences = ['++', '--', '**', '//', '==', '+-', '-+', '*/', '/*']
            for seq in invalid_sequences:
                if seq in eq:
                    return False
                    
            # Check if equation contains at least one mathematical operator or equals sign
            operators = set('+-*/=≠≈≤≥')
            if not any(op in eq for op in operators):
                return False
                
            return True
            
        except Exception as e:
            self.logger.warning(f"Equation validation error: {str(e)}")
            return False

    def _convert_to_latex(self, equation: str) -> str:
        """
        Helper method to convert equation to LaTeX format.
        Handles common mathematical notations and symbols.
        """
        if not equation or not self._validate_equation(equation):
            return ""
            
        try:
            # Create a mapping of common mathematical symbols to LaTeX
            latex_map = {
                '×': r'\times ',
                '÷': r'\div ',
                '≠': r'\neq ',
                '≤': r'\leq ',
                '≥': r'\geq ',
                '±': r'\pm ',
                '∞': r'\infty ',
                '≈': r'\approx ',
                '→': r'\rightarrow ',
                '←': r'\leftarrow ',
                '√': r'\sqrt{',
                '∑': r'\sum ',
                '∫': r'\int ',
                'π': r'\pi ',
                'θ': r'\theta ',
                'α': r'\alpha ',
                'β': r'\beta ',
                'Δ': r'\Delta '
            }
            
            # Convert equation to LaTeX
            latex_eq = equation
            
            # Replace symbols with their LaTeX equivalents
            for symbol, latex in latex_map.items():
                latex_eq = latex_eq.replace(symbol, latex)
            
            # Handle superscripts (e.g., x^2)
            latex_eq = re.sub(r'(\w+)\^(\d+)', r'{\1}^{\2}', latex_eq)
            
            # Handle subscripts (e.g., x_1)
            latex_eq = re.sub(r'(\w+)_(\d+)', r'{\1}_{\2}', latex_eq)
            
            # Handle fractions (e.g., 1/2)
            latex_eq = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex_eq)
            
            # Handle square roots if not already in LaTeX format
            if r'\sqrt{' not in latex_eq:
                latex_eq = re.sub(r'sqrt\((.*?)\)', r'\\sqrt{\1}', latex_eq)
            
            # Wrap in math mode delimiters if not already present
            if not latex_eq.startswith('$') and not latex_eq.startswith(r'\['):
                latex_eq = f'${latex_eq}$'
            
            return latex_eq
            
        except Exception as e:
            self.logger.warning(f"LaTeX conversion error: {str(e)}")
            return equation  # Return original equation if conversion fails
    
