"""
Cross-Modal Processing Module
-------------------------

Enhanced cross-modal reasoning system for educational content processing.
"""

from typing import Dict, List, Optional, Union, Any
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, WhisperModel
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import logging
from pathlib import Path

class CrossModalProcessor:
    """Advanced cross-modal processing for educational content."""
    
    def __init__(
        self,
        text_model: str = "sentence-transformers/all-mpnet-base-v2",
        vision_model: str = "openai/clip-vit-base-patch32",
        audio_model: str = "openai/whisper-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.text_model = AutoModel.from_pretrained(text_model).to(device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        
        self.vision_model = AutoModel.from_pretrained(vision_model).to(device)
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model)
        
        self.audio_model = WhisperModel.from_pretrained(audio_model).to(device)
        
    async def process_multimodal_query(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Path, Image.Image]] = None,
        audio: Optional[Union[str, Path, np.ndarray]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Process multi-modal educational query."""
        try:
            results = {}
            
            # Default modality weights
            weights = weights or {
                "text": 0.4,
                "image": 0.3,
                "audio": 0.3
            }
            
            # Process each modality
            if text:
                results["text"] = await self._process_text(text)
            
            if image:
                results["image"] = await self._process_image(image)
                
            if audio:
                results["audio"] = await self._process_audio(audio)
                
            # Combine embeddings with weights
            combined_embedding = self._combine_embeddings(results, weights)
            
            return {
                "embeddings": combined_embedding,
                "modality_scores": self._calculate_modality_scores(results),
                "confidence": self._calculate_confidence(results)
            }
            
        except Exception as e:
            self.logger.error(f"Cross-modal processing error: {str(e)}")
            raise

    async def _process_text(self, text: str) -> Dict[str, Any]:
        """Process text input."""
        try:
            # Tokenize and encode text
            inputs = self.text_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.text_model(**inputs)
                
            return {
                "embedding": outputs.last_hidden_state.mean(dim=1).cpu().numpy(),
                "attention": outputs.attentions[-1].cpu().numpy() if outputs.attentions else None,
                "confidence": self._calculate_text_confidence(outputs)
            }
            
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            raise

    async def _process_image(
        self,
        image: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """Process image input."""
        try:
            # Load and preprocess image
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
                
            # Process image
            inputs = self.vision_processor(image, return_tensors="pt").to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.vision_model(**inputs)
                
            return {
                "embedding": outputs.last_hidden_state.mean(dim=1).cpu().numpy(),
                "attention": outputs.attentions[-1].cpu().numpy() if outputs.attentions else None,
                "confidence": self._calculate_image_confidence(outputs)
            }
            
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            raise

    async def _process_audio(
        self,
        audio: Union[str, Path, np.ndarray]
    ) -> Dict[str, Any]:
        """Process audio input."""
        try:
            # Load audio if path provided
            if isinstance(audio, (str, Path)):
                audio = self._load_audio(audio)
                
            # Process audio
            with torch.no_grad():
                result = self.audio_model.transcribe(audio)
                
            return {
                "embedding": result.embeddings,
                "transcription": result.text,
                "confidence": result.confidence
            }
            
        except Exception as e:
            self.logger.error(f"Audio processing error: {str(e)}")
            raise

    def _combine_embeddings(
        self,
        results: Dict[str, Dict[str, Any]],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Combine embeddings from different modalities."""
        combined = None
        total_weight = 0
        
        for modality, weight in weights.items():
            if modality in results:
                embedding = results[modality]["embedding"]
                if combined is None:
                    combined = embedding * weight
                else:
                    combined += embedding * weight
                total_weight += weight
                
        return combined / total_weight if total_weight > 0 else None

    def _calculate_modality_scores(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for each modality."""
        return {
            modality: data["confidence"]
            for modality, data in results.items()
            if "confidence" in data
        }

    def _calculate_confidence(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score."""
        scores = [
            data["confidence"]
            for data in results.values()
            if "confidence" in data
        ]
        return np.mean(scores) if scores else 0.0

    def _calculate_text_confidence(self, outputs: Any) -> float:
        """Calculate confidence score for text processing."""
        attention_weights = outputs.attentions[-1] if outputs.attentions else None
        if attention_weights is not None:
            return float(attention_weights.mean().cpu().numpy())
        return 0.8  # Default confidence

    def _calculate_image_confidence(self, outputs: Any) -> float:
        """Calculate confidence score for image processing."""
        attention_weights = outputs.attentions[-1] if outputs.attentions else None
        if attention_weights is not None:
            return float(attention_weights.mean().cpu().numpy())
        return 0.7  # Default confidence

    def _load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load audio file."""
        import librosa
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            return audio
        except Exception as e:
            self.logger.error(f"Audio loading error: {str(e)}")
            raise 