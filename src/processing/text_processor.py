"""
Text Processing Module
-------------------
"""

from typing import List, Dict, Union, Optional
from pathlib import Path
import re
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    chunk_size: int = 512
    overlap: int = 50
    min_chunk_size: int = 100
    language: str = "en"

class TextProcessor:
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        
    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if not text:
            return []
            
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.config.chunk_size
            chunk = text[start:end]
            
            # Adjust chunk to end at sentence boundary if possible
            if end < len(text):
                sentence_end = re.search(r'[.!?]\s+', chunk[::-1])
                if sentence_end:
                    end = end - sentence_end.start()
                    chunk = text[start:end]
                    
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
                
            start = end - self.config.overlap
            
        return chunks
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better processing."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        # Normalize apostrophes
        text = re.sub(r'[''`]', "'", text)
        return text.strip()
        
    def extract_metadata(self, text: str) -> Dict[str, Union[int, str, List[str]]]:
        """Extract metadata from text."""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        return {
            'length': len(text),
            'sentence_count': len(sentences),
            'word_count': len(words),
            'language': self.detect_language(text),
            'keywords': self.extract_keywords(text)
        }
        
    def detect_language(self, text: str) -> str:
        """Detect text language."""
        # Placeholder - implement with langdetect or similar
        return self.config.language
        
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Placeholder - implement with keyword extraction
        words = text.lower().split()
        # Remove common words and return unique
        return list(set(words))[:10] 