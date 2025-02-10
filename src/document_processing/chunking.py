
"""
Text Chunking Module
------------------

Intelligent text chunking system that splits documents into semantically meaningful chunks
while preserving context and maintaining optimal chunk sizes.

Key Features:
- Sentence-aware splitting
- Overlapping chunks
- Metadata preservation
- Chunk size optimization
- Small chunk merging
- Unique chunk identification
- Statistical analysis

Technical Details:
- Uses regex for sentence splitting
- Implements sliding window approach
- Maintains chunk boundaries
- Generates unique chunk IDs
- Provides chunk statistics
- Handles metadata

Dependencies:
- re
- typing
- dataclasses
- hashlib
- logging

Example Usage:
    # Basic chunking
    chunker = TextChunker()
    chunks = chunker.chunk_text(document_text)
    
    # Advanced chunking with metadata
    chunks = chunker.chunk_text(
        text=document_text,
        metadata={"source": "textbook"}
    )
    
    # Get chunk statistics
    stats = chunker.get_chunk_statistics(chunks)

Chunking Strategies:
- Sentence boundary detection
- Fixed-size windowing
- Overlap control
- Small chunk merging
- Metadata enrichment

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
""" 

from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass
import logging
from hashlib import md5

@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    chunk_id: str

    @property
    def length(self) -> int:
        return len(self.text)

class TextChunker:
    """Handles text chunking for document processing"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 overlap: int = 50,
                 respect_sentences: bool = True,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
        self.min_chunk_size = min_chunk_size
        self.logger = logging.getLogger(__name__)

    def chunk_text(self, 
                  text: str, 
                  metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into overlapping chunks"""
        if not text:
            return []

        metadata = metadata or {}
        
        try:
            if self.respect_sentences:
                return self._chunk_by_sentences(text, metadata)
            return self._chunk_by_size(text, metadata)
        except Exception as e:
            self.logger.error(f"Error chunking text: {str(e)}")
            raise

    def _chunk_by_sentences(self, 
                          text: str, 
                          metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks while respecting sentence boundaries"""
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_id = self._generate_chunk_id(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    metadata=metadata.copy(),
                    chunk_id=chunk_id
                ))
                
                current_chunk = []
                current_length = 0
                start_idx += len(chunk_text)
            
            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = self._generate_chunk_id(chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=start_idx + len(chunk_text),
                metadata=metadata.copy(),
                chunk_id=chunk_id
            ))

        return chunks

    def _chunk_by_size(self, 
                      text: str, 
                      metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into chunks of fixed size with overlap"""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            if end > text_length:
                end = text_length
            
            chunk_text = text[start:end]
            chunk_id = self._generate_chunk_id(chunk_text)
            
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=start,
                end_idx=end,
                metadata=metadata.copy(),
                chunk_id=chunk_id
            ))
            
            start = end - self.overlap

        return self._merge_small_chunks(chunks)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # More comprehensive sentence splitting
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are too small"""
        if not chunks:
            return chunks

        merged = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            if current_chunk.length < self.min_chunk_size:
                # Merge with next chunk
                current_chunk = Chunk(
                    text=current_chunk.text + " " + next_chunk.text,
                    start_idx=current_chunk.start_idx,
                    end_idx=next_chunk.end_idx,
                    metadata=current_chunk.metadata,
                    chunk_id=self._generate_chunk_id(current_chunk.text + next_chunk.text)
                )
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk

        merged.append(current_chunk)
        return merged

    def _generate_chunk_id(self, text: str) -> str:
        """Generate a unique ID for a chunk"""
        return md5(text.encode()).hexdigest()

    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}

        lengths = [chunk.length for chunk in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(lengths) / len(chunks),
            "min_chunk_size": min(lengths),
            "max_chunk_size": max(lengths),
            "total_length": sum(lengths)
        }
