from typing import List, Dict, Optional, Union
import re
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import logging

@dataclass
class EducationalChunk:
    """Represents an educational content chunk with metadata."""
    content: str
    chunk_type: str  # e.g., 'definition', 'example', 'exercise', 'concept'
    subject_area: str
    difficulty_level: float  # 0.0 to 1.0
    prerequisites: List[str]
    learning_objectives: List[str]
    metadata: Dict
    start_idx: int
    end_idx: int

class EducationalChunker:
    """Intelligent chunking system optimized for educational content."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_size: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.device = device
        
        # Educational content markers
        self.section_markers = {
            'definition': r'Definition:|is defined as|refers to',
            'example': r'Example:|For example|Consider the following',
            'exercise': r'Exercise:|Problem:|Solve the following',
            'concept': r'Concept:|Key Point:|Important:',
            'theorem': r'Theorem:|Lemma:|Proposition:',
            'proof': r'Proof:|Demonstration:',
        }
        
        self.logger = logging.getLogger(__name__)

    def chunk_educational_content(
        self,
        content: Union[str, bytes],
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> List[EducationalChunk]:
        """Main chunking method for educational content."""
        if isinstance(content, bytes):
            if content_type == 'pdf':
                processed_content = self._process_pdf(content)
            else:
                raise ValueError(f"Unsupported binary content type: {content_type}")
        else:
            processed_content = content

        # Initial semantic segmentation
        segments = self._semantic_segmentation(processed_content)
        
        # Chunk segments while preserving educational context
        chunks = []
        for segment in segments:
            segment_chunks = self._create_semantic_chunks(
                segment,
                metadata or {}
            )
            chunks.extend(segment_chunks)
            
        # Post-process chunks for consistency
        final_chunks = self._post_process_chunks(chunks)
        
        return final_chunks

    def _semantic_segmentation(self, content: str) -> List[Dict]:
        """Segment content based on educational patterns."""
        segments = []
        current_position = 0
        
        # Split content into potential segments
        lines = content.split('\n')
        current_segment = []
        current_type = 'general'
        
        for line in lines:
            # Detect segment type based on markers
            detected_type = self._detect_segment_type(line)
            
            if detected_type != current_type and current_segment:
                # Save current segment
                segment_text = '\n'.join(current_segment)
                segments.append({
                    'content': segment_text,
                    'type': current_type,
                    'start': current_position,
                    'end': current_position + len(segment_text)
                })
                current_position += len(segment_text) + 1  # +1 for newline
                current_segment = []
                
            current_type = detected_type
            current_segment.append(line)
        
        # Add final segment
        if current_segment:
            segment_text = '\n'.join(current_segment)
            segments.append({
                'content': segment_text,
                'type': current_type,
                'start': current_position,
                'end': current_position + len(segment_text)
            })
            
        return segments

    def _detect_segment_type(self, text: str) -> str:
        """Detect the type of educational content in text."""
        for segment_type, patterns in self.section_markers.items():
            if re.search(patterns, text, re.IGNORECASE):
                return segment_type
        return 'general'

    def _create_semantic_chunks(
        self,
        segment: Dict,
        metadata: Dict
    ) -> List[EducationalChunk]:
        """Create chunks while preserving semantic meaning."""
        content = segment['content']
        chunk_type = segment['type']
        
        # Tokenize content
        tokens = self.tokenizer.encode(content)
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Determine optimal chunk size
            chunk_size = self._get_optimal_chunk_size(
                tokens[start_idx:start_idx + self.max_chunk_size]
            )
            
            # Extract chunk
            chunk_tokens = tokens[start_idx:start_idx + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Analyze chunk characteristics
            chunk_difficulty = self._estimate_difficulty(chunk_text)
            prerequisites = self._extract_prerequisites(chunk_text)
            objectives = self._extract_learning_objectives(chunk_text)
            
            # Create chunk object
            chunk = EducationalChunk(
                content=chunk_text,
                chunk_type=chunk_type,
                subject_area=metadata.get('subject_area', 'general'),
                difficulty_level=chunk_difficulty,
                prerequisites=prerequisites,
                learning_objectives=objectives,
                metadata=metadata,
                start_idx=segment['start'] + start_idx,
                end_idx=segment['start'] + start_idx + len(chunk_text)
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += chunk_size - self.overlap_size

        return chunks

    def _get_optimal_chunk_size(self, tokens: List[int]) -> int:
        """Determine optimal chunk size based on content structure."""
        if len(tokens) <= self.min_chunk_size:
            return len(tokens)
            
        # Look for natural break points
        for i in range(self.max_chunk_size, self.min_chunk_size, -1):
            if i >= len(tokens):
                continue
                
            # Check if position is a good break point
            token_text = self.tokenizer.decode([tokens[i]])
            if token_text in ['.', '?', '!', '\n']:
                return i
                
        return min(self.max_chunk_size, len(tokens))

    def _estimate_difficulty(self, text: str) -> float:
        """Estimate content difficulty level."""
        # Implement difficulty estimation logic
        # This could use factors like:
        # - Vocabulary complexity
        # - Sentence structure
        # - Mathematical notation density
        # - Technical term frequency
        return 0.5  # Placeholder

    def _extract_prerequisites(self, text: str) -> List[str]:
        """Extract prerequisite concepts from text."""
        # Implement prerequisite extraction logic
        return []  # Placeholder

    def _extract_learning_objectives(self, text: str) -> List[str]:
        """Extract learning objectives from text."""
        # Implement learning objectives extraction logic
        return []  # Placeholder

    def _post_process_chunks(
        self,
        chunks: List[EducationalChunk]
    ) -> List[EducationalChunk]:
        """Post-process chunks for consistency and quality."""
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Ensure chunk completeness
            if i > 0:
                # Check for incomplete sentences at start
                if not chunk.content[0].isupper():
                    # Merge with previous chunk if needed
                    continue
                    
            # Validate chunk
            if self._is_valid_chunk(chunk):
                processed_chunks.append(chunk)
                
        return processed_chunks

    def _is_valid_chunk(self, chunk: EducationalChunk) -> bool:
        """Validate chunk quality and completeness."""
        # Check minimum content length
        if len(chunk.content.strip()) < 50:
            return False
            
        # Check for balanced mathematical expressions
        if not self._has_balanced_math(chunk.content):
            return False
            
        # Ensure chunk doesn't break mid-sentence
        if not self._has_complete_sentences(chunk.content):
            return False
            
        return True

    def _process_pdf(self, content: bytes) -> str:
        """Process PDF content."""
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise

    def _has_balanced_math(self, text: str) -> bool:
        """Check if mathematical expressions are balanced."""
        stack = []
        math_delimiters = {'(': ')', '[': ']', '{': '}', '\\begin{': '\\end{'}
        
        for char in text:
            if char in math_delimiters:
                stack.append(char)
            elif char in math_delimiters.values():
                if not stack:
                    return False
                if char != math_delimiters[stack.pop()]:
                    return False
                    
        return len(stack) == 0

    def _has_complete_sentences(self, text: str) -> bool:
        """Check if text contains complete sentences."""
        # Simple heuristic: check if text ends with sentence terminator
        return bool(re.search(r'[.!?]\s*$', text.strip())) 