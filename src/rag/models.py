"""
Enhanced Unified Data Model
-------------------------

Comprehensive data models for RAG pipeline with support for all modalities
and processing stages.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
import numpy as np
from uuid import uuid4

class ProcessingStage(str, Enum):
    """Extended processing stages for document lifecycle."""
    RAW = "raw"
    EXTRACTED = "extracted"
    PREPROCESSED = "preprocessed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    RETRIEVED = "retrieved"
    GENERATED = "generated"

class ContentModality(str, Enum):
    """Supported content modalities with metadata requirements."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    VIDEO = "video"
    
    @property
    def required_metadata(self) -> List[str]:
        """Required metadata fields for each modality."""
        return {
            "text": ["encoding", "language"],
            "audio": ["duration", "sample_rate", "channels"],
            "image": ["width", "height", "format"],
            "pdf": ["pages", "title"],
            "video": ["duration", "fps", "resolution"]
        }[self.value]

class ProcessingMetrics(BaseModel):
    """Metrics collected during document processing."""
    processing_time: float
    token_count: Optional[int] = None
    chunk_count: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    confidence_score: Optional[float] = None
    error_rate: Optional[float] = None

class ProcessingEvent(BaseModel):
    """Detailed processing event information."""
    timestamp: datetime = Field(default_factory=datetime.now)
    stage: ProcessingStage
    processor: str
    metrics: Optional[ProcessingMetrics] = None
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"
    error: Optional[str] = None

class Document(BaseModel):
    """Enhanced unified document representation."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: Any
    modality: ContentModality
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    chunks: List["Chunk"] = Field(default_factory=list)
    processing_history: List[ProcessingEvent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: str = "1.0"
    
    @validator('metadata')
    def validate_required_metadata(cls, v, values):
        """Ensure required metadata fields are present."""
        if 'modality' in values:
            required_fields = values['modality'].required_metadata
            missing_fields = [f for f in required_fields if f not in v]
            if missing_fields:
                raise ValueError(f"Missing required metadata: {missing_fields}")
        return v
    
    def add_processing_event(self, event: ProcessingEvent):
        """Add a processing event to history."""
        self.processing_history.append(event)
        self.updated_at = datetime.now()

class Chunk(BaseModel):
    """Enhanced chunk representation with position and relations."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    relationships: Dict[str, List[str]] = Field(default_factory=dict)
    confidence_score: Optional[float] = None
    
    def update_embedding(self, embedding: List[float]):
        """Update chunk embedding with new vector."""
        self.embeddings = embedding
        self.metadata["last_embedded"] = datetime.now().isoformat()

class EmbeddingVector(BaseModel):
    """Vector embedding with metadata."""
    vector: List[float]
    model: str
    dimension: int
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchResult(BaseModel):
    """Search result with relevance information."""
    chunk: Chunk
    similarity_score: float
    ranking_position: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GenerationResult(BaseModel):
    """Generation result with source tracking."""
    text: str
    chunks_used: List[Chunk]
    confidence_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

# Circular reference resolution
Document.update_forward_refs() 