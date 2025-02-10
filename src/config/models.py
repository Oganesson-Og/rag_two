from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Enhanced model configuration with fallbacks."""
    
    primary_model: str
    fallback_model: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512
    use_fallback: bool = False
    
    def get_active_model(self) -> str:
        """Get the currently active model name."""
        return self.fallback_model if self.use_fallback else self.primary_model

@dataclass
class ProcessingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunks_per_doc: int = 50

@dataclass
class RetrievalConfig:
    top_k: int = 5
    score_threshold: float = 0.7
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3

@dataclass
class StorageConfig:
    vector_store_path: str = "vectors"
    document_store_path: str = "documents"
    cache_dir: str = "cache"

@dataclass
class RAGConfig:
    model: ModelConfig
    processing: ProcessingConfig
    retrieval: RetrievalConfig
    storage: StorageConfig 