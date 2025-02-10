from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import logging
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from datetime import datetime
#import h5py
from tqdm import tqdm
import threading
from queue import Queue
import pickle

class IndexType(Enum):
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    HYBRID = "hybrid"
    CURRICULUM = "curriculum"

@dataclass
class IndexConfig:
    """Configuration for educational vector index."""
    dimension: int = 768
    index_type: IndexType = IndexType.HYBRID
    nlist: int = 100  # For IVF
    M: int = 16      # For HNSW
    ef_construction: int = 200  # For HNSW
    ef_search: int = 128  # For HNSW search
    num_threads: int = 4
    batch_size: int = 1000
    cache_size_gb: float = 2.0

class EducationalVectorIndex:
    """Specialized vector index for educational content."""
    
    def __init__(
        self,
        config: Optional[IndexConfig] = None,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        index_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.logger = logging.getLogger(__name__)
        self.config = config or IndexConfig()
        self.device = device
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name).to(device)
        
        # Initialize directory
        self.index_dir = index_dir or Path("data/indices")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize indices
        self.main_index = self._create_index()
        self.subject_indices: Dict[str, faiss.Index] = {}
        self.grade_indices: Dict[int, faiss.Index] = {}
        
        # Initialize metadata storage
        self.metadata: Dict[int, Dict] = {}
        self.id_mapping: Dict[str, int] = {}
        self.reverse_mapping: Dict[int, str] = {}
        
        # Initialize thread pool for batch processing
        self.thread_pool = Queue()
        self._initialize_threads()
        
        # Initialize cache
        self.embedding_cache = {}
        
    def _create_index(self) -> faiss.Index:
        """Create appropriate FAISS index based on configuration."""
        d = self.config.dimension
        
        if self.config.index_type == IndexType.FLAT:
            return faiss.IndexFlatIP(d)
            
        elif self.config.index_type == IndexType.IVF:
            quantizer = faiss.IndexFlatIP(d)
            return faiss.IndexIVFFlat(
                quantizer,
                d,
                self.config.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            
        elif self.config.index_type == IndexType.HNSW:
            index = faiss.IndexHNSWFlat(
                d,
                self.config.M,
                faiss.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = self.config.ef_construction
            index.hnsw.efSearch = self.config.ef_search
            return index
            
        elif self.config.index_type == IndexType.HYBRID:
            # Combine IVF and HNSW
            quantizer = faiss.IndexHNSWFlat(
                d,
                self.config.M,
                faiss.METRIC_INNER_PRODUCT
            )
            index = faiss.IndexIVFFlat(
                quantizer,
                d,
                self.config.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            return index
            
        else:  # Curriculum-based index
            return self._create_curriculum_index(d)

    def _create_curriculum_index(self, dimension: int) -> faiss.Index:
        """Create curriculum-aware index structure."""
        # Create a composite index that maintains curriculum relationships
        index = faiss.IndexShards(dimension)
        
        # Add sub-indices for different educational levels
        for grade in range(1, 13):  # K-12
            grade_index = faiss.IndexHNSWFlat(
                dimension,
                self.config.M,
                faiss.METRIC_INNER_PRODUCT
            )
            self.grade_indices[grade] = grade_index
            index.add_shard(grade_index)
            
        return index

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict],
        batch_size: Optional[int] = None
    ) -> List[int]:
        """Add vectors to the index with educational metadata."""
        batch_size = batch_size or self.config.batch_size
        ids = []
        
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Generate IDs for the batch
            batch_ids = self._generate_ids(len(batch_vectors))
            ids.extend(batch_ids)
            
            # Add to main index
            self.main_index.add(batch_vectors)
            
            # Add to subject-specific indices
            self._add_to_subject_indices(
                batch_vectors,
                batch_ids,
                batch_metadata
            )
            
            # Add to grade-specific indices
            self._add_to_grade_indices(
                batch_vectors,
                batch_ids,
                batch_metadata
            )
            
            # Store metadata
            self._store_metadata(batch_ids, batch_metadata)
            
        return ids

    async def index_educational_content(
        self,
        content_list: List[Dict],
        batch_size: Optional[int] = None
    ) -> List[int]:
        """Index educational content with specialized processing."""
        batch_size = batch_size or self.config.batch_size
        all_ids = []
        
        for i in range(0, len(content_list), batch_size):
            batch = content_list[i:i + batch_size]
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(batch)
            
            # Process educational metadata
            metadata = self._process_educational_metadata(batch)
            
            # Add to index
            batch_ids = self.add_vectors(embeddings, metadata)
            all_ids.extend(batch_ids)
            
        return all_ids

    async def _generate_embeddings(
        self,
        content_list: List[Dict]
    ) -> np.ndarray:
        """Generate embeddings for educational content."""
        texts = []
        for content in content_list:
            # Combine relevant fields for embedding
            text_parts = [
                content.get("title", ""),
                content.get("content", ""),
                " ".join(content.get("learning_objectives", [])),
                " ".join(content.get("keywords", []))
            ]
            texts.append(" ".join(text_parts))
            
        # Generate embeddings in batches
        embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=True,
                device=self.device
            ).cpu().numpy()
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)

    def _process_educational_metadata(
        self,
        content_list: List[Dict]
    ) -> List[Dict]:
        """Process and enhance educational metadata."""
        processed_metadata = []
        
        for content in content_list:
            metadata = {
                "id": content.get("id"),
                "title": content.get("title"),
                "subject": content.get("subject"),
                "grade_level": content.get("grade_level"),
                "difficulty": content.get("difficulty", 0.5),
                "prerequisites": content.get("prerequisites", []),
                "learning_objectives": content.get("learning_objectives", []),
                "content_type": content.get("content_type"),
                "last_updated": datetime.now().isoformat(),
                "embedding_version": "1.0"
            }
            
            # Add derived metadata
            metadata.update(self._derive_educational_metadata(content))
            processed_metadata.append(metadata)
            
        return processed_metadata

    def _derive_educational_metadata(self, content: Dict) -> Dict:
        """Derive additional educational metadata."""
        derived = {}
        
        # Calculate complexity metrics
        text = content.get("content", "")
        derived["complexity_score"] = self._calculate_complexity(text)
        
        # Extract key concepts
        derived["key_concepts"] = self._extract_key_concepts(text)
        
        # Determine educational level
        derived["educational_level"] = self._determine_educational_level(content)
        
        return derived

    def _calculate_complexity(self, text: str) -> float:
        """Calculate educational content complexity."""
        # Implement complexity calculation logic
        # This could include:
        # - Vocabulary complexity
        # - Sentence structure analysis
        # - Concept density
        # For now, return a simple placeholder
        return 0.5

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key educational concepts from text."""
        # Implement concept extraction logic
        # This could include:
        # - NLP-based concept extraction
        # - Domain-specific keyword identification
        # For now, return a simple placeholder
        return []

    def _determine_educational_level(self, content: Dict) -> str:
        """Determine appropriate educational level."""
        grade_level = content.get("grade_level")
        if grade_level:
            return f"grade_{grade_level}"
        return "general"

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        subject: Optional[str] = None,
        grade_level: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> Tuple[List[int], List[float]]:
        """Search for similar educational content."""
        if subject and subject in self.subject_indices:
            index = self.subject_indices[subject]
        elif grade_level and grade_level in self.grade_indices:
            index = self.grade_indices[grade_level]
        else:
            index = self.main_index
            
        # Perform search
        D, I = index.search(query_vector.reshape(1, -1), k)
        
        # Apply filters
        if filters:
            I, D = self._apply_filters(I[0], D[0], filters)
        
        return I, D

    def _apply_filters(
        self,
        ids: np.ndarray,
        distances: np.ndarray,
        filters: Dict
    ) -> Tuple[List[int], List[float]]:
        """Apply educational filters to search results."""
        filtered_ids = []
        filtered_distances = []
        
        for id_, dist in zip(ids, distances):
            metadata = self.metadata.get(int(id_), {})
            if self._matches_filters(metadata, filters):
                filtered_ids.append(int(id_))
                filtered_distances.append(float(dist))
                
        return filtered_ids, filtered_distances

    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches educational filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if not any(v in metadata[key] for v in value):
                    return False
            elif metadata[key] != value:
                return False
        return True

    def save(self, path: Optional[Path] = None) -> None:
        """Save index and metadata."""
        save_path = path or self.index_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main index
        index_path = save_path / f"main_index_{timestamp}.faiss"
        faiss.write_index(self.main_index, str(index_path))
        
        # Save metadata
        metadata_path = save_path / f"metadata_{timestamp}.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_mapping": self.id_mapping,
                "reverse_mapping": self.reverse_mapping
            }, f)
            
        self.logger.info(f"Saved index and metadata to {save_path}")

    def load(self, path: Path) -> None:
        """Load index and metadata."""
        try:
            # Load main index
            index_files = list(path.glob("main_index_*.faiss"))
            if not index_files:
                raise FileNotFoundError("No index file found")
                
            latest_index = max(index_files, key=lambda x: x.stat().st_mtime)
            self.main_index = faiss.read_index(str(latest_index))
            
            # Load metadata
            metadata_files = list(path.glob("metadata_*.pkl"))
            if not metadata_files:
                raise FileNotFoundError("No metadata file found")
                
            latest_metadata = max(metadata_files, key=lambda x: x.stat().st_mtime)
            with open(latest_metadata, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self.id_mapping = data["id_mapping"]
                self.reverse_mapping = data["reverse_mapping"]
                
            self.logger.info(f"Loaded index and metadata from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading index: {str(e)}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "total_vectors": self.main_index.ntotal,
            "dimension": self.main_index.d,
            "num_subjects": len(self.subject_indices),
            "num_grades": len(self.grade_indices),
            "metadata_size": len(self.metadata),
            "index_type": self.config.index_type.value
        } 