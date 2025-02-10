from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..config.settings import QDRANT_CONFIG

class EducationVectorStore:
    """Enhanced vector storage system optimized for educational content."""
    
    def __init__(
        self,
        collection_name: str = "education",
        config: Optional[Dict] = None
    ):
        self.config = config or QDRANT_CONFIG
        self.client = QdrantClient(
            host=self.config["host"],
            port=self.config["port"],
            prefer_grpc=self.config["prefer_grpc"]
        )
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Ensure collection exists with optimal configuration."""
        try:
            collection_config = self.config["collections"][self.collection_name]
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=collection_config["vector_size"],
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=collection_config["optimizers_config"]["default_segment_number"],
                    memmap_threshold=collection_config["optimizers_config"]["memmap_threshold"]
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {str(e)}")

    def upsert_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> None:
        """Insert or update vectors with educational metadata."""
        if not ids:
            ids = [str(i) for i in range(len(vectors))]
            
        points = [
            models.PointStruct(
                id=id_,
                vector=vector.tolist(),
                payload={
                    **meta,
                    "timestamp": datetime.utcnow().isoformat(),
                    "vector_version": "1.0"
                }
            )
            for id_, vector, meta in zip(ids, vectors, metadata)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query_vector: np.ndarray,
        filters: Optional[Dict] = None,
        limit: int = 5,
        score_threshold: float = 0.7
    ) -> List[Dict]:
        """Enhanced search with educational context awareness."""
        search_params = models.SearchParams(
            hnsw_ef=128,
            exact=False
        )
        
        query_filter = None
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                    for key, value in filters.items()
                ]
            )

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            search_params=search_params
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload,
                "vector": hit.vector
            }
            for hit in results
        ]

    def batch_search(
        self,
        query_vectors: List[np.ndarray],
        filters: Optional[Dict] = None,
        limit: int = 5
    ) -> List[List[Dict]]:
        """Perform batch search operations."""
        return [
            self.search(qv, filters, limit)
            for qv in query_vectors
        ] 
    