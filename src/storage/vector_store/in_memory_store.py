import uuid
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from numpy.typing import NDArray
from .base import BaseVectorStore, Vector

class InMemoryVectorStore(BaseVectorStore):
    """A simple in-memory vector store for testing or small-scale usage."""

    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def add_vector(self, vector: Vector, metadata: Dict[str, Any] = None) -> str:
        vector_id = str(uuid.uuid4())
        self.store[vector_id] = {
            'id': vector_id,
            'vector': vector,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        return vector_id

    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        return self.store.get(vector_id)

    def search(
        self, 
        query_vector: Vector, 
        k: int = 10, 
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        # A naive cosine similarity implementation for illustration.
        import numpy as np

        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        results = []
        for entry in self.store.values():
            sim = cosine_similarity(query_vector, entry['vector'])
            results.append({
                'id': entry['id'],
                'vector': entry['vector'],
                'metadata': entry['metadata'],
                'score': float(sim)
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]

    def delete_vectors(self, ids: List[str]) -> int:
        count = 0
        for vector_id in ids:
            if vector_id in self.store:
                del self.store[vector_id]
                count += 1
        return count

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        if vector_id in self.store:
            self.store[vector_id]['metadata'].update(metadata)
            return True
        return False 