"""
Vector Store Module
-----------------

Manages vector storage, indexing and similarity search operations.
"""

import numpy as np
import pickle
from typing import List, Dict, Callable, Optional, Union
import faiss

class VectorStore:
    def __init__(self, config_manager):
        self.config = config_manager
        self.index = None
        self.metadata = []
        self.index_size = 0
        
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors.astype('float32'))
        self.metadata.extend(metadata)
        self.index_size += len(vectors)
        
    def calculate_similarity(self, query_vector: np.ndarray, vectors: np.ndarray, metric: str = 'cosine') -> np.ndarray:
        if metric == 'cosine':
            return np.dot(vectors, query_vector) / (
                np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
            )
        return -np.linalg.norm(vectors - query_vector, axis=1)
        
    def search_nearest(self, query_vector: np.ndarray, k: int = 5, 
                      return_metadata: bool = True,
                      filter_condition: Optional[Callable] = None) -> List[Dict]:
        D, I = self.index.search(query_vector.reshape(1, -1).astype('float32'), k)
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0:
                continue
            metadata = self.metadata[idx]
            if filter_condition and not filter_condition(metadata):
                continue
            results.append({
                'id': idx,
                'score': float(1 / (1 + dist)),
                'metadata': metadata
            })
        return results

    def batch_search_nearest(self, query_vectors: np.ndarray, k: int = 5) -> List[List[Dict]]:
        return [self.search_nearest(qv, k) for qv in query_vectors]

    def build_index(self, vectors: np.ndarray, metadata: List[Dict]):
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.add_vectors(vectors, metadata)

    def save_index(self, path: str):
        faiss.write_index(self.index, path)

    def load_index(self, path: str):
        self.index = faiss.read_index(path)
        self.index_size = self.index.ntotal

    def update_vectors(self, indices: List[int], vectors: np.ndarray, metadata: List[Dict]):
        for i, idx in enumerate(indices):
            if idx < self.index_size:
                self.index.remove_ids(np.array([idx]))
                self.index.add(vectors[i:i+1].astype('float32'))
                self.metadata[idx] = metadata[i]

    def delete_vectors(self, indices: List[int]):
        self.index.remove_ids(np.array(indices))
        for idx in sorted(indices, reverse=True):
            del self.metadata[idx]
        self.index_size -= len(indices)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'metadata': self.metadata, 'index_size': self.index_size}, f)
        faiss.write_index(self.index, f"{path}.index")

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.index_size = data['index_size']
        self.index = faiss.read_index(f"{path}.index") 