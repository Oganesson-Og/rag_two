import numpy as np
import faiss
from typing import List, Dict, Optional, Callable
from pathlib import Path

class VectorSearch:
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
        self.index = None
        self.metadata = []
        self.index_size = 0
        
    def create_index(self, dim: int, index_type: str = 'flat', **kwargs) -> faiss.Index:
        if index_type == 'flat':
            self.index = faiss.IndexFlatL2(dim)
        elif index_type == 'ivf':
            nlist = kwargs.get('nlist', 100)
            quantizer = faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        elif index_type == 'hnsw':
            M = kwargs.get('M', 16)
            self.index = faiss.IndexHNSWFlat(dim, M)
            self.index.hnsw.efConstruction = kwargs.get('ef_construction', 200)
        return self.index
        
    def add_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None):
        if self.index is None:
            raise ValueError("Index not initialized")
        if vectors.shape[1] != self.index.d:
            raise ValueError(f"Vector dimension mismatch: {vectors.shape[1]} vs {self.index.d}")
            
        self.index.add(vectors)
        if metadata:
            self.metadata.extend(metadata)
        self.index_size += len(vectors)
        
    def search(self, query_vector: np.ndarray, k: int = 5,
              filter_condition: Optional[Callable] = None) -> List[Dict]:
        if k < 1:
            raise ValueError("k must be positive")
            
        D, I = self.index.search(query_vector.reshape(1, -1), k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(D[0], I[0])):
            if idx < 0:
                continue
            result = {
                'id': idx,
                'score': float(1 / (1 + dist)),
                'metadata': self.metadata[idx] if self.metadata else None
            }
            if not filter_condition or filter_condition(result):
                results.append(result)
                
        return results
        
    def batch_search(self, query_vectors: np.ndarray, k: int = 5) -> List[List[Dict]]:
        return [self.search(qv, k) for qv in query_vectors]
        
    def save_index(self, path: str):
        faiss.write_index(self.index, path)
        
    def load_index(self, path: str):
        self.index = faiss.read_index(path)
        self.index_size = self.index.ntotal
        
    def train_index(self, training_vectors: np.ndarray):
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.train(training_vectors) 