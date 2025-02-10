import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class VectorOperations:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        
    def batch_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        return np.dot(vectors, query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
        
    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            return vectors / np.linalg.norm(vectors)
        return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        
    def top_k_similar(self, query: np.ndarray, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        similarities = self.batch_similarity(query, vectors)
        indices = np.argsort(similarities)[-k:][::-1]
        return indices, similarities[indices]
        
    def calculate_similarity(self, v1: np.ndarray, v2: np.ndarray, metric: 'SimilarityMetric') -> float:
        if metric == SimilarityMetric.COSINE:
            return self.cosine_similarity(v1, v2)
        elif metric == SimilarityMetric.DOT_PRODUCT:
            return np.dot(v1, v2)
        elif metric == SimilarityMetric.EUCLIDEAN:
            return np.linalg.norm(v1 - v2)
            
    def aggregate_vectors(self, vectors: np.ndarray, method: str = 'mean', weights: Optional[np.ndarray] = None) -> np.ndarray:
        if weights is not None:
            agg = np.average(vectors, weights=weights, axis=0)
        else:
            agg = np.mean(vectors, axis=0)
        return self.normalize(agg)
        
    def cluster_vectors(self, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(vectors)
        
    def reduce_dimensions(self, vectors: np.ndarray, n_components: int = 2) -> np.ndarray:
        pca = PCA(n_components=n_components)
        return pca.fit_transform(vectors) 