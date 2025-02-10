"""
Vector Metrics Module
-----------------
"""

from typing import List, Dict, Union, Optional
import numpy as np
from datetime import datetime

class VectorMetrics:
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        
    def record_distance(
        self,
        vector1: Union[List[float], np.ndarray],
        vector2: Union[List[float], np.ndarray],
        metric_name: str = 'cosine'
    ) -> float:
        """Record distance between vectors."""
        distance = self._calculate_distance(vector1, vector2, metric_name)
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(distance)
        return distance
        
    def _calculate_distance(
        self,
        vector1: Union[List[float], np.ndarray],
        vector2: Union[List[float], np.ndarray],
        metric: str
    ) -> float:
        """Calculate distance between vectors."""
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        if metric == 'cosine':
            return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        elif metric == 'euclidean':
            return np.linalg.norm(v1 - v2)
        else:
            raise ValueError(f"Unsupported metric: {metric}") 