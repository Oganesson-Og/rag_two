"""
Vector Store Module
---------------
"""

from typing import Dict, List, Union, Optional, TypedDict
import numpy as np
from datetime import datetime
from pathlib import Path

class VectorRecord(TypedDict):
    id: str
    vector: Union[List[float], np.ndarray]
    metadata: Dict[str, Union[str, int, float]]
    timestamp: datetime

class VectorStore:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.store: Dict[str, VectorRecord] = {}
        
    def add_vector(
        self,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict] = None
    ) -> str:
        """Add vector to store."""
        vector_id = self._generate_id()
        self.store[vector_id] = {
            'id': vector_id,
            'vector': vector,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        return vector_id
        
    def get_vector(
        self,
        vector_id: str
    ) -> Optional[VectorRecord]:
        """Get vector from store."""
        return self.store.get(vector_id)
        
    def _generate_id(self) -> str:
        """Generate unique vector ID."""
        import uuid
        return str(uuid.uuid4()) 