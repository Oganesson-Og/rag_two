"""
Vector Cache Module
----------------
"""

from typing import Dict, List, Union, Optional, TypedDict
import numpy as np
from datetime import datetime

class CacheEntry(TypedDict):
    vector: Union[List[float], np.ndarray]
    metadata: Dict[str, Union[str, int, float]]
    timestamp: datetime
    ttl: Optional[int]

class VectorCache:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.cache: Dict[str, CacheEntry] = {}
        
    def get(
        self,
        key: str
    ) -> Optional[Union[List[float], np.ndarray]]:
        """Get vector from cache."""
        entry = self.cache.get(key)
        if entry and not self._is_expired(entry):
            return entry['vector']
        return None
        
    def set(
        self,
        key: str,
        vector: Union[List[float], np.ndarray],
        metadata: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> None:
        """Set vector in cache."""
        self.cache[key] = {
            'vector': vector,
            'metadata': metadata or {},
            'timestamp': datetime.now(),
            'ttl': ttl
        }
        
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if not entry['ttl']:
            return False
        age = (datetime.now() - entry['timestamp']).total_seconds()
        return age > entry['ttl'] 