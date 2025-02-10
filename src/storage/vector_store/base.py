from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import numpy as np
from numpy.typing import NDArray

# Type alias for vector
Vector = Union[List[float], NDArray[np.float32]]

class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add_vector(self, vector: Vector, metadata: Dict[str, Any] = None) -> str:
        """Add a vector to the store and return its ID."""
        pass

    @abstractmethod
    def get_vector(self, vector_id: str) -> Dict[str, Any]:
        """Retrieve a stored vector."""
        pass

    @abstractmethod
    def search(
        self, 
        query_vector: Vector, 
        k: int = 10, 
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors using cosine similarity (or another metric)."""
        pass

    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> int:
        """Delete vector entries by IDs."""
        pass

    @abstractmethod
    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]) -> bool:
        """Update metadata for a stored vector."""
        pass 