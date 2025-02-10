"""
Query Model Module
--------------
"""

from typing import Dict, List, Union, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np

class QueryMetadata(TypedDict):
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    filters: Dict[str, Union[str, int, float, List[str]]]

class QueryResult(TypedDict):
    text: str
    score: float
    metadata: Dict[str, Union[str, int, float]]
    vector: Optional[Union[List[float], np.ndarray]]

class Query(BaseModel):
    text: str
    metadata: QueryMetadata = Field(default_factory=lambda: {
        'timestamp': datetime.now(),
        'filters': {}
    })
    top_k: int = 5
    
    class Config:
        arbitrary_types_allowed = True 