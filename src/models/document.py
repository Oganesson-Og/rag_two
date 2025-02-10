"""
Document Models
------------
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
from numpy.typing import NDArray

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    source: str = Field(..., min_length=1)
    created_at: datetime = Field(default_factory=datetime.now)
    language: str = "en"
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    def format_created_at(self) -> str:
        return self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 3600

class Document(BaseModel):
    """Document model."""
    id: str = Field(..., min_length=1)
    text: str = Field(..., max_length=100000)
    metadata: DocumentMetadata
    embedding: Optional[Union[List[float], NDArray[np.float32]]] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def text_length(self) -> int:
        return len(self.text)
        
    def get_truncated_text(self, max_length: int) -> str:
        return self.text[:max_length] + "..." if len(self.text) > max_length else self.text 