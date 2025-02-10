"""
Cache Models
---------
"""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class CacheMetadata(BaseModel):
    """Cache metadata model."""
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    size_bytes: int = Field(gt=0)
    last_accessed: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

class CacheEntry(BaseModel):
    """Cache entry model."""
    key: str = Field(..., min_length=1)
    value: Any
    metadata: CacheMetadata = Field(default_factory=CacheMetadata)

    class Config:
        arbitrary_types_allowed = True
    
    def is_expired(self) -> bool:
        if not self.metadata.expires_at:
            return False
        return datetime.now() > self.metadata.expires_at 