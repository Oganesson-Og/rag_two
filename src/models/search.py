"""
Search Models
----------
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

class SearchFilter(BaseModel):
    """Search filter model."""
    field: str = Field(..., min_length=1)
    value: Union[str, int, float, bool, List[Any]]
    operator: str = Field(default="eq")

    class Config:
        arbitrary_types_allowed = True

class SearchQuery(BaseModel):
    """Search query model."""
    text: str = Field(..., min_length=1)
    filters: List[SearchFilter] = Field(default_factory=list)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc")
    limit: int = Field(default=10, gt=0)
    offset: int = Field(default=0, ge=0)

    class Config:
        arbitrary_types_allowed = True

class SearchResult(BaseModel):
    """Search result model."""
    id: str = Field(..., min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    content: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True 