"""
API Models Module
-------------
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime

class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    level: str = Field(..., min_length=1)
    filters: Dict[str, Any] = Field(default_factory=dict)
    max_results: int = Field(default=5, gt=0)

    class Config:
        arbitrary_types_allowed = True

class QueryResponse(BaseModel):
    """Query response model."""
    results: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

class DocumentRequest(BaseModel):
    """Document request model."""
    document_id: str = Field(..., min_length=1)
    content: str = Field(..., max_length=100000)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

class DocumentResponse(BaseModel):
    """Document response model."""
    document_id: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)
    error_code: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class BatchQueryRequest(BaseModel):
    """Batch query request model."""
    queries: List[QueryRequest]
    batch_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "queries": [
                    {
                        "query": "What is quantum mechanics?",
                        "filters": {"subject": "physics"},
                        "top_k": 5
                    }
                ],
                "batch_id": "batch_001"
            }
        }

class AudioRequest(BaseModel):
    """Audio processing request model."""
    language: str = Field(default="en")
    task_type: str = Field(default="transcribe")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True 