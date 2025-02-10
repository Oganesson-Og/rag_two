"""
Audio Models Module
--------------
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
import numpy as np
from numpy.typing import NDArray

class AudioConfig(BaseModel):
    """Audio configuration model."""
    sample_rate: int = Field(default=16000, gt=0)
    channels: int = Field(default=1, gt=0)
    duration: Optional[float] = None
    format: str = Field(default="wav")
    device: Optional[str] = None
    model_name: str = Field(default="base")

    class Config:
        arbitrary_types_allowed = True

class AudioSegment(BaseModel):
    """Audio segment model."""
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    speaker: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

class AudioResult(BaseModel):
    """Audio processing result model."""
    text: str = Field(..., min_length=1)
    segments: List[AudioSegment] = Field(default_factory=list)
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True

class AudioFeatures(BaseModel):
    """Audio features model."""
    mfcc: Optional[NDArray[np.float32]] = None
    spectral: Optional[NDArray[np.float32]] = None
    temporal: Optional[NDArray[np.float32]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True 