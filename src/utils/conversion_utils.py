"""
Conversion Utilities
-----------------
"""

from typing import Dict, Union, Optional
from datetime import datetime
import json

class ConversionUtils:
    def bytes_to_mb(self, bytes_val: int) -> float:
        return bytes_val / (1024 * 1024)
        
    def mb_to_bytes(self, mb_val: float) -> int:
        return int(mb_val * 1024 * 1024)
        
    def seconds_to_minutes(self, seconds: int) -> float:
        return seconds / 60
        
    def minutes_to_seconds(self, minutes: float) -> int:
        return int(minutes * 60)
        
    def str_to_date(self, date_str: str, 
                    format: str = "%Y-%m-%d") -> datetime:
        """Convert string to datetime."""
        return datetime.strptime(date_str, format)
        
    def date_to_str(self, date: datetime, 
                    format: str = "%Y-%m-%d") -> str:
        """Convert datetime to string."""
        return date.strftime(format)
        
    def datetime_to_timestamp(self, dt: datetime) -> float:
        """Convert datetime to Unix timestamp."""
        return dt.timestamp()
        
    def timestamp_to_datetime(self, timestamp: float) -> datetime:
        """Convert Unix timestamp to datetime."""
        return datetime.fromtimestamp(timestamp)
        
    def to_json(self, data: Union[Dict, list], 
                pretty: bool = False) -> str:
        """Convert data to JSON string."""
        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data) 