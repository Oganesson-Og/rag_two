from typing import Any, Optional, Union
import redis
from pathlib import Path
import pickle
import hashlib
from datetime import datetime, timedelta

class MultiModalCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        file_cache_dir: Path = Path("./cache"),
        ttl: int = 3600  # 1 hour default TTL
    ):
        self.redis_client = redis.from_url(redis_url)
        self.file_cache_dir = file_cache_dir
        self.ttl = ttl
        self.file_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get(self, key: str, modality: str) -> Optional[Any]:
        cache_key = self._generate_key(key, modality)
        
        # Try Redis first for faster access
        redis_result = self.redis_client.get(cache_key)
        if redis_result:
            return pickle.loads(redis_result)
            
        # Fall back to file cache
        file_path = self.file_cache_dir / f"{cache_key}.pkl"
        if file_path.exists():
            with open(file_path, 'rb') as f:
                return pickle.load(f)
                
        return None
        
    def set(
        self,
        key: str,
        value: Any,
        modality: str,
        ttl: Optional[int] = None
    ) -> None:
        cache_key = self._generate_key(key, modality)
        pickled_value = pickle.dumps(value)
        
        # Store in Redis
        self.redis_client.set(
            cache_key,
            pickled_value,
            ex=ttl or self.ttl
        )
        
        # Store in file cache
        file_path = self.file_cache_dir / f"{cache_key}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(value, f)
            
    def _generate_key(self, key: str, modality: str) -> str:
        combined = f"{key}:{modality}"
        return hashlib.md5(combined.encode()).hexdigest()
        
    def clear(self, modality: Optional[str] = None) -> None:
        if modality:
            pattern = f"*:{modality}*"
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                
            # Clear file cache for modality
            for file_path in self.file_cache_dir.glob(f"*{modality}*.pkl"):
                file_path.unlink()
        else:
            self.redis_client.flushdb()
            for file_path in self.file_cache_dir.glob("*.pkl"):
                file_path.unlink()