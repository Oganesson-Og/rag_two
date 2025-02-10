import numpy as np
import faiss
from typing import List, Dict, Optional, Callable
from pathlib import Path

class VectorStore:
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
        self.index = None
        self.metadata = []
        self.index_size = 0
        
    # ... rest of the implementation ... 