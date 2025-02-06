"""
Embedding Generator Module
------------------------

Core module for generating text embeddings using transformer models.
Provides efficient and cached embedding generation with batch processing.

Features:
- Multiple model support
- Device optimization (CPU/GPU/MPS)
- Caching system
- Batch processing
- Progress tracking
- Memory optimization

Key Components:
1. Model Management: Loading and configuration
2. Embedding Generation: Single and batch processing
3. Cache System: LRU caching for efficiency
4. Device Handling: Multi-device support
5. Metadata Processing: Document enrichment

Technical Details:
- Transformer architecture
- Normalized embeddings
- Automatic device selection
- Memory-efficient batching
- Progress monitoring

Dependencies:
- torch>=2.0.0
- transformers>=4.30.0
- numpy>=1.24.0
- tqdm>=4.65.0

Example Usage:
    generator = EmbeddingGenerator(model_name='minilm')
    embedding = generator.generate_embeddings("Sample text")
    batch_embeddings = generator.generate_embeddings(["Text 1", "Text 2"])

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
""" 

from typing import List, Dict, Optional, Union
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from functools import lru_cache
import logging
from ..config.settings import EMBEDDING_CONFIG

class EmbeddingGenerator:
    """Generates embeddings for text using transformer models."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None,
        max_length: int = 384,
        batch_size: int = 32
    ):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the transformer model to use
            device: Device to run model on ('cuda', 'mps', or 'cpu')
            max_length: Maximum sequence length
            batch_size: Batch size for processing
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Set device (with Apple Silicon support)
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logging.info(f"Loaded embedding model {model_name} on {device}")
    
    @lru_cache(maxsize=10000)
    def _get_embedding_cached(self, text: str) -> np.ndarray:
        """Get embedding for a single text with caching.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        # Convert to numpy and normalize
        embedding = embeddings[0].cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for one or more texts.
        
        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar
            
        Returns:
            Single embedding vector or list of vectors
        """
        # Handle single text
        if isinstance(texts, str):
            return self._get_embedding_cached(texts)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Get embeddings for batch
            batch_embeddings = [
                self._get_embedding_cached(text)
                for text in batch_texts
            ]
            
            all_embeddings.extend(batch_embeddings)
            
            if show_progress:
                logging.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
        
        return all_embeddings
    
    def generate_embeddings_with_metadata(
        self,
        documents: List[Dict],
        text_key: str = 'content'
    ) -> List[Dict]:
        """Generate embeddings for documents with metadata.
        
        Args:
            documents: List of document dictionaries
            text_key: Key for text content in documents
            
        Returns:
            Documents with added embeddings
        """
        # Extract texts
        texts = [doc[text_key] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, show_progress=True)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding.tolist()
            
        return documents

# Create default instance
embedding_generator = EmbeddingGenerator(**EMBEDDING_CONFIG) 

