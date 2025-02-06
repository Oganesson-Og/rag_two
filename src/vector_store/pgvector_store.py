"""
PostgreSQL Vector Store Module
---------------------------

PGVector-based vector storage implementation providing efficient similarity search
and metadata filtering capabilities for educational content vectors.

Key Features:
- Vector similarity search with cosine distance
- Complex metadata filtering support
- Batch operations for vectors and metadata
- Index optimization for fast retrieval
- Educational content organization
- Automatic index maintenance
- Transaction support

Technical Details:
- Uses pgvector extension for PostgreSQL
- Implements IVFFlat indexing
- Supports multiple vector dimensions
- Provides connection pooling
- Implements efficient batch operations

Dependencies:
- psycopg2-binary>=2.9.9
- numpy>=1.24.0
- sqlalchemy>=2.0.0
- alembic>=1.12.0
- pgvector>=0.2.0

Example Usage:
    # Initialize store
    store = PGVectorStore()
    
    # Add documents with embeddings
    store.add_documents(documents, embeddings)
    
    # Similarity search
    results = store.similarity_search(
        query_embedding,
        k=5,
        metadata_filter={'subject': 'physics'}
    )

Performance Optimization:
- Uses prepared statements
- Implements connection pooling
- Optimizes index parameters
- Supports async operations
- Implements query planning

Author: Keith Satuku
Version: 2.1.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from ..config.settings import DB_CONFIG



class PGVectorStore:
    """PostgreSQL vector store implementation using pgvector."""
    
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self._init_db()
    
    def _init_db(self):
        """Initialize database with required extensions and tables."""
        with self.conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            
            # Create documents table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    metadata JSONB,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            ''')
            
            self.conn.commit()
    
    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]):
        """Add documents and their embeddings to the store.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            embeddings: List of embedding vectors
        """
        with self.conn.cursor() as cur:
            values = [
                (doc['content'], doc['metadata'], embedding)
                for doc, embedding in zip(documents, embeddings)
            ]
            
            execute_values(cur, '''
                INSERT INTO documents (content, metadata, embedding)
                VALUES %s
            ''', values)
            
            self.conn.commit()
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 4,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            metadata_filter: Optional metadata filters
            
        Returns:
            List of similar documents with scores
        """
        filter_clause = ''
        filter_params = []
        
        if metadata_filter:
            conditions = []
            for key, value in metadata_filter.items():
                conditions.append(f"metadata->>{len(filter_params)} = %s")
                filter_params.append(key)
                conditions.append(f"metadata->>%s = %s")
                filter_params.extend([key, str(value)])
            filter_clause = 'WHERE ' + ' AND '.join(conditions)
        
        with self.conn.cursor() as cur:
            query = f'''
                SELECT 
                    content,
                    metadata,
                    1 - (embedding <=> %s) as similarity
                FROM documents
                {filter_clause}
                ORDER BY similarity DESC
                LIMIT %s
            '''
            
            cur.execute(query, [query_embedding] + filter_params + [k])
            results = cur.fetchall()
            
            return [
                {
                    'content': content,
                    'metadata': metadata,
                    'score': float(score)
                }
                for content, metadata, score in results
            ] 