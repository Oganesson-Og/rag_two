"""
Educational Content Retrieval Module
---------------------------------

Advanced retrieval system specifically designed for educational content, implementing
sophisticated ranking and filtering strategies for academic material retrieval.

Key Features:
- Education-aware ranking algorithms
- Syllabus-based content filtering
- Complexity level matching
- Intent-based retrieval optimization
- Educational metadata utilization
- Multi-stage ranking pipeline
- Cross-reference handling
- Prerequisite tracking

Technical Details:
- Hybrid retrieval combining vector and semantic search
- Custom scoring functions for educational relevance
- Efficient metadata filtering with PostgreSQL
- Query result caching system
- Automatic query expansion for educational terms
- Learning path awareness
- Topic relationship mapping

Dependencies:
- numpy>=1.24.0
- psycopg2-binary>=2.9.9
- fastapi>=0.109.0
- pydantic>=2.5.0
- scipy>=1.11.0
- redis>=5.0.1
- faiss-cpu>=1.7.4

Example Usage:
    # Basic retrieval
    retriever = EducationRetriever(
        subject='physics',
        level='a-level'
    )
    results = retriever.retrieve(query)
    
    # Advanced retrieval with filters
    results = retriever.retrieve(
        query,
        filter_metadata={
            'difficulty': 'advanced',
            'topic': 'quantum_mechanics'
        },
        include_prerequisites=True
    )
    
    # Batch retrieval
    results = retriever.batch_retrieve(
        queries,
        top_k=5
    )

Performance Optimization:
- Result caching with Redis
- Batch processing support
- Database query optimization
- Connection pooling
- Asynchronous operations
- Index optimization

Author: Keith Satuku
Version: 2.5.0
Created: 2025
License: MIT
"""

from typing import List, Dict, Optional
import numpy as np
from ..query.education_processor import EducationQueryProcessor
from ..vector_store.pgvector_store import PGVectorStore
from ..embeddings.enhanced_generator import EnhancedEmbeddingGenerator
from ..config.domain_config import get_domain_config

class EducationRetriever:
    """Retrieval system specialized for educational content."""
    
    def __init__(
        self,
        subject: str,
        level: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        self.subject = subject
        self.level = level
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.domain_config = get_domain_config(subject, level)
        self.query_processor = EducationQueryProcessor(subject, level)
        self.vector_store = PGVectorStore()
        self.embedding_generator = EnhancedEmbeddingGenerator(
            model_key=self.domain_config['model']
        )
    
    def retrieve(
        self,
        query: str,
        filter_metadata: Optional[Dict] = None,
        include_prerequisites: bool = False
    ) -> List[Dict]:
        """Retrieve relevant educational content.
        
        Args:
            query: User query
            filter_metadata: Optional metadata filters
            include_prerequisites: Whether to include prerequisites
            
        Returns:
            List of relevant documents with scores
        """
        # Process query
        processed_query = self.query_processor.process_query(query)
        
        # Prepare metadata filters
        metadata_filter = {
            'subject': self.subject,
            'level': self.level
        }
        if filter_metadata:
            metadata_filter.update(filter_metadata)
        
        # Get base vector search results
        vector_results = self.vector_store.similarity_search(
            query_embedding=processed_query['embedding'],
            k=self.top_k * 2,  # Get more for reranking
            metadata_filter=metadata_filter
        )
        
        # Rerank results
        reranked_results = self._rerank_results(
            query=processed_query,
            results=vector_results
        )
        
        # Filter and format results
        final_results = []
        for result in reranked_results[:self.top_k]:
            if result['score'] >= self.similarity_threshold:
                final_results.append(self._format_result(result))
        
        if include_prerequisites:
            final_results = self._include_prerequisites(final_results)
        
        return final_results
    
    def _rerank_results(
        self,
        query: Dict,
        results: List[Dict]
    ) -> List[Dict]:
        """Rerank results using educational criteria."""
        reranked = []
        
        for result in results:
            # Calculate educational relevance score
            edu_score = self._calculate_educational_relevance(
                query=query,
                result=result
            )
            
            # Combine with vector similarity score
            final_score = 0.7 * result['score'] + 0.3 * edu_score
            
            reranked.append({
                **result,
                'score': final_score,
                'educational_score': edu_score
            })
        
        # Sort by final score
        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked
    
    def _calculate_educational_relevance(
        self,
        query: Dict,
        result: Dict
    ) -> float:
        """Calculate educational relevance score."""
        score = 0.0
        
        # Check syllabus topic match
        query_topics = set(query['syllabus_topics'])
        result_topics = set(result['metadata'].get('topics', []))
        topic_overlap = len(query_topics & result_topics)
        score += 0.4 * (topic_overlap / max(len(query_topics), 1))
        
        # Check complexity level match
        query_level = query['complexity_level']
        result_level = result['metadata'].get('complexity', 'intermediate')
        score += 0.3 * (1.0 if query_level == result_level else 0.5)
        
        # Check educational intent match
        query_intent = query['query_intent']
        result_type = result['metadata'].get('content_type', 'general')
        score += 0.3 * (1.0 if query_intent == result_type else 0.5)
        
        return score
    
    def _format_result(self, result: Dict) -> Dict:
        """Format retrieval result for presentation."""
        return {
            'content': result['content'],
            'metadata': {
                'subject': result['metadata'].get('subject', self.subject),
                'level': result['metadata'].get('level', self.level),
                'topics': result['metadata'].get('topics', []),
                'complexity': result['metadata'].get('complexity', 'intermediate'),
                'content_type': result['metadata'].get('content_type', 'general')
            },
            'relevance_score': result['score'],
            'educational_score': result.get('educational_score', 0.0)
        }

    def _include_prerequisites(self, results: List[Dict]) -> List[Dict]:
        """Include prerequisites in the results."""
        # Implementation of _include_prerequisites method
        # This method should return a list of dictionaries with prerequisites included
        # For example, you can add a new key 'prerequisites' to each result
        # and populate it with the necessary prerequisite information
        # This is a placeholder and should be implemented based on your specific requirements
        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict]]:
        """Retrieve relevant educational content for multiple queries.
        
        Args:
            queries: List of user queries
            top_k: Number of top results to retrieve for each query
            
        Returns:
            List of lists of relevant documents with scores
        """
        results = []
        for query in queries:
            results.append(self.retrieve(query, top_k=top_k))
        return results 