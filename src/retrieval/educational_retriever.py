from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path
import logging
import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from datetime import datetime

class RetrievalStrategy(Enum):
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CURRICULUM_AWARE = "curriculum_aware"
    PREREQUISITE_BASED = "prerequisite_based"
    DIFFICULTY_ADAPTIVE = "difficulty_adaptive"

@dataclass
class RetrievalResult:
    """Represents a retrieval result with educational context."""
    content_id: str
    content: str
    similarity_score: float
    educational_relevance: float
    difficulty_level: float
    prerequisites_met: bool
    learning_objectives: List[str]
    metadata: Dict[str, Any]

class EducationalRetriever:
    """Enhanced retrieval system optimized for educational content."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        index_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.logger = logging.getLogger(__name__)
        self.device = device
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize embedding model
        self.model = SentenceTransformer(model_name).to(device)
        
        # Initialize FAISS index
        self.index_path = index_path
        self.index = self._initialize_index()
        
        # Initialize content store
        self.content_store: Dict[str, Dict] = {}
        
        # Initialize retrieval metrics
        self.retrieval_metrics = {
            "total_queries": 0,
            "avg_response_time": 0,
            "cache_hits": 0
        }
        
        # Initialize cache
        self.query_cache: Dict[str, List[RetrievalResult]] = {}

    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load retrieval configuration."""
        default_config = {
            "max_results": 10,
            "similarity_threshold": 0.7,
            "cache_size": 1000,
            "difficulty_weight": 0.3,
            "prerequisite_weight": 0.2,
            "relevance_weight": 0.5,
            "cache_ttl": 3600  # 1 hour
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)
                
        return default_config

    def _initialize_index(self) -> faiss.Index:
        """Initialize FAISS index."""
        if self.index_path and self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        else:
            return faiss.IndexFlatIP(768)  # Default dimension for MPNet

    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        user_context: Optional[Dict] = None,
        filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """Retrieve educational content using specified strategy."""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, strategy, filters)
            if cache_key in self.query_cache:
                self.retrieval_metrics["cache_hits"] += 1
                return self.query_cache[cache_key]
            
            # Generate query embedding
            query_embedding = self.model.encode(
                query,
                convert_to_tensor=True,
                device=self.device
            ).cpu().numpy()
            
            # Apply retrieval strategy
            if strategy == RetrievalStrategy.SEMANTIC:
                results = await self._semantic_retrieval(query_embedding, filters)
            elif strategy == RetrievalStrategy.HYBRID:
                results = await self._hybrid_retrieval(query, query_embedding, filters)
            elif strategy == RetrievalStrategy.CURRICULUM_AWARE:
                results = await self._curriculum_aware_retrieval(
                    query,
                    query_embedding,
                    user_context,
                    filters
                )
            elif strategy == RetrievalStrategy.PREREQUISITE_BASED:
                results = await self._prerequisite_based_retrieval(
                    query,
                    query_embedding,
                    user_context,
                    filters
                )
            else:  # DIFFICULTY_ADAPTIVE
                results = await self._difficulty_adaptive_retrieval(
                    query,
                    query_embedding,
                    user_context,
                    filters
                )
                
            # Update cache
            self.query_cache[cache_key] = results
            if len(self.query_cache) > self.config["cache_size"]:
                self._prune_cache()
                
            # Update metrics
            self.retrieval_metrics["total_queries"] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Retrieval error: {str(e)}")
            raise

    async def _semantic_retrieval(
        self,
        query_embedding: np.ndarray,
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """Perform semantic similarity-based retrieval."""
        # Search FAISS index
        D, I = self.index.search(
            query_embedding.reshape(1, -1),
            self.config["max_results"]
        )
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if score < self.config["similarity_threshold"]:
                continue
                
            content = self.content_store[str(idx)]
            if self._apply_filters(content, filters):
                results.append(
                    RetrievalResult(
                        content_id=str(idx),
                        content=content["text"],
                        similarity_score=float(score),
                        educational_relevance=self._calculate_educational_relevance(
                            content,
                            query_embedding
                        ),
                        difficulty_level=content.get("difficulty_level", 0.5),
                        prerequisites_met=True,  # Default for semantic search
                        learning_objectives=content.get("learning_objectives", []),
                        metadata=content.get("metadata", {})
                    )
                )
                
        return results

    async def _hybrid_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """Combine semantic and keyword-based retrieval."""
        # Get semantic results
        semantic_results = await self._semantic_retrieval(query_embedding, filters)
        
        # Get keyword results
        keyword_results = self._keyword_based_retrieval(query, filters)
        
        # Combine and deduplicate results
        combined_results = {}
        for result in semantic_results + keyword_results:
            if result.content_id not in combined_results:
                combined_results[result.content_id] = result
            else:
                # Take the higher scoring result
                existing = combined_results[result.content_id]
                if result.similarity_score > existing.similarity_score:
                    combined_results[result.content_id] = result
                    
        return list(combined_results.values())

    async def _curriculum_aware_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        user_context: Optional[Dict],
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """Retrieve content considering curriculum context."""
        base_results = await self._semantic_retrieval(query_embedding, filters)
        
        if not user_context:
            return base_results
            
        # Adjust results based on curriculum position
        curriculum_position = user_context.get("curriculum_position", {})
        current_topic = curriculum_position.get("current_topic")
        completed_topics = curriculum_position.get("completed_topics", [])
        
        scored_results = []
        for result in base_results:
            curriculum_score = self._calculate_curriculum_relevance(
                result,
                current_topic,
                completed_topics
            )
            result.educational_relevance *= curriculum_score
            scored_results.append(result)
            
        return sorted(
            scored_results,
            key=lambda x: x.educational_relevance,
            reverse=True
        )

    async def _prerequisite_based_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        user_context: Optional[Dict],
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """Retrieve content considering prerequisites."""
        base_results = await self._semantic_retrieval(query_embedding, filters)
        
        if not user_context:
            return base_results
            
        user_knowledge = user_context.get("completed_topics", [])
        
        filtered_results = []
        for result in base_results:
            prerequisites_met = self._check_prerequisites(
                result,
                user_knowledge
            )
            result.prerequisites_met = prerequisites_met
            if prerequisites_met:
                filtered_results.append(result)
                
        return filtered_results

    async def _difficulty_adaptive_retrieval(
        self,
        query: str,
        query_embedding: np.ndarray,
        user_context: Optional[Dict],
        filters: Optional[Dict]
    ) -> List[RetrievalResult]:
        """Retrieve content adapted to user's difficulty level."""
        base_results = await self._semantic_retrieval(query_embedding, filters)
        
        if not user_context:
            return base_results
            
        user_level = user_context.get("difficulty_level", 0.5)
        
        # Score results based on difficulty match
        scored_results = []
        for result in base_results:
            difficulty_score = 1 - abs(
                result.difficulty_level - user_level
            )
            result.educational_relevance *= (
                difficulty_score * self.config["difficulty_weight"] +
                result.similarity_score * (1 - self.config["difficulty_weight"])
            )
            scored_results.append(result)
            
        return sorted(
            scored_results,
            key=lambda x: x.educational_relevance,
            reverse=True
        )

    def _calculate_educational_relevance(
        self,
        content: Dict,
        query_embedding: np.ndarray
    ) -> float:
        """Calculate educational relevance score."""
        # Combine multiple factors:
        # 1. Content quality
        quality_score = content.get("quality_score", 0.5)
        
        # 2. Learning objective alignment
        objective_score = self._calculate_objective_alignment(
            content,
            query_embedding
        )
        
        # 3. Usage statistics
        usage_score = content.get("usage_score", 0.5)
        
        # Weighted combination
        return (
            0.4 * quality_score +
            0.4 * objective_score +
            0.2 * usage_score
        )

    def _calculate_objective_alignment(
        self,
        content: Dict,
        query_embedding: np.ndarray
    ) -> float:
        """Calculate alignment with learning objectives."""
        objectives = content.get("learning_objectives", [])
        if not objectives:
            return 0.5
            
        objective_embeddings = self.model.encode(
            objectives,
            convert_to_tensor=True,
            device=self.device
        ).cpu().numpy()
        
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            objective_embeddings
        )
        
        return float(np.max(similarities))

    def _calculate_curriculum_relevance(
        self,
        result: RetrievalResult,
        current_topic: Optional[str],
        completed_topics: List[str]
    ) -> float:
        """Calculate curriculum-based relevance."""
        if not current_topic:
            return 1.0
            
        # Check if content matches current topic
        content_topics = result.metadata.get("topics", [])
        if current_topic in content_topics:
            return 1.0
            
        # Check relation to completed topics
        topic_overlap = len(
            set(content_topics) & set(completed_topics)
        ) / len(content_topics) if content_topics else 0
        
        return 0.5 + 0.5 * topic_overlap

    def _check_prerequisites(
        self,
        result: RetrievalResult,
        user_knowledge: List[str]
    ) -> bool:
        """Check if prerequisites are met."""
        prerequisites = result.metadata.get("prerequisites", [])
        if not prerequisites:
            return True
            
        return all(prereq in user_knowledge for prereq in prerequisites)

    def _apply_filters(
        self,
        content: Dict,
        filters: Optional[Dict]
    ) -> bool:
        """Apply content filters."""
        if not filters:
            return True
            
        for key, value in filters.items():
            if key not in content:
                return False
            if isinstance(value, list):
                if not any(v in content[key] for v in value):
                    return False
            elif content[key] != value:
                return False
                
        return True

    def _generate_cache_key(
        self,
        query: str,
        strategy: RetrievalStrategy,
        filters: Optional[Dict]
    ) -> str:
        """Generate cache key for query."""
        key_parts = [query, strategy.value]
        if filters:
            key_parts.append(json.dumps(filters, sort_keys=True))
        return "_".join(key_parts)

    def _prune_cache(self) -> None:
        """Remove oldest entries from cache."""
        sorted_keys = sorted(
            self.query_cache.keys(),
            key=lambda k: self.query_cache[k][0].metadata.get(
                "timestamp",
                datetime.min
            )
        )
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(sorted_keys) // 10)
        for key in sorted_keys[:num_to_remove]:
            del self.query_cache[key]

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            "total_queries": self.retrieval_metrics["total_queries"],
            "cache_hit_rate": (
                self.retrieval_metrics["cache_hits"] /
                max(1, self.retrieval_metrics["total_queries"])
            ),
            "avg_response_time": self.retrieval_metrics["avg_response_time"],
            "index_size": self.index.ntotal,
            "cache_size": len(self.query_cache)
        }

    def save_index(self, path: Optional[Path] = None) -> None:
        """Save FAISS index to file."""
        save_path = path or self.index_path
        if save_path:
            faiss.write_index(self.index, str(save_path))
            self.logger.info(f"Saved index to {save_path}") 