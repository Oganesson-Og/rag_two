from typing import Dict, List, Optional
from .ab_testing import ABTestingFramework, Variant, ExperimentType
from ..vector_store.education_vector_store import EducationVectorStore
from ..embeddings.embedding_manager import EmbeddingManager
import uuid
from datetime import datetime

class RAGExperimentManager:
    def __init__(
        self,
        vector_store: EducationVectorStore,
        embedding_manager: EmbeddingManager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.active_experiments: Dict[str, ABTestingFramework] = {}

    def create_retrieval_experiment(
        self,
        experiment_name: str,
        variants: List[Dict],
        traffic_split: Optional[List[float]] = None
    ) -> None:
        """Create a new retrieval strategy experiment."""
        experiment_variants = []
        
        for variant_config in variants:
            variant = Variant(
                id=str(uuid.uuid4()),
                name=variant_config["name"],
                config=variant_config["config"],
                implementation=variant_config["implementation"]
            )
            experiment_variants.append(variant)

        experiment = ABTestingFramework(
            experiment_name=experiment_name,
            experiment_type=ExperimentType.RETRIEVAL_STRATEGY,
            variants=experiment_variants,
            metrics=[
                "relevance_score",
                "response_time",
                "user_satisfaction",
                "context_precision"
            ],
            traffic_split=traffic_split
        )
        
        self.active_experiments[experiment_name] = experiment

    def process_query(
        self,
        experiment_name: str,
        user_id: str,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process query using assigned experimental variant."""
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
            
        experiment = self.active_experiments[experiment_name]
        variant = experiment.assign_variant(user_id)
        
        # Generate query embedding
        query_embedding = self.embedding_manager.get_text_embedding(query)
        
        # Execute variant implementation
        start_time = datetime.utcnow()
        results = variant.implementation(
            query_embedding=query_embedding,
            vector_store=self.vector_store,
            context=context,
            **variant.config
        )
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Record metrics
        metrics = {
            "response_time": response_time,
            "num_results": len(results),
            # Additional metrics would be calculated here
        }
        
        experiment.record_result(
            variant_id=variant.id,
            metrics=metrics,
            user_id=user_id,
            query_id=str(uuid.uuid4())
        )
        
        return {
            "results": results,
            "variant": variant.name,
            "metrics": metrics
        }

    def get_experiment_results(self, experiment_name: str) -> Dict:
        """Get results for specific experiment."""
        if experiment_name not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
            
        experiment = self.active_experiments[experiment_name]
        return {
            "stats": experiment.get_experiment_stats(),
            "winning_variant": experiment.get_winning_variant("relevance_score")
        } 