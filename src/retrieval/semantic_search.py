from typing import List, Dict, Optional, Callable
import numpy as np

class SemanticSearch:
    def __init__(self, embedding_generator):
        self.embedding_generator = embedding_generator
        self.documents = []
        self.embeddings = None
        
    def index_documents(self, documents: List[Dict]):
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        self.embeddings = self.embedding_generator.generate_embeddings(texts)
        
    def search(self, query: str, k: int = 3, threshold: float = 0.0,
              filters: Optional[Dict] = None, context: Optional[str] = None,
              rerank_fn: Optional[Callable] = None) -> List[Dict]:
        if not query:
            raise ValueError("Query cannot be empty")
        if k < 1:
            raise ValueError("k must be positive")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
            
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        scores = np.dot(self.embeddings, query_embedding)
        
        results = []
        for i, score in enumerate(scores):
            if score >= threshold:
                doc = self.documents[i].copy()
                doc['score'] = float(score)
                if not filters or all(doc['metadata'].get(k) == v for k, v in filters.items()):
                    results.append(doc)
                    
        results = sorted(results, key=lambda x: x['score'], reverse=True)[:k]
        
        if rerank_fn:
            results = rerank_fn(results, query)
            
        return results
        
    def batch_search(self, queries: List[str], **kwargs) -> List[List[Dict]]:
        return [self.search(query, **kwargs) for query in queries]
        
    def add_documents(self, documents: List[Dict]):
        if not self.embeddings:
            self.index_documents(documents)
        else:
            self.documents.extend(documents)
            new_embeddings = self.embedding_generator.generate_embeddings(
                [doc['text'] for doc in documents]
            )
            self.embeddings = np.vstack([self.embeddings, new_embeddings]) 