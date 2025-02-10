from typing import List, Dict
import re
from collections import Counter

class KeywordSearch:
    def __init__(self):
        self.documents = []
        self.index = {}
        
    def index_documents(self, documents: List[Dict]):
        self.documents = documents
        self.index = {}
        
        for i, doc in enumerate(documents):
            terms = self._tokenize(doc['text'])
            for term in terms:
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(i)
                
    def search(self, query: str, k: int = 3) -> List[Dict]:
        query_terms = self._tokenize(query)
        scores = Counter()
        
        for term in query_terms:
            if term in self.index:
                for doc_idx in self.index[term]:
                    scores[doc_idx] += 1
                    
        results = []
        for doc_idx, score in scores.most_common(k):
            doc = self.documents[doc_idx].copy()
            doc['score'] = score / len(query_terms)
            doc['exact_match'] = self._check_exact_match(query, doc['text'])
            results.append(doc)
            
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization and normalization."""
        text = text.lower()
        words = re.findall(r'\w+', text)
        return words
        
    def _check_exact_match(self, query: str, text: str) -> bool:
        """Check if query appears exactly in text."""
        return query.lower() in text.lower() 