"""
Local RAG Pipeline Test Module
----------------------------

Test implementation of the RAG pipeline using Ollama local models.

Author: Keith Satuku
Version: 1.0.0
Created: 2025
License: MIT
"""

import os
from pathlib import Path
import json
from typing import List, Dict
import requests
import numpy as np
from ..nlp import Tokenizer, SearchEngine
from ..document_processing.text_enhancer import TextEnhancer
from ..document_processing.chunking.syllabus_chunker import SyllabusChunker
from ..config.settings import (
    DATA_DIR,
    CACHE_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    OLLAMA_BASE_URL
)

class LocalRAGTest:
    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        chunk_size: int = 500
    ):
        self.embed_model = embed_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.tokenizer = Tokenizer()
        self.search_engine = SearchEngine()
        self.enhancer = TextEnhancer()
        self.chunker = SyllabusChunker(subject='form4')
        self.documents = []
        self.embeddings = []

    def load_documents(self, folder_path: str = "knowledge_bank/Form 4"):
        """Load and process documents from the specified folder."""
        folder = Path(folder_path)
        for file_path in folder.glob("**/*.*"):
            if file_path.suffix in ['.txt', '.md', '.pdf']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Process the document
                enhanced_content = self.enhancer.process_text(content)
                chunks = self.chunker.chunk_document(
                    enhanced_content,
                    chunk_size=self.chunk_size
                )
                
                # Store chunks with metadata
                for chunk in chunks:
                    self.documents.append({
                        'content': chunk,
                        'source': str(file_path),
                        'subject': 'Form 4'
                    })
        
        print(f"Loaded {len(self.documents)} chunks from {folder_path}")

    def generate_embeddings(self):
        """Generate embeddings using Ollama's nomic-embed-text model."""
        for doc in self.documents:
            response = requests.post(
                'http://localhost:11434/api/embeddings',
                json={
                    'model': self.embed_model,
                    'prompt': doc['content']
                }
            )
            
            if response.status_code == 200:
                embedding = response.json()['embedding']
                self.embeddings.append(embedding)
            else:
                print(f"Error generating embedding: {response.text}")

        print(f"Generated {len(self.embeddings)} embeddings")

    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Perform similarity search using generated embeddings."""
        # Get query embedding
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={
                'model': self.embed_model,
                'prompt': query
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error getting query embedding: {response.text}")
            
        query_embedding = response.json()['embedding']
        
        # Calculate similarities
        similarities = []
        for idx, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((similarity, idx))
        
        # Get top_k results
        top_results = sorted(similarities, reverse=True)[:top_k]
        
        return [
            {
                'content': self.documents[idx]['content'],
                'source': self.documents[idx]['source'],
                'similarity': score
            }
            for score, idx in top_results
        ]

    def generate_response(self, query: str, context: List[Dict]) -> str:
        """Generate response using Ollama's phi4 model."""
        # Prepare prompt with context
        context_text = "\n\n".join([doc['content'] for doc in context])
        prompt = f"""Based on the following context, please answer the question.

Context:
{context_text}

Question: {query}

Answer:"""

        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': self.llm_model,
                'prompt': prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Error generating response: {response.text}")

    def query(self, query: str, top_k: int = 3) -> Dict:
        """Process a query through the entire RAG pipeline."""
        # Get relevant documents
        relevant_docs = self.similarity_search(query, top_k=top_k)
        
        # Generate response
        response = self.generate_response(query, relevant_docs)
        
        return {
            'query': query,
            'response': response,
            'sources': relevant_docs
        }

# Example usage
if __name__ == "__main__":
    # Initialize the RAG pipeline
    rag = LocalRAGTest()
    
    # Load and process documents
    rag.load_documents()
    
    # Generate embeddings
    rag.generate_embeddings()
    
    # Test query
    test_query = "What are the key concepts in Form 4 Mathematics?"
    result = rag.query(test_query)
    
    print("\nQuery:", result['query'])
    print("\nResponse:", result['response'])
    print("\nSources:")
    for idx, source in enumerate(result['sources'], 1):
        print(f"\n{idx}. From {source['source']} (similarity: {source['similarity']:.4f}):")
        print(source['content'][:200] + "...") 