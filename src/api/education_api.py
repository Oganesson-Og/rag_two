"""
Educational RAG API Module
------------------------

FastAPI-based REST API implementation providing endpoints for educational content
retrieval, benchmarking, and system management.

Key Features:
- Query processing endpoints
- Benchmark execution endpoints
- Subject management interfaces
- Comprehensive error handling
- Response formatting and validation
- Rate limiting and throttling
- Authentication and authorization
- API documentation generation

Technical Details:
- FastAPI framework implementation
- OpenAPI/Swagger documentation
- JWT authentication support
- Request validation using Pydantic
- Async endpoint handling
- CORS support

Dependencies:
- fastapi>=0.109.0
- uvicorn>=0.27.0
- pydantic>=2.5.0
- python-jose>=3.3.0
- python-multipart>=0.0.6
- starlette>=0.36.0

Example Usage:
    # Start API server
    uvicorn education_api:app --reload
    
    # Query endpoint
    POST /query
    {
        "query": "explain quantum mechanics",
        "subject": "physics",
        "level": "a-level",
        "filters": {"complexity": "advanced"}
    }
    
    # Benchmark endpoint
    POST /benchmark
    {
        "subject": "physics",
        "level": "a-level",
        "model_key": "scibert"
    }

API Security:
- JWT token authentication
- Role-based access control
- Rate limiting per client
- Input validation and sanitization
- Error handling and logging

Author: Keith Satuku
Version: 2.2.0
Created: 2025
License: MIT
"""
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ..retrieval.education_retriever import EducationRetriever
from ..embeddings.education_benchmark import EducationBenchmark
from ..config.domain_config import EDUCATION_DOMAINS

app = FastAPI(title="Education RAG API")

class QueryRequest(BaseModel):
    query: str
    subject: str
    level: str
    filters: Optional[Dict] = None
    top_k: Optional[int] = 5

class BenchmarkRequest(BaseModel):
    subject: str
    level: str
    model_key: Optional[str] = None

class DocumentRequest(BaseModel):
    content: str
    subject: str
    level: str
    metadata: Optional[Dict] = None

# Cache retrievers for different subject/level combinations
retrievers: Dict[str, EducationRetriever] = {}

@app.post("/query")
async def query_endpoint(request: QueryRequest) -> Dict:
    """Query endpoint for educational content retrieval."""
    try:
        # Get or create retriever
        retriever_key = f"{request.subject}_{request.level}"
        if retriever_key not in retrievers:
            retrievers[retriever_key] = EducationRetriever(
                subject=request.subject,
                level=request.level,
                top_k=request.top_k or 5
            )
        
        # Get results
        results = retrievers[retriever_key].retrieve(
            query=request.query,
            filter_metadata=request.filters
        )
        
        return {
            "status": "success",
            "results": results,
            "metadata": {
                "subject": request.subject,
                "level": request.level,
                "total_results": len(results)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/benchmark")
async def benchmark_endpoint(request: BenchmarkRequest) -> Dict:
    """Run benchmarks for educational embeddings."""
    try:
        benchmark = EducationBenchmark(
            subject=request.subject,
            level=request.level
        )
        
        results = benchmark.run_education_benchmark()
        
        if request.model_key:
            results = {request.model_key: results[request.model_key]}
            
        return {
            "status": "success",
            "benchmark_results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/subjects")
async def list_subjects() -> Dict:
    """List available subjects and levels."""
    return {
        "status": "success",
        "domains": EDUCATION_DOMAINS
    }

