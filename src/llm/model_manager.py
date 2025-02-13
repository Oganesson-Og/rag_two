"""LLM Model Manager with fallback support."""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from openai import AsyncOpenAI
import google.generativeai as genai
from llama_cpp import Llama
from ..error.models import ModelError, RateLimitError

class ModelProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GOOGLE = "google"
    LLAMA = "llama"

@dataclass
class ModelConfig:
    """Configuration for LLM model."""
    provider: ModelProvider
    model_name: str
    api_key: Optional[str] = None
    organization: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    priority: int = 0  # Lower number = higher priority

class LLMManager:
    """Manages multiple LLM models with fallback support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.models: Dict[str, ModelConfig] = {
            "primary": ModelConfig(
                provider=ModelProvider.OPENAI,
                model_name="gpt-4",
                api_key=config.get("openai_api_key"),
                organization=config.get("openai_org"),
                priority=0
            ),
            "secondary": ModelConfig(
                provider=ModelProvider.GOOGLE,
                model_name="gemini-pro",
                api_key=config.get("google_api_key"),
                priority=1
            ),
            "tertiary": ModelConfig(
                provider=ModelProvider.LLAMA,
                model_name="llama-2-70b",
                priority=2
            )
        }
        
        # Initialize clients
        self._init_clients(config)
        
        # Track failures for circuit breaking
        self.failure_counts: Dict[str, int] = {
            model_id: 0 for model_id in self.models
        }
        self.max_failures = 3
        self.failure_reset_time = 300  # 5 minutes
        
        self.lock = asyncio.Lock()
    
    def _init_clients(self, config: Dict[str, Any]):
        """Initialize API clients for each provider."""
        self.clients = {}
        
        # OpenAI client
        if config.get("openai_api_key"):
            self.clients[ModelProvider.OPENAI] = AsyncOpenAI(
                api_key=config["openai_api_key"],
                organization=config.get("openai_org")
            )
        
        # Google client
        if config.get("google_api_key"):
            genai.configure(api_key=config["google_api_key"])
            self.clients[ModelProvider.GOOGLE] = genai
        
        # Llama client
        self.clients[ModelProvider.LLAMA] = Llama(
            model_path=config.get("llama_model_path", "models/llama-2-70b.gguf"),
            n_ctx=4096
        )
    
    async def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion with fallback support."""
        # Sort models by priority
        sorted_models = sorted(
            self.models.items(),
            key=lambda x: x[1].priority
        )
        
        last_error = None
        
        for model_id, model_config in sorted_models:
            if self.failure_counts[model_id] >= self.max_failures:
                continue
                
            try:
                response = await self._generate_with_model(
                    model_config,
                    prompt,
                    system_prompt,
                    **kwargs
                )
                
                # Reset failure count on success
                self.failure_counts[model_id] = 0
                
                return response
                
            except Exception as e:
                last_error = e
                
                async with self.lock:
                    self.failure_counts[model_id] += 1
                
                # Schedule failure count reset
                asyncio.create_task(
                    self._reset_failure_count(model_id)
                )
                
                continue
        
        # If all models failed
        raise ModelError(
            f"All models failed. Last error: {str(last_error)}"
        )
    
    async def _generate_with_model(
        self,
        model_config: ModelConfig,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate completion using specific model."""
        if model_config.provider == ModelProvider.OPENAI:
            response = await self.clients[ModelProvider.OPENAI].chat.completions.create(
                model=model_config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                top_p=model_config.top_p,
                frequency_penalty=model_config.frequency_penalty,
                presence_penalty=model_config.presence_penalty,
                **kwargs
            )
            return response.choices[0].message.content
            
        elif model_config.provider == ModelProvider.GOOGLE:
            model = self.clients[ModelProvider.GOOGLE].GenerativeModel(
                model_config.model_name
            )
            response = model.generate_content(prompt)
            return response.text
            
        elif model_config.provider == ModelProvider.LLAMA:
            response = self.clients[ModelProvider.LLAMA](
                prompt,
                max_tokens=model_config.max_tokens,
                temperature=model_config.temperature,
                top_p=model_config.top_p
            )
            return response['choices'][0]['text']
    
    async def _reset_failure_count(self, model_id: str):
        """Reset failure count after timeout."""
        await asyncio.sleep(self.failure_reset_time)
        async with self.lock:
            self.failure_counts[model_id] = 0 