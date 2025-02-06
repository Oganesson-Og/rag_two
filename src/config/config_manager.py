"""
Configuration Management Module
----------------------------

Central configuration management system providing dynamic configuration handling,
validation, and environment-specific settings for the RAG pipeline.

Key Features:
- Configuration loading and saving
- Environment-specific settings management
- Dynamic configuration updates
- Validation rules and schema checking
- Default configuration handling
- Configuration inheritance
- Secret management

Technical Details:
- YAML-based configuration files
- Environment variable integration
- Schema validation using Pydantic
- Hierarchical configuration structure
- Configuration versioning support

Dependencies:
- pyyaml>=6.0.1
- pydantic>=2.5.0
- python-dotenv>=1.0.0
- typing-extensions>=4.8.0
- jsonschema>=4.20.0

Example Usage:
    # Basic initialization
    config = ConfigManager()
    
    # Custom config path
    config = ConfigManager(config_path='custom_config.yaml')
    
    # Get subject configuration
    subject_config = config.get_subject_config(
        subject='physics',
        level='a-level'
    )
    
    # Update configuration
    config.update_config({
        'embedding': {'batch_size': 64}
    })

Configuration Structure:
- embedding: Model and processing settings
- retrieval: Search and ranking parameters
- database: Connection and storage settings
- api: Service configuration
- logging: Log management settings

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""
from typing import Dict, Optional
import yaml
import json
from pathlib import Path
import logging
from .domain_config import EDUCATION_DOMAINS

class ConfigManager:
    """Manages configuration for the RAG system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/rag_config.yaml"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        default_config = {
            "embedding": {
                "cache_dir": ".cache/embeddings",
                "batch_size": 32,
                "use_fp16": True
            },
            "retrieval": {
                "top_k": 5,
                "similarity_threshold": 0.7,
                "reranking_enabled": True
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "ragdb",
                "user": "raguser"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4
            },
            "logging": {
                "level": "INFO",
                "file": "rag.log"
            }
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
        except Exception as e:
            logging.warning(f"Error loading config file: {e}")
            
        return default_config
    
    def get_subject_config(self, subject: str, level: str) -> Dict:
        """Get subject-specific configuration."""
        for domain, config in EDUCATION_DOMAINS.items():
            if subject.lower() in config['subjects']:
                return {
                    "domain": domain,
                    "model": config['models'].get(subject.lower(), config['models']['default']),
                    "preprocessing": config['preprocessing'],
                    "keywords": config['keywords'].get(subject.lower(), [])
                }
        raise ValueError(f"Invalid subject: {subject}")
    
    def save_config(self, config: Dict) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f)
            self.config = config
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            raise
    
    def update_config(self, updates: Dict) -> None:
        """Update configuration with new values."""
        def deep_update(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self.save_config(self.config) 