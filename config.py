#!/usr/bin/env python3
"""
Configuration module for RAG Python Module

This module handles loading and managing configuration settings from
environment variables and .env files.
"""

import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class for the RAG Python Module.
    Simplified to only use the required environment variables.
    """
    
    def __init__(self):
        """Initialize configuration with environment variables."""
        self._load_environment_variables()
        self._validate_configuration()
        self._setup_logging()
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        # Required variables
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        self.POSTGRES_URL = os.getenv('POSTGRES_URL')
        
        # Essential defaults only
        self.EMBEDDINGS_TABLE = 'document_embeddings'
        self.GEMINI_EMBEDDING_MODEL = 'models/embedding-001'
        self.DEFAULT_SEARCH_RESULTS = 5
    
    def _validate_configuration(self):
        """Validate that required configuration is present."""
        errors = []        
        if not self.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")
        
        if not self.POSTGRES_URL:
            errors.append("POSTGRES_URL is required")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

def get_config() -> Config:
    """Get configuration instance."""
    return Config()
