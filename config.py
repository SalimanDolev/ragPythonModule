#!/usr/bin/env python3
"""
Configuration module for RAG Python Module

This module handles loading and managing configuration settings from
environment variables and .env files with comprehensive defaults.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

class Config:
    """Configuration class for RAG module settings."""
    
    def __init__(self):
        """Initialize configuration with default values and environment overrides."""
        
        # =============================================================================
        # REQUIRED: Google Gemini API Configuration
        # =============================================================================
        
        # Google Gemini API key (REQUIRED)
        self.GOOGLE_API_KEY = self._get_env('GOOGLE_API_KEY', required=True)
        
        # =============================================================================
        # REQUIRED: PostgreSQL Database Configuration
        # =============================================================================
        
        # PostgreSQL host
        self.POSTGRES_HOST = self._get_env('POSTGRES_HOST', default='localhost')
        
        # PostgreSQL port
        self.POSTGRES_PORT = self._get_env_int('POSTGRES_PORT', default=5432)
        
        # PostgreSQL database name
        self.POSTGRES_DB = self._get_env('POSTGRES_DB', default='rag_database')
        
        # PostgreSQL username
        self.POSTGRES_USER = self._get_env('POSTGRES_USER', default='rag_user')
        
        # PostgreSQL password
        self.POSTGRES_PASSWORD = self._get_env('POSTGRES_PASSWORD', required=True)
        
        # PostgreSQL connection string (auto-generated)
        self.POSTGRES_CONNECTION_STRING = self._build_postgres_connection_string()
        
        # =============================================================================
        # OPTIONAL: Database Schema Configuration
        # =============================================================================
        
        # Table name for storing embeddings
        self.EMBEDDINGS_TABLE = self._get_env('EMBEDDINGS_TABLE', default='document_embeddings')
        
        # Vector dimension (from Gemini embedding model)
        self.VECTOR_DIMENSION = self._get_env_int('VECTOR_DIMENSION', default=768)
        
        # =============================================================================
        # OPTIONAL: Embedding Model Configuration
        # =============================================================================
        
        # Gemini embedding model
        self.GEMINI_EMBEDDING_MODEL = self._get_env('GEMINI_EMBEDDING_MODEL', default='models/gemini-embedding-001')
        
        # =============================================================================
        # OPTIONAL: Text Processing Configuration
        # =============================================================================
        
        # Default chunk size for text processing
        self.DEFAULT_CHUNK_SIZE = self._get_env_int('DEFAULT_CHUNK_SIZE', default=1000)
        
        # Default overlap between text chunks
        self.DEFAULT_CHUNK_OVERLAP = self._get_env_int('DEFAULT_CHUNK_OVERLAP', default=200)
        
        # Split strategy to use
        self.SPLIT_STRATEGY = self._get_env('SPLIT_STRATEGY', default='sentence_boundary')
        
        # =============================================================================
        # OPTIONAL: Search Configuration
        # =============================================================================
        
        # Default number of search results
        self.DEFAULT_SEARCH_RESULTS = self._get_env_int('DEFAULT_SEARCH_RESULTS', default=5)
        
        # Default similarity threshold
        self.DEFAULT_SIMILARITY_THRESHOLD = self._get_env_float('DEFAULT_SIMILARITY_THRESHOLD', default=0.5)
        
        # =============================================================================
        # OPTIONAL: Logging Configuration
        # =============================================================================
        
        # Logging level
        self.LOG_LEVEL = self._get_env('LOG_LEVEL', default='INFO')
        
        # Enable/disable API call logging
        self.LOG_API_CALLS = self._get_env_bool('LOG_API_CALLS', default=True)
        
        # =============================================================================
        # OPTIONAL: Performance Configuration
        # =============================================================================
        
        # Batch size for embedding generation
        self.EMBEDDING_BATCH_SIZE = self._get_env_int('EMBEDDING_BATCH_SIZE', default=10)
        
        # API request timeout
        self.API_TIMEOUT = self._get_env_int('API_TIMEOUT', default=30)
        
        # Database connection pool size
        self.DB_POOL_SIZE = self._get_env_int('DB_POOL_SIZE', default=5)
        
        # =============================================================================
        # OPTIONAL: Security Configuration
        # =============================================================================
        
        # Enable API key validation
        self.VALIDATE_API_KEY = self._get_env_bool('VALIDATE_API_KEY', default=True)
        
        # =============================================================================
        # DEVELOPMENT/DEBUGGING (OPTIONAL)
        # =============================================================================
        
        # Debug mode
        self.DEBUG_MODE = self._get_env_bool('DEBUG_MODE', default=False)
        
        # Test mode
        self.TEST_MODE = self._get_env_bool('TEST_MODE', default=False)
        
        # Apply debug mode settings
        if self.DEBUG_MODE:
            self.LOG_LEVEL = 'DEBUG'
            self.LOG_API_CALLS = True
        
        # Apply test mode settings
        if self.TEST_MODE:
            self.DEFAULT_CHUNK_SIZE = 500
            self.DEFAULT_SEARCH_RESULTS = 3
            self.EMBEDDING_BATCH_SIZE = 5
    
    def _build_postgres_connection_string(self) -> str:
        """Build PostgreSQL connection string from configuration."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    def _get_env(self, key: str, default: Optional[str] = None, required: bool = False) -> str:
        """
        Get environment variable with optional default and required validation.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If required variable is missing
        """
        value = os.getenv(key, default)
        
        if required and not value:
            raise ValueError(
                f"Required environment variable '{key}' is not set. "
                f"Please set it in your .env file or environment. "
                f"See env.example for configuration options."
            )
        
        return value
    
    def _get_env_int(self, key: str, default: int) -> int:
        """Get environment variable as integer."""
        value = self._get_env(key, str(default))
        try:
            return int(value)
        except ValueError:
            logging.warning(f"Invalid integer value for {key}: {value}, using default: {default}")
            return default
    
    def _get_env_float(self, key: str, default: float) -> float:
        """Get environment variable as float."""
        value = self._get_env(key, str(default))
        try:
            return float(value)
        except ValueError:
            logging.warning(f"Invalid float value for {key}: {value}, using default: {default}")
            return default
    
    def _get_env_bool(self, key: str, default: bool) -> bool:
        """Get environment variable as boolean."""
        value = self._get_env(key, str(default).lower())
        if value.lower() in ('true', '1', 'yes', 'on'):
            return True
        elif value.lower() in ('false', '0', 'no', 'off'):
            return False
        else:
            logging.warning(f"Invalid boolean value for {key}: {value}, using default: {default}")
            return default
    
    def validate(self) -> bool:
        """
        Validate configuration settings.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        errors = []
        
        # Validate required settings
        if not self.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required")
        
        if not self.POSTGRES_PASSWORD:
            errors.append("POSTGRES_PASSWORD is required")
        
        # Validate numeric ranges
        if self.DEFAULT_CHUNK_SIZE < 100:
            errors.append("DEFAULT_CHUNK_SIZE should be at least 100")
        
        if self.DEFAULT_CHUNK_OVERLAP < 0:
            errors.append("DEFAULT_CHUNK_OVERLAP should be non-negative")
        
        if self.DEFAULT_CHUNK_OVERLAP >= self.DEFAULT_CHUNK_SIZE:
            errors.append("DEFAULT_CHUNK_OVERLAP should be less than DEFAULT_CHUNK_SIZE")
        
        if not 0.0 <= self.DEFAULT_SIMILARITY_THRESHOLD <= 1.0:
            errors.append("DEFAULT_SIMILARITY_THRESHOLD should be between 0.0 and 1.0")
        
        if self.DEFAULT_SEARCH_RESULTS < 1:
            errors.append("DEFAULT_SEARCH_RESULTS should be at least 1")
        
        if self.EMBEDDING_BATCH_SIZE < 1:
            errors.append("EMBEDDING_BATCH_SIZE should be at least 1")
        
        if self.API_TIMEOUT < 1:
            errors.append("API_TIMEOUT should be at least 1 second")
        
        if self.POSTGRES_PORT < 1 or self.POSTGRES_PORT > 65535:
            errors.append("POSTGRES_PORT should be between 1 and 65535")
        
        if self.VECTOR_DIMENSION < 1:
            errors.append("VECTOR_DIMENSION should be at least 1")
        
        # Report errors
        if errors:
            for error in errors:
                logging.error(f"Configuration error: {error}")
            return False
        
        return True
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database connection information (without sensitive data)."""
        return {
            'host': self.POSTGRES_HOST,
            'port': self.POSTGRES_PORT,
            'database': self.POSTGRES_DB,
            'user': self.POSTGRES_USER,
            'table': self.EMBEDDINGS_TABLE,
            'vector_dimension': self.VECTOR_DIMENSION
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            'POSTGRES_HOST': self.POSTGRES_HOST,
            'POSTGRES_PORT': self.POSTGRES_PORT,
            'POSTGRES_DB': self.POSTGRES_DB,
            'POSTGRES_USER': self.POSTGRES_USER,
            'EMBEDDINGS_TABLE': self.EMBEDDINGS_TABLE,
            'VECTOR_DIMENSION': self.VECTOR_DIMENSION,
            'GEMINI_EMBEDDING_MODEL': self.GEMINI_EMBEDDING_MODEL,
            'DEFAULT_CHUNK_SIZE': self.DEFAULT_CHUNK_SIZE,
            'DEFAULT_CHUNK_OVERLAP': self.DEFAULT_CHUNK_OVERLAP,
            'SPLIT_STRATEGY': self.SPLIT_STRATEGY,
            'DEFAULT_SEARCH_RESULTS': self.DEFAULT_SEARCH_RESULTS,
            'DEFAULT_SIMILARITY_THRESHOLD': self.DEFAULT_SIMILARITY_THRESHOLD,
            'LOG_LEVEL': self.LOG_LEVEL,
            'LOG_API_CALLS': self.LOG_API_CALLS,
            'EMBEDDING_BATCH_SIZE': self.EMBEDDING_BATCH_SIZE,
            'API_TIMEOUT': self.API_TIMEOUT,
            'DB_POOL_SIZE': self.DB_POOL_SIZE,
            'VALIDATE_API_KEY': self.VALIDATE_API_KEY,
            'DEBUG_MODE': self.DEBUG_MODE,
            'TEST_MODE': self.TEST_MODE,
        }
    
    def print_config(self, include_sensitive: bool = False):
        """Print current configuration (for debugging)."""
        print("=" * 60)
        print("RAG Module Configuration")
        print("=" * 60)
        
        config_dict = self.to_dict()
        
        if include_sensitive:
            config_dict['GOOGLE_API_KEY'] = self.GOOGLE_API_KEY[:10] + "..." if self.GOOGLE_API_KEY else "Not set"
            config_dict['POSTGRES_PASSWORD'] = "***" if self.POSTGRES_PASSWORD else "Not set"
        
        for key, value in config_dict.items():
            print(f"{key}: {value}")
        
        print("=" * 60)

# Global configuration instance
_config: Optional[Config] = None

def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config()
        if _config.VALIDATE_API_KEY:
            _config.validate()
    return _config

def load_config() -> Config:
    """
    Load and validate configuration.
    
    Returns:
        Validated configuration instance
        
    Raises:
        ValueError: If configuration is invalid
    """
    config = get_config()
    if not config.validate():
        raise ValueError("Invalid configuration. Check the logs for details.")
    return config

def reset_config():
    """Reset the global configuration instance (useful for testing)."""
    global _config
    _config = None
