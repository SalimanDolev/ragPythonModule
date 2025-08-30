"""
RAG Python Module for Document Indexing and Search

This module provides functionality for:
- Indexing PDF and DOCX documents using Gemini embeddings
- Searching through indexed documents using semantic similarity
- Comprehensive configuration management with environment variables
- PostgreSQL database storage with pgvector for vector operations
"""

from .index_documents import index_document, DocumentIndexer
from .search_documents import search_documents, DocumentSearcher
from .config import Config, get_config, load_config
from .database import DatabaseManager

__version__ = "1.0.0"
__author__ = "RAG Module Developer"

__all__ = [
    "index_document", 
    "search_documents", 
    "DocumentIndexer", 
    "DocumentSearcher",
    "Config",
    "get_config",
    "load_config",
    "DatabaseManager"
]
