"""
RAG Python Module for Document Indexing and Search

This module provides functionality for:
- Indexing PDF and DOCX documents using Gemini embeddings
- Searching through indexed documents using semantic similarity
"""

from .index_documents import index_document, DocumentIndexer
from .search_documents import search_documents, DocumentSearcher

__version__ = "1.0.0"
__author__ = "RAG Module Developer"

__all__ = [
    "index_document", 
    "search_documents", 
    "DocumentIndexer", 
    "DocumentSearcher"
]
