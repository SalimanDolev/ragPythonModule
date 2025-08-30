"""
RAG Python Module for Document Indexing and Search

A complete RAG system using Google Gemini AI, PostgreSQL, and pgvector for semantic search.

Features:
- Index PDF/DOCX files with 3 chunking strategies (fixed-size, sentence, paragraph)
- Semantic search using Gemini API embeddings and cosine similarity
- PostgreSQL storage with pgvector for vector operations

Cosine Similarity:
Measures semantic closeness between text vectors (-1 to 1, where 1 = identical meaning).
Enables intelligent search beyond keyword matching.

Usage:
    from ragPythonModule import index_document, search_documents
    index_document("doc.pdf", chunk_strategy="sentence")
    results = search_documents("your query")
"""

from .index_documents import index_document, DocumentIndexer
from .search_documents import search_documents, DocumentSearcher
from .config import Config, get_config
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
    "DatabaseManager"
]
