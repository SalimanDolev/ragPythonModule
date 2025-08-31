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
This module provides two main scripts:
- index_documents.py: Document indexing with multiple chunking strategies
- search_documents.py: Semantic search using Gemini embeddings

For usage examples, see the README.md file.
"""

__version__ = "1.0.0"
__author__ = "RAG Module Developer"
