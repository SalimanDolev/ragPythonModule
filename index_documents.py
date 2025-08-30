#!/usr/bin/env python3
"""
Document Indexing Script for RAG Module

This script processes PDF and DOCX files, extracts text content,
and stores it in a PostgreSQL database for semantic search.
Uses Google Gemini API for generating embeddings.
"""

import os
import logging
import json
from typing import Optional, List
from pathlib import Path

import PyPDF2
from docx import Document
import google.generativeai as genai
import numpy as np

# Import configuration and database
from config import get_config, load_config
from database import DatabaseManager

# Configure logging based on config
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIndexer:
    """Handles document indexing for RAG operations using Gemini embeddings and PostgreSQL."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the document indexer.
        
        Args:
            api_key: Google Gemini API key (uses config default if None)
            model_name: Name of the Gemini embedding model (uses config default if None)
        """
        # Load configuration
        self.config = get_config()
        
        # Use provided values or defaults from config
        self.api_key = api_key or self.config.GOOGLE_API_KEY
        self.model_name = model_name or self.config.GEMINI_EMBEDDING_MODEL
        
        # Initialize Gemini API
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key is required. Set GOOGLE_API_KEY environment variable, "
                "pass api_key parameter, or configure it in your .env file. "
                "See env.example for configuration options."
            )
        
        genai.configure(api_key=self.api_key)
        
        # Initialize database manager
        self.db_manager = DatabaseManager()
        
        logger.info(f"Initialized DocumentIndexer with Gemini model: {self.model_name}")
        logger.info(f"Using PostgreSQL database: {self.config.POSTGRES_DB}")
        logger.info(f"Table: {self.config.EMBEDDINGS_TABLE}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Gemini API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of float values representing the embedding
        """
        try:
            # Use Gemini's embedding API directly
            result = genai.embed_content(
                model=self.model_name,
                content=text
            )
            embedding = result['embedding']
            
            # Convert to list of floats
            return [float(x) for x in embedding]
            
        except Exception as e:
            logger.error(f"Error generating embedding with Gemini: {str(e)}")
            raise
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                logger.info(f"Successfully extracted text from PDF: {file_path}")
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text content from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text content
        """
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            logger.info(f"Successfully extracted text from DOCX: {file_path}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
            raise
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None, chunk_strategy: str = None) -> List[str]:
        """
        Split text into chunks using different strategies for better retrieval.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk (uses config default if None)
            overlap: Overlap between consecutive chunks (uses config default if None)
            chunk_strategy: Chunking strategy ('fixed-size', 'sentence', or 'paragraph')
            
        Returns:
            List of text chunks
        """
        # Use config defaults if not provided
        chunk_size = chunk_size or self.config.DEFAULT_CHUNK_SIZE
        overlap = overlap or self.config.DEFAULT_CHUNK_OVERLAP
        chunk_strategy = chunk_strategy or self.config.SPLIT_STRATEGY
        
        logger.info(f"Using chunking strategy: {chunk_strategy}")
        
        if chunk_strategy == 'sentence':
            return self._chunk_by_sentences(text, chunk_size)
        elif chunk_strategy == 'paragraph':
            return self._chunk_by_paragraphs(text, chunk_size)
        else:  # fixed-size (default)
            return self._chunk_fixed_size(text, chunk_size, overlap)
    
    def _chunk_fixed_size(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                for i in range(end, max(start + chunk_size - 100, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        logger.info(f"Created {len(chunks)} fixed-size chunks (size: {chunk_size}, overlap: {overlap})")
        return chunks
    
    def _chunk_by_sentences(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into chunks based on sentence boundaries."""
        # Split by sentence endings (., !, ?)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max size, save current chunk and start new one
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} sentence-based chunks (max size: {max_chunk_size})")
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into chunks based on paragraph boundaries."""
        # Split by double newlines (paragraph breaks)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If adding this paragraph would exceed max size, save current chunk and start new one
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it exists
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} paragraph-based chunks (max size: {max_chunk_size})")
        return chunks
    
    def index_document(self, file_path: str, chunk_size: int = None, overlap: int = None, 
                      force_update: bool = False, chunk_strategy: str = None) -> bool:
        """
        Index a document by extracting text and storing it in the PostgreSQL database.
        Automatically prevents duplicates unless force_update is True.
        
        Args:
            file_path: Path to the document file (PDF or DOCX)
            chunk_size: Maximum size of each text chunk (uses config default if None)
            overlap: Overlap between consecutive chunks (uses config default if None)
            force_update: If True, delete existing embeddings and re-index (default: False)
            chunk_strategy: Text chunking strategy ('fixed-size', 'sentence', or 'paragraph')
            
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            # Check if file already exists in database
            existing_chunks = self.db_manager.get_file_embeddings_count(str(file_path))
            if existing_chunks > 0:
                if force_update:
                    logger.info(f"File {file_path} already exists with {existing_chunks} chunks. Deleting existing data for re-indexing...")
                    self.db_manager.delete_file_embeddings(str(file_path))
                else:
                    logger.warning(f"File {file_path} already exists with {existing_chunks} chunks. Use force_update=True to re-index.")
                    logger.info(f"Existing file will be skipped. Use --force-update flag to override.")
                    return True  # Return True since the file is already indexed
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(str(file_path))
            elif file_path.suffix.lower() == '.docx':
                text = self.extract_text_from_docx(str(file_path))
            else:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return False
            
            if not text.strip():
                logger.warning(f"No text content extracted from {file_path}")
                return False
            
            # Chunk the text using specified strategy
            chunks = self.chunk_text(text, chunk_size, overlap, chunk_strategy)
            
            # Generate embeddings using Gemini API
            logger.info("Generating embeddings with Gemini API...")
            embeddings_data = []
            batch_size = self.config.EMBEDDING_BATCH_SIZE
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                try:
                    for j, chunk in enumerate(batch):
                        embedding = self.generate_embedding(chunk)
                        
                        # Prepare data for database insertion
                        embeddings_data.append({
                            'chunk_text': chunk,
                            'embedding': embedding,
                            'file_name': str(file_path),
                            'split_strategy': self.config.SPLIT_STRATEGY,
                            'chunk_index': i + j,
                            'total_chunks': len(chunks),

                            'metadata': json.dumps({
                                'file_size': file_path.stat().st_size,
                                'total_pages': len(chunks) if file_path.suffix.lower() == '.pdf' else None,
                                'processing_timestamp': str(np.datetime64('now'))
                            })
                        })
                    
                    if self.config.LOG_API_CALLS:
                        logger.info(f"Generated embeddings for chunks {i+1}-{min(i+batch_size, len(chunks))}/{len(chunks)}")
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch starting at chunk {i}: {str(e)}")
                    raise
            
            # Store in PostgreSQL database
            success = self.db_manager.insert_embeddings(embeddings_data)
            
            if success:
                logger.info(f"Successfully indexed {file_path} with {len(chunks)} chunks")
                return True
            else:
                logger.error(f"Failed to store embeddings in database for {file_path}")
                return False
            
        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {str(e)}")
            return False
    
    def get_collection_info(self) -> dict:
        """Get information about the indexed documents."""
        stats = self.db_manager.get_collection_stats()
        return {
            "total_documents": stats.get('total_embeddings', 0),  # Alias for compatibility
            "total_embeddings": stats.get('total_embeddings', 0),
            "unique_files": stats.get('unique_files', 0),
            "database": self.config.POSTGRES_DB,
            "table_name": self.config.EMBEDDINGS_TABLE,  # Alias for compatibility
            "table": self.config.EMBEDDINGS_TABLE,
            "collection_name": self.config.EMBEDDINGS_TABLE,  # Alias for compatibility
            "embedding_model": f"Gemini {self.model_name}",
            "vector_dimension": self.config.VECTOR_DIMENSION
        }
    
    def delete_document(self, file_name: str) -> bool:
        """
        Delete all embeddings for a specific document.
        
        Args:
            file_name: Name of the file to delete
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            success = self.db_manager.delete_file_embeddings(file_name)
            if success:
                logger.info(f"Successfully deleted embeddings for: {file_name}")
            return success
        except Exception as e:
            logger.error(f"Error deleting document {file_name}: {str(e)}")
            return False

def index_document(file_path: str, api_key: str = None, force_update: bool = False, **kwargs) -> bool:
    """
    Convenience function to index a single document.
    
    Args:
        file_path: Path to the document file
        api_key: Google Gemini API key (uses config default if None)
        force_update: If True, delete existing embeddings and re-index (default: False)
        **kwargs: Additional arguments passed to DocumentIndexer.index_document
        
    Returns:
        True if indexing was successful, False otherwise
    """
    indexer = DocumentIndexer(api_key=api_key)
    return indexer.index_document(file_path, force_update=force_update, **kwargs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents for RAG search using Gemini embeddings and PostgreSQL")
    parser.add_argument("file_path", help="Path to the document file (PDF or DOCX)")
    parser.add_argument("--api-key", help="Google Gemini API key (uses config default if not specified)")
    parser.add_argument("--chunk-size", type=int, help="Maximum chunk size (uses config default if not specified)")
    parser.add_argument("--overlap", type=int, help="Overlap between chunks (uses config default if not specified)")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration")
    parser.add_argument("--delete", help="Delete embeddings for specified file")
    parser.add_argument("--force-update", action="store_true", help="Force re-indexing even if file already exists")
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        config = get_config()
        config.print_config()
        print()
    
    # Handle delete operation
    if args.delete:
        indexer = DocumentIndexer(api_key=args.api_key)
        success = indexer.delete_document(args.delete)
        if success:
            print(f"Successfully deleted embeddings for: {args.delete}")
        else:
            print(f"Failed to delete embeddings for: {args.delete}")
            exit(1)
        exit(0)
    
    # Index document
    success = index_document(
        args.file_path,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        force_update=args.force_update
    )
    
    if success:
        print(f"Successfully indexed: {args.file_path}")
    else:
        print(f"Failed to index: {args.file_path}")
        exit(1)
