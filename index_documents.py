#!/usr/bin/env python3
"""
Document Indexing Script for RAG Module

This script processes PDF and DOCX files, extracts text content,
and stores it in a vector database for semantic search.
Uses Google Gemini API for generating embeddings.
"""

import os
import logging
from typing import Optional, List
from pathlib import Path

import PyPDF2
from docx import Document
import chromadb
import google.generativeai as genai
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentIndexer:
    """Handles document indexing for RAG operations using Gemini embeddings."""
    
    def __init__(self, db_path: str = "./chroma_db", api_key: str = None, model_name: str = "models/embedding-001"):
        """
        Initialize the document indexer.
        
        Args:
            db_path: Path to store the ChromaDB database
            api_key: Google Gemini API key (if None, will try to get from GOOGLE_API_KEY env var)
            model_name: Name of the Gemini embedding model to use
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize Gemini API
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key is None:
                raise ValueError("Google Gemini API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        genai.configure(api_key=api_key)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized DocumentIndexer with Gemini model: {model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Gemini API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of float values representing the embedding
        """
        try:
            # Use Gemini's embedding model
            embedding_model = genai.get_model('models/embedding-001')
            result = embedding_model.embed_content(text)
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
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Overlap between consecutive chunks
            
        Returns:
            List of text chunks
        """
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
        
        logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def index_document(self, file_path: str, chunk_size: int = 1000, overlap: int = 200) -> bool:
        """
        Index a document by extracting text and storing it in the vector database.
        
        Args:
            file_path: Path to the document file (PDF or DOCX)
            chunk_size: Maximum size of each text chunk
            overlap: Overlap between consecutive chunks
            
        Returns:
            True if indexing was successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
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
            
            # Chunk the text
            chunks = self.chunk_text(text, chunk_size, overlap)
            
            # Generate embeddings using Gemini API
            logger.info("Generating embeddings with Gemini API...")
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.generate_embedding(chunk)
                    embeddings.append(embedding)
                    if (i + 1) % 10 == 0:  # Log progress every 10 chunks
                        logger.info(f"Generated embeddings for {i + 1}/{len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Error generating embedding for chunk {i}: {str(e)}")
                    raise
            
            # Prepare metadata for each chunk
            metadatas = [
                {
                    "source": str(file_path),
                    "chunk_index": i,
                    "file_type": file_path.suffix.lower(),
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=[f"{file_path.stem}_chunk_{i}" for i in range(len(chunks))]
            )
            
            logger.info(f"Successfully indexed {file_path} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document {file_path}: {str(e)}")
            return False
    
    def get_collection_info(self) -> dict:
        """Get information about the indexed documents."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "database_path": self.db_path,
            "embedding_model": f"Gemini {self.model_name}"
        }

def index_document(file_path: str, db_path: str = "./chroma_db", api_key: str = None, **kwargs) -> bool:
    """
    Convenience function to index a single document.
    
    Args:
        file_path: Path to the document file
        db_path: Path to store the ChromaDB database
        api_key: Google Gemini API key
        **kwargs: Additional arguments passed to DocumentIndexer.index_document
        
    Returns:
        True if indexing was successful, False otherwise
    """
    indexer = DocumentIndexer(db_path=db_path, api_key=api_key)
    return indexer.index_document(file_path, **kwargs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Index documents for RAG search using Gemini embeddings")
    parser.add_argument("file_path", help="Path to the document file (PDF or DOCX)")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to ChromaDB database")
    parser.add_argument("--api-key", help="Google Gemini API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    success = index_document(
        args.file_path,
        db_path=args.db_path,
        api_key=args.api_key,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    if success:
        print(f"Successfully indexed: {args.file_path}")
    else:
        print(f"Failed to index: {args.file_path}")
        exit(1)
