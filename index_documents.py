#!/usr/bin/env python3
"""
Document Indexing Script for RAG Module

This script processes PDF and DOCX files, extracts text content,
and stores it in a PostgreSQL database for semantic search.
Uses Google Gemini API for generating embeddings.
"""

import logging
import argparse
from typing import List
from pathlib import Path

import PyPDF2
from docx import Document
import google.generativeai as genai

from config import get_config
from database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_embedding(text: str, api_key: str, model_name: str) -> List[float]:
    """Generate embedding using Gemini API."""
    try:
        genai.configure(api_key=api_key)
        result = genai.embed_content(model=model_name, content=text)
        return [float(x) for x in result['embedding']]
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text += page_text + "\n"
            
            logger.info(f"Successfully extracted text from PDF: {file_path}")
            return text.strip()
            
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        raise

def extract_text_from_docx(file_path: str) -> str:
    """Extract text content from a DOCX file."""
    try:
        doc = Document(file_path)
        text = ""
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        
        logger.info(f"Successfully extracted text from DOCX: {file_path}")
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        raise

def _chunk_fixed_size(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    
    for start in range(0, len(text), chunk_size - overlap):
        end = start + chunk_size
        
        # Try to break at sentence boundary if not the last chunk
        if end < len(text):
            for i in range(end, max(start + chunk_size - 100, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} fixed-size chunks (size: {chunk_size}, overlap: {overlap})")
    return chunks

def _chunk_by_sentences(text: str) -> List[str]:
    """Split text into chunks based on sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    logger.info(f"Created {len(chunks)} sentence-based chunks")
    return chunks

def _chunk_by_paragraphs(text: str) -> List[str]:
    """Split text into chunks based on paragraph boundaries."""
    paragraphs = text.split('\n\n')
    
    chunks = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            chunks.append(paragraph)
    
    logger.info(f"Created {len(chunks)} paragraph-based chunks")
    return chunks

def chunk_text(text: str, chunk_strategy: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks using different strategies."""
    logger.info(f"Using chunking strategy: {chunk_strategy}")
    
    if chunk_strategy == 'sentence':
        return _chunk_by_sentences(text)
    elif chunk_strategy == 'paragraph':
        return _chunk_by_paragraphs(text)
    else:  # fixed-size (default)
        return _chunk_fixed_size(text, chunk_size, overlap)

def index_document(file_path: str, chunk_strategy: str, api_key: str, model_name: str, chunk_size: int = 1000, overlap: int = 200) -> bool:
    """Index a document by extracting text and storing it in the PostgreSQL database."""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        # Extract text based on file type
        if file_path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(str(file_path))
        elif file_path.suffix.lower() == '.docx':
            text = extract_text_from_docx(str(file_path))
        else:
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return False
        
        if not text.strip():
            logger.warning(f"No text content extracted from {file_path}")
            return False
        
        # Chunk the text using specified strategy
        chunks = chunk_text(text, chunk_strategy, chunk_size, overlap)
        
        # Generate embeddings using Gemini API
        logger.info("Generating embeddings with Gemini API...")
        embeddings_data = []
        
        for chunk in chunks:
            embedding = generate_embedding(chunk, api_key, model_name)
            
            embeddings_data.append({
                'chunk_text': chunk,
                'embedding': embedding,
                'file_name': str(file_path),
                'split_strategy': chunk_strategy,
                'created_at': None  # Will use database default
            })
        
        # Store in PostgreSQL database
        db_manager = DatabaseManager()
        success = db_manager.insert_embeddings(embeddings_data)
        
        if success:
            logger.info(f"Successfully indexed {file_path} with {len(chunks)} chunks")
            return True
        else:
            logger.error(f"Failed to store embeddings in database for {file_path}")
            return False
        
    except Exception as e:
        logger.error(f"Error indexing document {file_path}: {e}")
        return False

def main():
    """Main function to handle CLI arguments and execute indexing."""
    parser = argparse.ArgumentParser(description="Index documents for RAG search using Gemini embeddings")
    parser.add_argument("file_path", help="Path to the document file (PDF or DOCX)")
    parser.add_argument("--chunk-strategy", choices=['fixed-size', 'sentence', 'paragraph'], 
                       default='fixed-size', help="Text chunking strategy")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for fixed-size strategy (default: 1000)")
    parser.add_argument("--overlap", type=int, default=200, help="Overlap for fixed-size strategy (default: 200)")
    
    args = parser.parse_args()
    
    try:
        # Get configuration from environment variables
        config = get_config()
        
        # Index document
        success = index_document(
            args.file_path,
            args.chunk_strategy,
            config.GEMINI_API_KEY,
            config.GEMINI_EMBEDDING_MODEL,
            args.chunk_size,
            args.overlap
        )
        
        if success:
            logger.info(f"Successfully indexed: {args.file_path}")
        else:
            logger.error(f"Failed to index: {args.file_path}")
            exit(1)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
