#!/usr/bin/env python3
"""
Document Search Script for RAG Module

This script allows users to search through indexed documents
using semantic similarity and retrieve relevant content.
Uses Google Gemini API for generating embeddings.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import chromadb
import google.generativeai as genai
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentSearcher:
    """Handles document search operations for RAG using Gemini embeddings."""
    
    def __init__(self, db_path: str = "./chroma_db", api_key: str = None, model_name: str = "models/embedding-001"):
        """
        Initialize the document searcher.
        
        Args:
            db_path: Path to the ChromaDB database
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
        self.embedding_model = genai.GenerativeModel(model_name)
        
        # Get the collection
        try:
            self.collection = self.client.get_collection(name="documents")
            logger.info(f"Connected to existing collection in {db_path}")
            logger.info(f"Using Gemini embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Collection not found. Please index documents first: {str(e)}")
            raise
    
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
    
    def search_documents(self, query: str, n_results: int = 5, threshold: float = 0.5) -> List[Dict]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: The search query
            n_results: Number of results to return
            threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            # Generate embedding for the query using Gemini
            query_embedding = self.generate_embedding(query)
            
            # Search in the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process and filter results
            processed_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (cosine similarity)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= threshold:
                        processed_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'rank': i + 1
                        })
            
            logger.info(f"Found {len(processed_results)} relevant results for query: '{query}'")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def search_by_source(self, query: str, source_file: str, n_results: int = 5) -> List[Dict]:
        """
        Search for documents within a specific source file.
        
        Args:
            query: The search query
            source_file: Name of the source file to search in
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks from the specified source
        """
        try:
            # Generate embedding for the query using Gemini
            query_embedding = self.generate_embedding(query)
            
            # Search with source filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where={"source": source_file},
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    similarity_score = 1 - distance
                    processed_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(processed_results)} results in {source_file} for query: '{query}'")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching by source: {str(e)}")
            raise
    
    def get_similar_documents(self, document_chunk: str, n_results: int = 5) -> List[Dict]:
        """
        Find documents similar to a given document chunk.
        
        Args:
            document_chunk: The document chunk to find similar documents for
            n_results: Number of similar documents to return
            
        Returns:
            List of similar document chunks
        """
        try:
            # Generate embedding for the document chunk using Gemini
            chunk_embedding = self.generate_embedding(document_chunk)
            
            # Search for similar documents
            results = self.collection.query(
                query_embeddings=[chunk_embedding],
                n_results=n_results + 1,  # +1 because the original might be in results
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results, excluding exact matches
            processed_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Skip if this is the same document chunk
                    if doc.strip() != document_chunk.strip():
                        similarity_score = 1 - distance
                        processed_results.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'rank': i + 1
                        })
                        
                        if len(processed_results) >= n_results:
                            break
            
            logger.info(f"Found {len(processed_results)} similar documents")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        try:
            count = self.collection.count()
            
            # Get unique sources
            all_metadata = self.collection.get(include=["metadatas"])
            sources = set()
            file_types = set()
            
            if all_metadata['metadatas']:
                for metadata in all_metadata['metadatas']:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
                    if metadata and 'file_type' in metadata:
                        file_types.add(metadata['file_type'])
            
            return {
                'total_chunks': count,
                'unique_sources': len(sources),
                'file_types': list(file_types),
                'database_path': self.db_path,
                'embedding_model': f"Gemini {self.model_name}"
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def format_search_results(self, results: List[Dict], show_metadata: bool = True) -> str:
        """
        Format search results for display.
        
        Args:
            results: List of search results
            show_metadata: Whether to show metadata information
            
        Returns:
            Formatted string representation of results
        """
        if not results:
            return "No results found."
        
        formatted_output = f"Found {len(results)} relevant results:\n\n"
        
        for i, result in enumerate(results):
            formatted_output += f"Result {i + 1} (Score: {result['similarity_score']:.3f})\n"
            formatted_output += f"{'='*50}\n"
            
            if show_metadata and result['metadata']:
                metadata = result['metadata']
                formatted_output += f"Source: {metadata.get('source', 'Unknown')}\n"
                formatted_output += f"Chunk: {metadata.get('chunk_index', 'Unknown')}/{metadata.get('total_chunks', 'Unknown')}\n"
                formatted_output += f"File Type: {metadata.get('file_type', 'Unknown')}\n\n"
            
            # Truncate content if too long
            content = result['content']
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted_output += f"Content:\n{content}\n\n"
        
        return formatted_output

def search_documents(query: str, db_path: str = "./chroma_db", api_key: str = None, n_results: int = 5, **kwargs) -> List[Dict]:
    """
    Convenience function to search documents.
    
    Args:
        query: The search query
        db_path: Path to the ChromaDB database
        api_key: Google Gemini API key
        n_results: Number of results to return
        **kwargs: Additional arguments passed to DocumentSearcher.search_documents
        
    Returns:
        List of relevant document chunks
    """
    searcher = DocumentSearcher(db_path=db_path, api_key=api_key)
    return searcher.search_documents(query, n_results=n_results, **kwargs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search indexed documents using Gemini embeddings")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to ChromaDB database")
    parser.add_argument("--api-key", help="Google Gemini API key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
    parser.add_argument("--source", help="Limit search to specific source file")
    parser.add_argument("--show-stats", action="store_true", help="Show collection statistics")
    
    args = parser.parse_args()
    
    try:
        searcher = DocumentSearcher(db_path=args.db_path, api_key=args.api_key)
        
        if args.show_stats:
            stats = searcher.get_collection_stats()
            print("Collection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        
        if args.source:
            results = searcher.search_by_source(args.query, args.source, args.n_results)
        else:
            results = searcher.search_documents(args.query, args.n_results, args.threshold)
        
        if results:
            formatted_results = searcher.format_search_results(results)
            print(formatted_results)
        else:
            print("No results found.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
