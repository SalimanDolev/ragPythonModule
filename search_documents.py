#!/usr/bin/env python3
"""
Document Search Script for RAG Module

This script allows users to search through indexed documents
using semantic similarity and retrieve relevant content.
Uses Google Gemini API for generating embeddings and PostgreSQL for storage.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

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

class DocumentSearcher:
    """Handles document search operations for RAG using Gemini embeddings and PostgreSQL."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize the document searcher.
        
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
        
        logger.info(f"Initialized DocumentSearcher with Gemini model: {self.model_name}")
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
    
    def search_documents(self, query: str, n_results: int = None, threshold: float = None) -> List[Dict]:
        """
        Search for documents relevant to the query.
        
        Args:
            query: The search query
            n_results: Number of results to return (uses config default if None)
            threshold: Minimum similarity score threshold (uses config default if None)
            
        Returns:
            List of relevant document chunks with metadata
        """
        # Use config defaults if not provided
        n_results = n_results or self.config.DEFAULT_SEARCH_RESULTS
        threshold = threshold or self.config.DEFAULT_SIMILARITY_THRESHOLD
        
        try:
            # Generate embedding for the query using Gemini
            query_embedding = self.generate_embedding(query)
            
            # Search in PostgreSQL database
            results = self.db_manager.search_similar_embeddings(
                query_embedding=query_embedding,
                n_results=n_results,
                threshold=threshold
            )
            
            # Process results for consistency with old API
            processed_results = []
            for i, result in enumerate(results):
                processed_results.append({
                    'content': result['chunk_text'],
                    'metadata': {
                        'source': result['file_name'],
                        'chunk_index': result['chunk_index'],
                        'total_chunks': result['total_chunks'],
                        'split_strategy': result['split_strategy'],
                        'created_at': result['created_at']
                    },
                    'similarity_score': result['similarity'],
                    'rank': i + 1
                })
            
            if self.config.LOG_API_CALLS:
                logger.info(f"Found {len(processed_results)} relevant results for query: '{query}'")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def search_by_source(self, query: str, source_file: str, n_results: int = None) -> List[Dict]:
        """
        Search for documents within a specific source file.
        
        Args:
            query: The search query
            source_file: Name of the source file to search in
            n_results: Number of results to return (uses config default if None)
            
        Returns:
            List of relevant document chunks from the specified source
        """
        # Use config default if not provided
        n_results = n_results or self.config.DEFAULT_SEARCH_RESULTS
        
        try:
            # Generate embedding for the query using Gemini
            query_embedding = self.generate_embedding(query)
            
            # Search in PostgreSQL database with file filter
            results = self.db_manager.search_similar_embeddings(
                query_embedding=query_embedding,
                n_results=n_results,
                threshold=0.0,  # No threshold for source-specific search
                file_name=source_file
            )
            
            # Process results for consistency with old API
            processed_results = []
            for i, result in enumerate(results):
                processed_results.append({
                    'content': result['chunk_text'],
                    'metadata': {
                        'source': result['file_name'],
                        'chunk_index': result['chunk_index'],
                        'total_chunks': result['total_chunks'],
                        'split_strategy': result['split_strategy'],
                        'created_at': result['created_at']
                    },
                    'similarity_score': result['similarity'],
                    'rank': i + 1
                })
            
            if self.config.LOG_API_CALLS:
                logger.info(f"Found {len(processed_results)} results in {source_file} for query: '{query}'")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching by source: {str(e)}")
            raise
    
    def get_similar_documents(self, document_chunk: str, n_results: int = None) -> List[Dict]:
        """
        Find documents similar to a given document chunk.
        
        Args:
            document_chunk: The document chunk to find similar documents for
            n_results: Number of similar documents to return (uses config default if None)
            
        Returns:
            List of similar document chunks
        """
        # Use config default if not provided
        n_results = n_results or self.config.DEFAULT_SEARCH_RESULTS
        
        try:
            # Generate embedding for the document chunk using Gemini
            chunk_embedding = self.generate_embedding(document_chunk)
            
            # Search for similar documents
            results = self.db_manager.search_similar_embeddings(
                query_embedding=chunk_embedding,
                n_results=n_results + 1,  # +1 because the original might be in results
                threshold=0.0  # No threshold for similarity search
            )
            
            # Process results, excluding exact matches
            processed_results = []
            for i, result in enumerate(results):
                # Skip if this is the same document chunk
                if result['chunk_text'].strip() != document_chunk.strip():
                    processed_results.append({
                        'content': result['chunk_text'],
                        'metadata': {
                            'source': result['file_name'],
                            'chunk_index': result['chunk_index'],
                            'total_chunks': result['total_chunks'],
                            'split_strategy': result['split_strategy'],
                            'created_at': result['created_at']
                        },
                        'similarity_score': result['similarity'],
                        'rank': i + 1
                    })
                    
                    if len(processed_results) >= n_results:
                        break
            
            if self.config.LOG_API_CALLS:
                logger.info(f"Found {len(processed_results)} similar documents")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the indexed documents."""
        try:
            stats = self.db_manager.get_collection_stats()
            return {
                'total_chunks': stats.get('total_embeddings', 0),
                'unique_sources': stats.get('unique_files', 0),

                'database': self.config.POSTGRES_DB,
                'table': self.config.EMBEDDINGS_TABLE,
                'embedding_model': f"Gemini {self.model_name}",
                'vector_dimension': self.config.VECTOR_DIMENSION,
                'split_strategies': stats.get('split_strategies', [])
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

                formatted_output += f"Split Strategy: {metadata.get('split_strategy', 'Unknown')}\n"
                formatted_output += f"Created: {metadata.get('created_at', 'Unknown')}\n\n"
            
            # Truncate content if too long
            content = result['content']
            if len(content) > 500:
                content = content[:500] + "..."
            
            formatted_output += f"Content:\n{content}\n\n"
        
        return formatted_output

def search_documents(query: str, api_key: str = None, n_results: int = None, **kwargs) -> List[Dict]:
    """
    Convenience function to search documents.
    
    Args:
        query: The search query
        api_key: Google Gemini API key (uses config default if None)
        n_results: Number of results to return (uses config default if None)
        **kwargs: Additional arguments passed to DocumentSearcher.search_documents
        
    Returns:
        List of relevant document chunks
    """
    searcher = DocumentSearcher(api_key=api_key)
    return searcher.search_documents(query, n_results=n_results, **kwargs)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Search indexed documents using Gemini embeddings and PostgreSQL")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--api-key", help="Google Gemini API key (uses config default if not specified)")
    parser.add_argument("--n-results", type=int, help="Number of results to return (uses config default if not specified)")
    parser.add_argument("--threshold", type=float, help="Similarity threshold (uses config default if not specified)")
    parser.add_argument("--source", help="Limit search to specific source file")
    parser.add_argument("--show-stats", action="store_true", help="Show collection statistics")
    parser.add_argument("--show-config", action="store_true", help="Show current configuration")
    
    args = parser.parse_args()
    
    # Show configuration if requested
    if args.show_config:
        config = get_config()
        config.print_config()
        print()
    
    try:
        searcher = DocumentSearcher(api_key=args.api_key)
        
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
