#!/usr/bin/env python3
"""
Document Search Script for RAG Module

This script searches indexed documents using Gemini embeddings and PostgreSQL.
Input: textual query from user
Output: 5 most relevant chunks using cosine similarity
"""

import logging
import argparse
from typing import List, Dict
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

def search_documents(query: str, api_key: str, model_name: str) -> List[Dict]:
    """Search for documents relevant to the query."""
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query, api_key, model_name)
        
        # Search in database
        db_manager = DatabaseManager()
        results = db_manager.search_similar_embeddings(
            query_embedding=query_embedding,
            n_results=5
        )
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'rank': i + 1,
                'content': result['chunk_text'],
                'source': result['file_name'],
                'similarity': result['similarity']
            })
        
        return formatted_results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise

def main():
    """Main function to handle CLI arguments and execute search."""
    parser = argparse.ArgumentParser(description="Search indexed documents using Gemini embeddings")
    parser.add_argument("query", help="Search query")

    
    args = parser.parse_args()
    
    try:
        # Get configuration from environment variables
        config = get_config()
        
        # Search documents (always returns 5 results as per assignment requirements)
        results = search_documents(args.query, config.GEMINI_API_KEY, config.GEMINI_EMBEDDING_MODEL)
        
        # Display results
        if results:
            logger.info(f"Found {len(results)} relevant results:")
            for result in results:
                logger.info(f"Result {result['rank']} (Score: {result['similarity']:.3f})")
                logger.info(f"Source: {result['source']}")
                logger.info(f"Content: {result['content'][:200]}...")
                logger.info("-" * 50)
        else:
            logger.info("No results found.")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
