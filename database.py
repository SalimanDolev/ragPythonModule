#!/usr/bin/env python3
"""
Database module for RAG Python Module

This module handles PostgreSQL database operations including:
- Table creation with required schema
- Insert embeddings
- Search with cosine similarity
"""

import logging
import psycopg2
from typing import List, Dict, Union
from config import get_config

logger = logging.getLogger(__name__)

EmbeddingData = Dict[str, Union[str, List[float], None]]
SearchResult = Dict[str, Union[int, str, float, None]]

class DatabaseManager:
    """Manages PostgreSQL database operations for RAG system."""
    
    def __init__(self):
        self.config = get_config()
        self._create_table_if_not_exists()
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.config.POSTGRES_URL)
    
    def _create_table_if_not_exists(self):
        """Create the embeddings table if it doesn't exist."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config.EMBEDDINGS_TABLE} (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding vector(768) NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                split_strategy VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Table {self.config.EMBEDDINGS_TABLE} is ready")
            
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    def insert_embeddings(self, embeddings_data: List[EmbeddingData]) -> bool:
        """
        Insert embeddings into the database.
        
        Args:
            embeddings_data: List of dictionaries containing embedding data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            insert_sql = f"""
            INSERT INTO {self.config.EMBEDDINGS_TABLE} 
            (chunk_text, embedding, file_name, split_strategy, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            for item in embeddings_data:
                cursor.execute(insert_sql, (
                    item['chunk_text'],
                    item['embedding'],
                    item['file_name'],
                    item['split_strategy'],
                    item.get('created_at')
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Successfully inserted {len(embeddings_data)} embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            return False
    
    def search_similar_embeddings(self, query_embedding: List[float], 
                                 n_results: int = 5) -> List[SearchResult]:
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return (default: 5)
            
        Returns:
            List of similar embeddings with metadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            search_sql = f"""
                SELECT 
                    id, chunk_text, embedding, file_name, split_strategy, 
                    created_at,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {self.config.EMBEDDINGS_TABLE}
                ORDER BY similarity DESC
                LIMIT %s
            """
            
            cursor.execute(search_sql, (query_embedding, n_results))
            
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convert to list of dicts - return exactly what table stores + calculated similarity
            processed_results = []
            for row in results:
                processed_results.append({
                    'id': row[0],                    # Unique identifier
                    'chunk_text': row[1],            # The text chunk
                    'embedding': row[2],             # Embedding vector
                    'file_name': row[3],             # Filename of original document
                    'split_strategy': row[4],        # Split strategy used
                    'created_at': row[5],            # Date/time created
                    'similarity': row[6]             # Calculated similarity score
                })
            
            logger.info(f"Found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            return []
