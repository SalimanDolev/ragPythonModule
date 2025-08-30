#!/usr/bin/env python3
"""
Database module for RAG Python Module

This module handles PostgreSQL database operations including:
- Table creation and management
- Vector storage and retrieval
- Similarity search using pgvector
"""

import logging
import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from config import get_config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL database operations for the RAG module."""
    
    def __init__(self):
        """Initialize database manager with configuration."""
        self.config = get_config()
        self.connection = None
        self._ensure_pgvector_extension()
        self._create_table_if_not_exists()
    
    def _get_connection(self):
        """Get database connection."""
        if self.connection is None or self.connection.closed:
            try:
                self.connection = psycopg2.connect(
                    host=self.config.POSTGRES_HOST,
                    port=self.config.POSTGRES_PORT,
                    database=self.config.POSTGRES_DB,
                    user=self.config.POSTGRES_USER,
                    password=self.config.POSTGRES_PASSWORD
                )
                logger.info(f"Connected to PostgreSQL database: {self.config.POSTGRES_DB}")
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                raise
        return self.connection
    
    def _ensure_pgvector_extension(self):
        """Ensure pgvector extension is installed and enabled."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if pgvector extension exists
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if not cursor.fetchone():
                # Create the extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                logger.info("pgvector extension created successfully")
            else:
                logger.info("pgvector extension already exists")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Failed to ensure pgvector extension: {e}")
            raise
    
    def _create_table_if_not_exists(self):
        """Create the embeddings table if it doesn't exist."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create table with all required columns
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config.EMBEDDINGS_TABLE} (
                id SERIAL PRIMARY KEY,
                chunk_text TEXT NOT NULL,
                embedding vector({self.config.VECTOR_DIMENSION}) NOT NULL,
                file_name VARCHAR(255) NOT NULL,
                split_strategy VARCHAR(100) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_index INTEGER,
                total_chunks INTEGER,
                metadata JSONB
            );
            """
            
            cursor.execute(create_table_sql)
            
            # Create indexes for better performance
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.EMBEDDINGS_TABLE}_file_name 
            ON {self.config.EMBEDDINGS_TABLE} (file_name);
            """)
            
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.EMBEDDINGS_TABLE}_created_at 
            ON {self.config.EMBEDDINGS_TABLE} (created_at);
            """)
            
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.EMBEDDINGS_TABLE}_split_strategy 
            ON {self.config.EMBEDDINGS_TABLE} (split_strategy);
            """)
            
            # Create vector similarity index
            cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.config.EMBEDDINGS_TABLE}_embedding 
            ON {self.config.EMBEDDINGS_TABLE} 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            """)
            
            conn.commit()
            logger.info(f"Table {self.config.EMBEDDINGS_TABLE} created/verified successfully")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    def insert_embeddings(self, embeddings_data: List[Dict[str, Any]]) -> bool:
        """
        Insert multiple embeddings into the database.
        
        Args:
            embeddings_data: List of dictionaries containing embedding data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare insert statement
            insert_sql = f"""
            INSERT INTO {self.config.EMBEDDINGS_TABLE} 
            (chunk_text, embedding, file_name, split_strategy, chunk_index, total_chunks, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            # Prepare data for batch insert
            data_to_insert = []
            for item in embeddings_data:
                data_to_insert.append((
                    item['chunk_text'],
                    item['embedding'],
                    item['file_name'],
                    item['split_strategy'],
                    item.get('chunk_index'),
                    item.get('total_chunks'),
                    item.get('metadata', {})
                ))
            
            # Execute batch insert
            psycopg2.extras.execute_batch(cursor, insert_sql, data_to_insert)
            conn.commit()
            
            logger.info(f"Successfully inserted {len(embeddings_data)} embeddings")
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            if conn:
                conn.rollback()
            return False
    
    def search_similar_embeddings(self, query_embedding: List[float], 
                                 n_results: int = 5, 
                                 threshold: float = 0.5,
                                 file_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            threshold: Minimum similarity threshold
            file_name: Optional filter by file name
            
        Returns:
            List of similar embeddings with metadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build search query
            if file_name:
                search_sql = f"""
                SELECT 
                    id, chunk_text, embedding, file_name, split_strategy, 
                    created_at, chunk_index, total_chunks, metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {self.config.EMBEDDINGS_TABLE}
                WHERE file_name = %s
                ORDER BY similarity DESC
                LIMIT %s
                """
                cursor.execute(search_sql, (query_embedding, file_name, n_results))
            else:
                search_sql = f"""
                SELECT 
                    id, chunk_text, embedding, file_name, split_strategy, 
                    created_at, chunk_index, total_chunks, metadata,
                    1 - (embedding <=> %s::vector) as similarity
                FROM {self.config.EMBEDDINGS_TABLE}
                ORDER BY similarity
                LIMIT %s
                """
                cursor.execute(search_sql, (query_embedding, n_results))
            
            results = cursor.fetchall()
            
            # Convert to list of dicts and sort by similarity (highest first)
            filtered_results = [dict(row) for row in results]
            filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Apply threshold filtering after sorting to ensure we get the best results
            if threshold > 0.0:
                filtered_results = [row for row in filtered_results if row['similarity'] >= threshold]
            
            logger.info(f"Found {len(filtered_results)} results above threshold {threshold}")
            cursor.close()
            return filtered_results
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored embeddings."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM {self.config.EMBEDDINGS_TABLE}")
            total_count = cursor.fetchone()[0]
            
            # Get unique file names
            cursor.execute(f"SELECT DISTINCT file_name FROM {self.config.EMBEDDINGS_TABLE}")
            file_names = [row[0] for row in cursor.fetchall()]
            

            
            # Get split strategies
            cursor.execute(f"SELECT DISTINCT split_strategy FROM {self.config.EMBEDDINGS_TABLE}")
            split_strategies = [row[0] for row in cursor.fetchall()]
            
            # Get database info
            db_info = self.config.get_database_info()
            
            cursor.close()
            
            return {
                'total_embeddings': total_count,
                'unique_files': len(file_names),
                'file_names': file_names,
                'split_strategies': split_strategies,
                'database_info': db_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def delete_file_embeddings(self, file_name: str) -> bool:
        """
        Delete all embeddings for a specific file.
        
        Args:
            file_name: Name of the file to delete embeddings for
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"DELETE FROM {self.config.EMBEDDINGS_TABLE} WHERE file_name = %s", (file_name,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            cursor.close()
            
            logger.info(f"Deleted {deleted_count} embeddings for file: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for {file_name}: {e}")
            if conn:
                conn.rollback()
            return False
    
    def get_file_embeddings(self, file_name: str) -> List[Dict[str, Any]]:
        """
        Get all embeddings for a specific file.
        
        Args:
            file_name: Name of the file to get embeddings for
            
        Returns:
            List of embeddings with metadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(f"""
                SELECT * FROM {self.config.EMBEDDINGS_TABLE}
                WHERE file_name = %s
                ORDER BY chunk_index
            """, (file_name,))
            
            results = cursor.fetchall()
            cursor.close()
            
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Failed to get embeddings for {file_name}: {e}")
            return []
    
    def get_file_embeddings_count(self, file_name: str) -> int:
        """
        Get the count of embeddings for a specific file.
        
        Args:
            file_name: Name of the file to count embeddings for
            
        Returns:
            Number of embeddings for the file
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT COUNT(*) FROM {self.config.EMBEDDINGS_TABLE}
                WHERE file_name = %s
            """, (file_name,))
            
            count = cursor.fetchone()[0]
            cursor.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to get embeddings count for {file_name}: {e}")
            return 0
    
    def close_connection(self):
        """Close the database connection."""
        if self.connection and not self.connection.closed:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()
