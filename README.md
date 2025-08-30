# RAG Python Module with Gemini API and PostgreSQL

A Python module for implementing Retrieval-Augmented Generation (RAG) that allows you to index PDF and DOCX documents and search through them using semantic similarity. **Now powered by Google Gemini API for high-quality embeddings and PostgreSQL with pgvector for scalable vector storage!**

> **🎯 Recommended Usage**: Use `python main.py --interactive` for the best user experience with guided menus and chunking strategy selection!

## Features

- **Document Processing**: Extract text from PDF and DOCX files
- **Text Chunking**: Intelligent text chunking with configurable overlap
- **Vector Storage**: Store document embeddings in PostgreSQL using pgvector
- **Semantic Search**: Find relevant documents using Google Gemini embeddings
- **Flexible Search**: Search across all documents or within specific sources
- **Similarity Scoring**: Rank results by semantic similarity
- **Gemini Integration**: Uses Google's state-of-the-art embedding models
- **PostgreSQL Backend**: Enterprise-grade database with vector similarity search
- **Configuration Management**: Comprehensive .env file support with validation

## Prerequisites

- **Google Gemini API Key**: Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **PostgreSQL Database**: Version 12+ with pgvector extension
- **Python 3.8+**: Required for the latest features

## Installation

1. Clone or download this module to your project directory
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up PostgreSQL with pgvector extension (see setup instructions below)
4. Set up your configuration:

```bash
# Copy the example configuration file
cp env.example .env

# Edit the .env file with your settings
nano .env
```

## PostgreSQL Setup

### 1. Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```

**macOS:**
```bash
brew install postgresql
```

**Windows:** Download from [PostgreSQL official site](https://www.postgresql.org/download/windows/)

### 2. Install pgvector Extension

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql-14-pgvector
# or for other versions: sudo apt-get install postgresql-{version}-pgvector
```

**macOS:**
```bash
brew install pgvector
```

**From Source:**
```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 3. Create Database and User

```sql
-- Connect to PostgreSQL as superuser
sudo -u postgres psql

-- Create database
CREATE DATABASE rag_database;

-- Create user
CREATE USER rag_user WITH PASSWORD 'your_secure_password';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE rag_database TO rag_user;

-- Connect to the database
\c rag_database

-- Enable pgvector extension
CREATE EXTENSION vector;

-- Exit
\q
```

## Configuration

The module uses a comprehensive configuration system with the following options:

### Quick Setup

1. **Copy the example file**: `cp env.example .env`
2. **Edit .env file**: Add your Gemini API key and PostgreSQL credentials
3. **Required settings**: 
   - `GOOGLE_API_KEY`: Your Gemini API key
   - `POSTGRES_PASSWORD`: Your PostgreSQL password

### Configuration Options

#### Required Settings
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `POSTGRES_PASSWORD`: PostgreSQL password

#### Database Configuration
- `POSTGRES_HOST`: Database host (default: localhost)
- `POSTGRES_PORT`: Database port (default: 5432)
- `POSTGRES_DB`: Database name (default: rag_database)
- `POSTGRES_USER`: Database username (default: rag_user)
- `EMBEDDINGS_TABLE`: Table name for embeddings (default: document_embeddings)
- `VECTOR_DIMENSION`: Embedding vector dimension (default: 768)

#### Text Processing
- `DEFAULT_CHUNK_SIZE`: Text chunk size in characters (default: 1000)
- `DEFAULT_CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `SPLIT_STRATEGY`: Text splitting strategy (default: sentence_boundary)

#### Search Configuration
- `DEFAULT_SEARCH_RESULTS`: Number of results to return (default: 5)
- `DEFAULT_SIMILARITY_THRESHOLD`: Similarity threshold 0.0-1.0 (default: 0.5)

## Quick Start

### 1. Index a Document

```python
from index_documents import index_document

# Uses configuration from .env file
success = index_document("path/to/document.pdf")
if success:
    print("Document indexed successfully!")
```

### 2. Search Documents

```python
from search_documents import search_documents

# Uses configuration from .env file
results = search_documents("What is machine learning?", n_results=5)
for result in results:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
    print("---")
```

### 3. Check Configuration

```python
from config import get_config

config = get_config()
print(f"Database: {config.POSTGRES_DB}")
print(f"Table: {config.EMBEDDINGS_TABLE}")
config.print_config()  # Show all settings
```

## Command Line Usage

### Indexing Documents

```bash
# Index a PDF file (uses .env configuration)
python index_documents.py document.pdf

# Show current configuration
python index_documents.py --show-config

# Override configuration
python index_documents.py document.pdf --chunk-size 800 --overlap 150

# Delete embeddings for a file
python index_documents.py --delete "document.pdf"
```

### Searching Documents

```bash
# Basic search (uses .env configuration)
python search_documents.py "What is artificial intelligence?"

# Show configuration and statistics
python search_documents.py "test query" --show-config --show-stats

# Override configuration
python search_documents.py "query" --n-results 10 --threshold 0.7

# Search within specific source
python search_documents.py "neural networks" --source "document.pdf"
```

## Main CLI Application (`main.py`)

The `main.py` script is now the **primary entry point** for all RAG operations, providing a unified interface for indexing, searching, and managing documents.

### Quick Start

```bash
# Check if your system is ready
python main.py --check-system

# Index a document
python main.py document.pdf

# Search documents
python main.py --search "your query here"

# List all indexed files
python main.py --list-files

# Show system information
python main.py --info

# Interactive mode
python main.py --interactive
```

### All Available Commands

```bash
# System operations
python main.py --check-system    # Check system requirements
python main.py --info            # Show system configuration
python main.py --verbose         # Enable detailed logging

# Document management
python main.py document.pdf      # Index a document
python main.py --force-update document.pdf  # Force re-indexing
python main.py --list-files      # List indexed documents
python main.py --delete "file.pdf"  # Delete a file

# Search operations
python main.py --search "query"  # Search documents

# Chunking strategies
python main.py --chunk-strategy sentence document.pdf
python main.py --chunk-strategy paragraph document.pdf
python main.py --chunk-strategy fixed-size --chunk-size 800 --overlap 100 document.pdf
```

### Interactive Mode

For the best user experience, use interactive mode:

```bash
python main.py --interactive
```

This provides a guided interface where you can:
1. **Choose chunking strategies** from a menu
2. **Set parameters** interactively
3. **Index documents** with your chosen strategy
4. **Search content** easily
5. **Manage files** through simple menus

### Chunking Strategy Guide

#### **Fixed-Size Chunks** (Default)
- **Best for**: General purpose, consistent chunk sizes
- **Use when**: You want predictable chunk sizes and need overlap for context
- **Benefits**: Consistent embedding generation, good for similarity search
- **Example**: `python main.py --chunk-strategy fixed-size --chunk-size 1000 --overlap 200 document.pdf`

#### **Sentence-Based Chunks**
- **Best for**: Semantic search and question answering
- **Use when**: You want to preserve complete thoughts and semantic meaning
- **Benefits**: Better semantic coherence, natural language boundaries
- **Note**: No size limits - each sentence becomes a natural chunk
- **Example**: `python main.py --chunk-strategy sentence document.pdf`

#### **Paragraph-Based Chunks**
- **Best for**: Document structure preservation
- **Use when**: You want to maintain logical document sections
- **Benefits**: Preserves document organization, good for topic-based retrieval
- **Note**: No size limits - each paragraph becomes a natural chunk
- **Example**: `python main.py --chunk-strategy paragraph document.pdf`

#### **Strategy Selection Tips**
- **Start with**: Fixed-size chunks (default) for general use
- **For Q&A**: Use sentence-based for better semantic understanding
- **For documents**: Use paragraph-based to preserve structure
- **Customize**: Adjust chunk size based on your content and use case

## API Reference

### Configuration Management

```python
from config import Config, get_config, load_config

# Get configuration instance
config = get_config()

# Load and validate configuration
config = load_config()  # Raises ValueError if invalid

# Access settings
api_key = config.GOOGLE_API_KEY
db_host = config.POSTGRES_HOST

# Print configuration
config.print_config(include_sensitive=False)
```

### Database Management

```python
from database import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager()

# Get collection statistics
stats = db_manager.get_collection_stats()

# Search embeddings
results = db_manager.search_similar_embeddings(
    query_embedding=embedding_vector,
    n_results=5,
    threshold=0.5
)

# Delete file embeddings
db_manager.delete_file_embeddings("document.pdf")
```

### DocumentIndexer Class

The main class for indexing documents using Gemini embeddings and PostgreSQL.

#### Methods

- `__init__(api_key=None, model_name=None)`
  - Initialize the indexer (uses config defaults if parameters are None)
  
- `index_document(file_path, chunk_size=None, overlap=None)`
  - Index a document file (PDF or DOCX)
  - Returns `True` if successful, `False` otherwise
  
- `generate_embedding(text)`
  - Generate embedding for text using Gemini API
  
- `get_collection_info()`
  - Get information about the indexed documents
  
- `delete_document(file_name)`
  - Delete all embeddings for a specific document

### DocumentSearcher Class

The main class for searching documents using Gemini embeddings and PostgreSQL.

#### Methods

- `__init__(api_key=None, model_name=None)`
  - Initialize the searcher (uses config defaults if parameters are None)
  
- `search_documents(query, n_results=None, threshold=None)`
  - Search for documents relevant to the query
  - Returns list of relevant document chunks
  
- `search_by_source(query, source_file, n_results=None)`
  - Search within a specific source file
  
- `get_similar_documents(document_chunk, n_results=None)`
  - Find documents similar to a given chunk
  
- `get_collection_stats()`
  - Get statistics about the indexed documents
  
- `format_search_results(results, show_metadata=True)`
  - Format search results for display

## Database Schema

The module creates a table with the following structure:

```sql
CREATE TABLE document_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    split_strategy VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_index INTEGER,
    total_chunks INTEGER,

    metadata JSONB
);
```

**Columns:**
- `id`: Unique identifier for each embedding
- `chunk_text`: The text content from the chunk
- `embedding`: The vector representation (using pgvector)
- `file_name`: Original file name
- `split_strategy`: Text splitting strategy used
- `created_at`: Timestamp when the embedding was added
- `chunk_index`: Position of chunk in the document
- `total_chunks`: Total number of chunks in the document

- `metadata`: Additional information in JSON format

## Example Workflow

```python
# 1. Load configuration
from config import load_config
config = load_config()

# 2. Index multiple documents
from index_documents import DocumentIndexer

indexer = DocumentIndexer()  # Uses config defaults
indexer.index_document("research_paper.pdf")
indexer.index_document("technical_spec.docx")

# 3. Search through all documents
from search_documents import DocumentSearcher

searcher = DocumentSearcher()  # Uses config defaults
results = searcher.search_documents("What are the main findings?")

# 4. Get detailed results
for result in results:
    print(f"Source: {result['metadata']['source']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
    print(f"Content: {result['content']}")
    print("---")

# 5. Search within specific document
paper_results = searcher.search_by_source("methodology", "research_paper.pdf")

# 6. Get collection statistics
stats = searcher.get_collection_stats()
print(f"Total embeddings: {stats['total_chunks']}")
```

## File Structure

```
ragPythonModule/
├── main.py                  # 🆕 MAIN CLI APPLICATION (recommended entry point)
├── __init__.py              # Module initialization
├── index_documents.py       # Document indexing with Gemini + PostgreSQL
├── search_documents.py      # Document search with Gemini + PostgreSQL
├── config.py               # Configuration management
├── database.py             # PostgreSQL database operations
├── requirements.txt         # Python dependencies
├── README.md               # This comprehensive documentation
├── env_template.env        # Configuration template
├── example_usage.py        # Working example with config
└── .env                    # Your configuration (create from env_template.env)
```

## Dependencies

- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX text extraction
- **psycopg2-binary**: PostgreSQL adapter
- **pgvector**: Vector operations for PostgreSQL
- **google-generativeai**: Google Gemini API client
- **python-dotenv**: Environment variable management
- **numpy**: Numerical operations
- **tiktoken**: Text tokenization for chunking

## Troubleshooting

### Common Issues

1. **"Google Gemini API key is required" error**
   - Set `GOOGLE_API_KEY` in your `.env` file
   - Or set the environment variable: `export GOOGLE_API_KEY='your_key'`
   - Get your API key from: https://makersuite.google.com/app/apikey

2. **PostgreSQL connection errors**
   - Verify PostgreSQL is running: `sudo systemctl status postgresql`
   - Check credentials in `.env` file
   - Ensure pgvector extension is installed: `CREATE EXTENSION vector;`
   - Verify database and user exist with proper permissions

3. **Configuration validation errors**
   - Check your `.env` file format (no spaces around `=`)
   - Ensure numeric values are valid (integers/floats)
   - Boolean values should be: `true`, `false`, `1`, `0`, `yes`, `no`
   - Verify `VECTOR_DIMENSION` matches your Gemini model output

4. **"Collection not found" error**
   - Make sure you've indexed at least one document before searching
   - Check that the database table exists and is accessible
   - Verify database connection settings

5. **Memory issues with large documents**
   - Reduce `DEFAULT_CHUNK_SIZE` in your `.env` file
   - Increase `DEFAULT_CHUNK_OVERLAP` for better context preservation
   - Adjust `EMBEDDING_BATCH_SIZE` for your system

6. **API rate limiting**
   - Reduce `EMBEDDING_BATCH_SIZE` in your `.env` file
   - Increase `API_TIMEOUT` for slower connections
   - Monitor your Gemini API usage

7. **Vector dimension mismatch**
   - Check the actual output dimension of your Gemini model
   - Update `VECTOR_DIMENSION` in `.env` to match
   - Recreate the database table if needed

### Performance Tips

- **Debug mode**: Set `DEBUG_MODE=true` in `.env` for detailed logging
- **Test mode**: Set `TEST_MODE=true` for smaller chunks and fewer results
- **Batch processing**: Adjust `EMBEDDING_BATCH_SIZE` based on your API limits
- **Database optimization**: Use appropriate indexes and connection pooling
- **Vector similarity**: pgvector provides efficient similarity search

### Configuration Validation

The module validates your configuration on startup:

```python
from config import load_config

try:
    config = load_config()
    print("Configuration is valid!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Security Notes

- **Never commit .env files to version control**
- Use environment variables for sensitive data in production
- The module logs API calls when `LOG_API_CALLS=true`
- Consider implementing API key rotation for production use
- Validate configuration before deployment
- Use strong PostgreSQL passwords and restrict network access

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this module.

## License

This module is provided as-is for educational and development purposes.
