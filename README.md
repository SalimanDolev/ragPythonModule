# RAG Python Module with Gemini API

A Python module for implementing Retrieval-Augmented Generation (RAG) that allows you to index PDF and DOCX documents and search through them using semantic similarity. **Now powered by Google Gemini API for high-quality embeddings!**

## Features

- **Document Processing**: Extract text from PDF and DOCX files
- **Text Chunking**: Intelligent text chunking with configurable overlap
- **Vector Storage**: Store document embeddings in ChromaDB for fast retrieval
- **Semantic Search**: Find relevant documents using Google Gemini embeddings
- **Flexible Search**: Search across all documents or within specific sources
- **Similarity Scoring**: Rank results by semantic similarity
- **Gemini Integration**: Uses Google's state-of-the-art embedding models

## Prerequisites

- **Google Gemini API Key**: Get your free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Python 3.8+**: Required for the latest features

## Installation

1. Clone or download this module to your project directory
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Gemini API key:

```bash
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

Or set it in your Python code:
```python
import os
os.environ['GOOGLE_API_KEY'] = 'your_gemini_api_key_here'
```

## Quick Start

### 1. Index a Document

```python
from index_documents import index_document

# Index a PDF file (API key from environment variable)
success = index_document("path/to/document.pdf")
if success:
    print("Document indexed successfully!")

# Or pass API key directly
success = index_document("path/to/document.pdf", api_key="your_api_key")
```

### 2. Search Documents

```python
from search_documents import search_documents

# Search for relevant content (API key from environment variable)
results = search_documents("What is machine learning?", n_results=5)
for result in results:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Content: {result['content'][:200]}...")
    print("---")
```

## Command Line Usage

### Indexing Documents

```bash
# Index a PDF file (API key from environment variable)
python index_documents.py document.pdf

# Index with custom parameters
python index_documents.py document.pdf --chunk-size 800 --overlap 150

# Specify custom database path and API key
python index_documents.py document.pdf --db-path ./my_database --api-key YOUR_API_KEY
```

### Searching Documents

```bash
# Basic search (API key from environment variable)
python search_documents.py "What is artificial intelligence?"

# Search with custom parameters
python search_documents.py "machine learning algorithms" --n-results 10 --threshold 0.7

# Search within a specific source
python search_documents.py "neural networks" --source "document.pdf"

# Show collection statistics
python search_documents.py "test query" --show-stats

# Pass API key directly
python search_documents.py "your query" --api-key YOUR_API_KEY
```

## API Reference

### DocumentIndexer Class

The main class for indexing documents using Gemini embeddings.

#### Methods

- `__init__(db_path="./chroma_db", api_key=None, model_name="models/embedding-001")`
  - Initialize the indexer with database path and Gemini API key
  
- `index_document(file_path, chunk_size=1000, overlap=200)`
  - Index a document file (PDF or DOCX)
  - Returns `True` if successful, `False` otherwise
  
- `generate_embedding(text)`
  - Generate embedding for text using Gemini API
  
- `get_collection_info()`
  - Get information about the indexed documents

### DocumentSearcher Class

The main class for searching documents using Gemini embeddings.

#### Methods

- `__init__(db_path="./chroma_db", api_key=None, model_name="models/embedding-001")`
  - Initialize the searcher with database path and Gemini API key
  
- `search_documents(query, n_results=5, threshold=0.5)`
  - Search for documents relevant to the query
  - Returns list of relevant document chunks
  
- `search_by_source(query, source_file, n_results=5)`
  - Search within a specific source file
  
- `get_similar_documents(document_chunk, n_results=5)`
  - Find documents similar to a given chunk
  
- `get_collection_stats()`
  - Get statistics about the indexed documents
  
- `format_search_results(results, show_metadata=True)`
  - Format search results for display

## Configuration

### Gemini API Settings

- **API Key**: Required for all operations
- **Embedding Model**: `models/embedding-001` (default, high-quality embeddings)
- **Rate Limits**: Follow Google's API usage guidelines

### Database Settings

- **ChromaDB**: Persistent vector database for storing embeddings
- **Default path**: `./chroma_db` (relative to current directory)
- **Collection name**: `documents`

### Text Chunking

- **Default chunk size**: 1000 characters
- **Default overlap**: 200 characters
- **Smart chunking**: Breaks at sentence boundaries when possible

## Example Workflow

```python
# 1. Set your API key
import os
os.environ['GOOGLE_API_KEY'] = 'your_api_key_here'

# 2. Index multiple documents
from index_documents import DocumentIndexer

indexer = DocumentIndexer("./my_docs_db")
indexer.index_document("research_paper.pdf")
indexer.index_document("technical_spec.docx")

# 3. Search through all documents
from search_documents import DocumentSearcher

searcher = DocumentSearcher("./my_docs_db")
results = searcher.search_documents("What are the main findings?")

# 4. Get detailed results
for result in results:
    print(f"Source: {result['metadata']['source']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
    print(f"Content: {result['content']}")
    print("---")

# 5. Search within specific document
paper_results = searcher.search_by_source("methodology", "research_paper.pdf")
```

## File Structure

```
ragPythonModule/
├── __init__.py              # Module initialization
├── index_documents.py       # Document indexing with Gemini
├── search_documents.py      # Document search with Gemini
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── example_usage.py        # Working example with Gemini
├── test_module.py          # Test suite
└── chroma_db/              # Vector database (created automatically)
```

## Dependencies

- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX text extraction
- **chromadb**: Vector database for embeddings
- **google-generativeai**: Google Gemini API client
- **numpy**: Numerical operations
- **langchain**: Optional integration with LangChain ecosystem

## Troubleshooting

### Common Issues

1. **"Google Gemini API key is required" error**
   - Set the `GOOGLE_API_KEY` environment variable
   - Or pass the API key directly to the functions
   - Get your API key from: https://makersuite.google.com/app/apikey

2. **"Collection not found" error**
   - Make sure you've indexed at least one document before searching
   - Check that the database path is correct

3. **Memory issues with large documents**
   - Reduce chunk size: `--chunk-size 500`
   - Increase overlap: `--overlap 100`

4. **API rate limiting**
   - Gemini API has rate limits for free tier
   - Consider implementing retry logic for production use

5. **PDF text extraction issues**
   - Some PDFs may have embedded images or scanned text
   - Consider using OCR tools for such documents

### Performance Tips

- Gemini embeddings are high-quality but may be slower than local models
- Use appropriate chunk sizes for your document characteristics
- Consider caching embeddings for frequently accessed documents
- Monitor API usage to stay within rate limits

## Security Notes

- **Never commit API keys to version control**
- Use environment variables or secure secret management
- The module logs API calls for debugging purposes
- Consider implementing API key rotation for production use

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this module.

## License

This module is provided as-is for educational and development purposes.
