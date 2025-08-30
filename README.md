# RAG Python Module with Gemini API and PostgreSQL

<<<<<<< Updated upstream
A **Retrieval Augmented Generation (RAG)** system that combines Google's Gemini AI with PostgreSQL for efficient document indexing and semantic search.

## 🚀 **Installation**

### 1. **Clone and Install Dependencies**
=======
A powerful **Retrieval Augmented Generation (RAG)** system that combines Google's Gemini AI with PostgreSQL and pgvector for efficient document indexing and semantic search.

## 🚀 **Installation**

### 1. **Clone and Install**
>>>>>>> Stashed changes
```bash
git clone <your-repo>
cd ragPythonModule
pip install -r requirements.txt
```

### 2. **Environment Setup**
```bash
# Copy template and edit with your credentials
cp env_template.env .env
nano .env
```

**Required in `.env`:**
```bash
<<<<<<< Updated upstream
GEMINI_API_KEY=your_actual_api_key_here
POSTGRES_URL=postgresql://username:password@localhost:5432/database_name
```

### 3. **Database Setup**
=======
GOOGLE_API_KEY=your_actual_api_key_here
POSTGRES_HOST=localhost
POSTGRES_DB=your_database_name
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_secure_password
```

### 3. **Database Setup**
```sql
CREATE DATABASE your_database_name;
CREATE EXTENSION IF NOT EXISTS vector;
```

### 4. **Verify Installation**
>>>>>>> Stashed changes
```bash
# Ensure PostgreSQL is running and create a database
# The pgvector extension will be automatically installed when needed
```

## 📖 **Usage**

<<<<<<< Updated upstream
### **Core Scripts**

This module provides two main scripts:

- **`index_documents.py`** - Document indexing with multiple chunking strategies
- **`search_documents.py`** - Semantic search using Gemini embeddings

### **Command Line Usage**

#### **Index Documents**
```bash
# Basic indexing with default strategy (fixed-size)
python index_documents.py document.pdf

# Use sentence-based chunking
python index_documents.py document.pdf --chunk-strategy sentence

# Use paragraph-based chunking
python index_documents.py document.pdf --chunk-strategy paragraph

# Fixed-size with custom chunk size and overlap
python index_documents.py document.pdf --chunk-strategy fixed-size --chunk-size 800 --overlap 150
```

#### **Search Documents**
```bash
# Search documents (always returns 5 most relevant results)
python search_documents.py "your query"
```

### **Chunking Strategies**

- **Fixed-size**: Configurable chunk size with overlap (default: 1000 chars, 200 overlap)
- **Sentence-based**: Natural sentence boundaries (no size limits)
- **Paragraph-based**: Natural paragraph boundaries (no size limits)
=======
### **Interactive Mode (Recommended)**
```bash
python main.py --interactive
```

### **Command Line**
```bash
# Index documents
python main.py --index document.pdf --chunk-strategy sentence

# Search documents
python main.py --search "your query here"

# List files
python main.py --list-files

# Delete file
python main.py --delete-file filename.pdf
```

### **Chunking Strategies**
- **Fixed-size**: `--chunk-strategy fixed-size --chunk-size 1000 --overlap 200`
- **Sentence-based**: `--chunk-strategy sentence` (no size limits)
- **Paragraph-based**: `--chunk-strategy paragraph` (no size limits)
>>>>>>> Stashed changes

## 💡 **Examples**

### **Basic Workflow**
<<<<<<< Updated upstream
```bash
# 1. Index a PDF document
python index_documents.py research_paper.pdf --chunk-strategy sentence

# 2. Search the indexed content
python search_documents.py "What are the main findings?"
```

### **Different Chunking Strategies**
```bash
# Fixed-size chunks (good for consistent retrieval)
python index_documents.py long_document.pdf --chunk-strategy fixed-size --chunk-size 500 --overlap 100

# Sentence-based chunks (good for natural language understanding)
python index_documents.py article.pdf --chunk-strategy sentence

# Paragraph-based chunks (good for structured content)
python index_documents.py report.pdf --chunk-strategy paragraph
```

### **Search Examples**
```bash
# Find information about specific topics
python search_documents.py "machine learning algorithms"

# Search for specific topics
python search_documents.py "artificial intelligence"

# Search for technical concepts
python search_documents.py "neural networks and deep learning"
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key
POSTGRES_URL=postgresql://username:password@localhost:5432/database_name
```

### **Default Settings**
- **Chunk Strategy**: `fixed-size`
- **Chunk Size**: `1000` characters
- **Overlap**: `200` characters
- **Search Results**: Always `5` most relevant chunks

## 📋 **Requirements**

- Python 3.8+
- PostgreSQL with pgvector extension
- Google Gemini API key
- PDF and DOCX file support

---

**Need Help?** Run any script with `--help` flag for detailed usage information!
=======
```python
from index_documents import DocumentIndexer
from search_documents import DocumentSearcher

# Index a document
indexer = DocumentIndexer()
indexer.index_document("document.pdf", chunk_strategy="sentence")

# Search documents
searcher = DocumentSearcher()
results = searcher.search_documents("What is machine learning?")
```

### **Programmatic Usage**
```python
# Index multiple files
files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
for file in files:
    indexer.index_document(file, chunk_strategy="paragraph")

# Search with custom parameters
results = searcher.search_documents(
    "artificial intelligence", 
    n_results=10, 
    threshold=0.7
)
```

### **Get Collection Stats**
```python
stats = searcher.get_collection_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Unique files: {stats['unique_sources']}")
```

---

**Need Help?** Run `python main.py --interactive` for guided assistance!
>>>>>>> Stashed changes
