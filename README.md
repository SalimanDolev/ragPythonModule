# RAG Python Module with Gemini API and PostgreSQL

A **Retrieval Augmented Generation (RAG)** system that combines Google's Gemini AI with PostgreSQL for efficient document indexing and semantic search.

## 🚀 **Installation**

### 1. **Clone and Install Dependencies**
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
GEMINI_API_KEY=your_actual_api_key_here
POSTGRES_URL=postgresql://username:password@localhost:5432/database_name
```

### 3. **Database Setup**
```bash
# Ensure PostgreSQL is running and create a database
# The pgvector extension will be automatically installed when needed
```

## 📖 **Usage**

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

## 💡 **Examples**

### **Basic Workflow**
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
