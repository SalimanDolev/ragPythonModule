# RAG Python Module with Gemini API and PostgreSQL

A powerful **Retrieval Augmented Generation (RAG)** system that combines Google's Gemini AI with PostgreSQL and pgvector for efficient document indexing and semantic search.

## 🚀 **Installation**

### 1. **Clone and Install**
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
```bash
python main.py --check-system
```

## 📖 **Usage**

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

## 💡 **Examples**

### **Basic Workflow**
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
