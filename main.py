
#!/usr/bin/env python3
"""
RAG Python Module - Main CLI Application

A comprehensive command-line interface for the RAG system with Gemini API and PostgreSQL.
Handles all requirements checking, system validation, and provides easy access to all features.

Usage:
    python test_module.py <file_path> [options]
    python test_module.py --check-system
    python test_module.py --search "query"
    python test_module.py --list-files
    python test_module.py --delete <file_name>
    python test_module.py --force-update <file_path>
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGSystemManager:
    """Main manager class for the RAG system operations."""
    
    def __init__(self):
        """Initialize the RAG system manager."""
        self.config = None
        self.indexer = None
        self.searcher = None
        
    def check_system_requirements(self) -> bool:
        """Check all system requirements and dependencies."""
        print("🔍 Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("  ❌ Python 3.8+ required (current: {})".format(sys.version))
            return False
        print("  ✅ Python version: {}".format(sys.version.split()[0]))
        
        # Check required modules
        required_modules = [
            'PyPDF2', 'docx', 'google.generativeai', 'psycopg2', 
            'pgvector', 'numpy', 'tiktoken', 'dotenv'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                if module == 'python-dotenv':
                    __import__('dotenv')
                    print(f"  ✅ {module}")
                else:
                    __import__(module)
                    print(f"  ✅ {module}")
            except ImportError:
                missing_modules.append(module)
                print(f"  ❌ {module} - MISSING")
        
        if missing_modules:
            print(f"\n❌ Missing modules: {', '.join(missing_modules)}")
            print("💡 Install missing modules with: pip install -r requirements.txt")
            return False
        
        # Check environment variables
        print("\n🔑 Checking environment configuration...")
        env_vars = ['GOOGLE_API_KEY', 'POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
        missing_env = []
        
        for var in env_vars:
            if os.getenv(var):
                if var == 'GOOGLE_API_KEY':
                    print(f"  ✅ {var}: {'*' * 10}")  # Hide API key
                else:
                    print(f"  ✅ {var}: {os.getenv(var)}")
            else:
                missing_env.append(var)
                print(f"  ❌ {var} - NOT SET")
        
        if missing_env:
            print(f"\n❌ Missing environment variables: {', '.join(missing_env)}")
            print("💡 Create a .env file or set environment variables")
            return False
        
        # Check PostgreSQL connection
        print("\n🗄️  Checking PostgreSQL connection...")
        try:
            from database import DatabaseManager
            db_manager = DatabaseManager()
            conn = db_manager._get_connection()
            if conn and not conn.closed:
                print("  ✅ PostgreSQL connection successful")
                conn.close()
            else:
                print("  ❌ PostgreSQL connection failed")
                return False
        except Exception as e:
            print(f"  ❌ PostgreSQL connection error: {e}")
            return False
        
        # Check Gemini API
        print("\n🤖 Checking Gemini API...")
        try:
            import google.generativeai as genai
            api_key = os.getenv('GOOGLE_API_KEY')
            genai.configure(api_key=api_key)
            
            # Test with a simple embedding
            result = genai.embed_content(
                model='models/embedding-001',
                content='test'
            )
            if 'embedding' in result:
                print("  ✅ Gemini API connection successful")
            else:
                print("  ❌ Gemini API response format error")
                return False
        except Exception as e:
            print(f"  ❌ Gemini API error: {e}")
            return False
        
        print("\n🎉 All system requirements met!")
        return True
    
    def initialize_system(self) -> bool:
        """Initialize the RAG system components."""
        try:
            from config import get_config
            from index_documents import DocumentIndexer
            from search_documents import DocumentSearcher
            
            self.config = get_config()
            self.indexer = DocumentIndexer()
            self.searcher = DocumentSearcher()
            
            print("✅ RAG system initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            return False
    
    def index_document(self, file_path: str, force_update: bool = False, 
                      chunk_strategy: str = None, chunk_size: int = None, overlap: int = None) -> bool:
        """Index a document using the RAG system."""
        if not self.indexer:
            print("❌ RAG system not initialized")
            return False
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return False
            
            print(f"📄 Indexing document: {file_path.name}")
            
            # Show chunking strategy if specified
            if chunk_strategy:
                print(f"   📏 Using chunk strategy: {chunk_strategy}")
                if chunk_size:
                    print(f"   📐 Chunk size: {chunk_size}")
                if overlap:
                    print(f"   🔄 Overlap: {overlap}")
            
            success = self.indexer.index_document(
                str(file_path), 
                force_update=force_update,
                chunk_strategy=chunk_strategy,
                chunk_size=chunk_size,
                overlap=overlap
            )
            
            if success:
                info = self.indexer.get_collection_info()
                print(f"✅ Document indexed successfully")
                print(f"   📊 Total chunks: {info['total_documents']}")
                print(f"   🗄️  Database: {info['database']}")
                return True
            else:
                print(f"❌ Failed to index document")
                return False
                
        except Exception as e:
            print(f"❌ Error indexing document: {e}")
            return False
    
    def search_documents(self, query: str, n_results: int = 5) -> bool:
        """Search documents using the RAG system."""
        if not self.searcher:
            print("❌ RAG system not initialized")
            return False
        
        try:
            print(f"🔍 Searching for: '{query}'")
            results = self.searcher.search_documents(query, n_results=n_results)
            
            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    score = result['similarity_score']
                    content = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                    print(f"   {i}. Score: {score:.3f}")
                    print(f"      File: {result['metadata']['source']}")
                    print(f"      Content: {content}")
                    print()
            else:
                print("⚠️  No results found")
            
            return True
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return False
    
    def list_files(self) -> bool:
        """List all indexed files in the system."""
        if not self.searcher:
            print("❌ RAG system not initialized")
            return False
        
        try:
            stats = self.searcher.get_collection_stats()
            print("📚 Indexed Files:")
            print(f"   📊 Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   📁 Unique files: {stats.get('unique_sources', 0)}")
            
            # Get file names directly from database
            try:
                from database import DatabaseManager
                db_manager = DatabaseManager()
                conn = db_manager._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT file_name FROM document_embeddings ORDER BY file_name")
                file_names = [row[0] for row in cursor.fetchall()]
                cursor.close()
                conn.close()
                
                if file_names:
                    print("\n   Files:")
                    for file_name in file_names:
                        print(f"      • {file_name}")
                else:
                    print("   No files indexed yet")
            except Exception as e:
                print(f"   ⚠️  Could not retrieve file names: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return False
    
    def delete_file(self, file_name: str) -> bool:
        """Delete a file and all its embeddings."""
        if not self.indexer:
            print("❌ RAG system not initialized")
            return False
        
        try:
            print(f"🗑️  Deleting file: {file_name}")
            success = self.indexer.delete_document(file_name)
            
            if success:
                print(f"✅ File deleted successfully")
            else:
                print(f"❌ Failed to delete file")
            
            return success
            
        except Exception as e:
            print(f"❌ Error deleting file: {e}")
            return False
    
    def show_system_info(self) -> bool:
        """Show comprehensive system information."""
        if not self.config:
            print("❌ RAG system not initialized")
            return False
        
        try:
            print("ℹ️  System Information:")
            print(f"   🗄️  Database: {self.config.POSTGRES_HOST}:{self.config.POSTGRES_PORT}/{self.config.POSTGRES_DB}")
            print(f"   👤 User: {self.config.POSTGRES_USER}")
            print(f"   🤖 Model: {self.config.GEMINI_EMBEDDING_MODEL}")
            print(f"   📏 Chunk size: {self.config.DEFAULT_CHUNK_SIZE}")
            print(f"   🔄 Overlap: {self.config.DEFAULT_CHUNK_OVERLAP}")
            print(f"   📊 Search results: {self.config.DEFAULT_SEARCH_RESULTS}")
            print(f"   🎯 Similarity threshold: {self.config.DEFAULT_SIMILARITY_THRESHOLD}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error showing system info: {e}")
            return False

def run_interactive_mode(manager: RAGSystemManager):
    """Run the RAG system in interactive mode."""
    print("\n🎯 **RAG Python Module - Interactive Mode**")
    print("=" * 50)
    
    # Check system requirements first
    print("\n🔍 Checking system requirements...")
    if not manager.check_system_requirements():
        print("❌ System requirements not met. Cannot continue in interactive mode.")
        return
    
    print("✅ System requirements met!")
    
    # Initialize the system
    if not manager.initialize_system():
        print("❌ Failed to initialize RAG system.")
        return
    
    print("✅ RAG system initialized successfully!")
    
    while True:
        print("\n" + "=" * 50)
        print("📋 **Main Menu**")
        print("1. 📄 Index a document")
        print("2. 🔍 Search documents")
        print("3. 📚 List indexed files")
        print("4. 🗑️  Delete a file")
        print("5. ℹ️  Show system info")
        print("6. 🚪 Exit")
        print("=" * 50)
        
        choice = input("\n🎯 Choose an option (1-6): ").strip()
        
        if choice == "1":
            handle_index_document(manager)
        elif choice == "2":
            handle_search_documents(manager)
        elif choice == "3":
            manager.list_files()
        elif choice == "4":
            handle_delete_file(manager)
        elif choice == "5":
            manager.show_system_info()
        elif choice == "6":
            print("\n👋 Goodbye! Thanks for using RAG Python Module!")
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1-6.")

def handle_index_document(manager: RAGSystemManager):
    """Handle document indexing in interactive mode."""
    print("\n📄 **Index Document**")
    print("-" * 30)
    
    # Get file path
    file_path = input("📁 Enter the path to your document (PDF/DOCX): ").strip().strip('"')
    
    if not file_path:
        print("❌ No file path provided.")
        return
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Choose chunking strategy
    print("\n📏 **Choose Chunking Strategy**")
    print("1. Fixed-size chunks (with overlap) - Best for general use")
    print("2. Sentence-based splitting - Best for semantic search (natural sentence chunks)")
    print("3. Paragraph-based splitting - Best for document structure (natural paragraph chunks)")
    
    print("\n💡 **How it works:**")
    print("   • Fixed-size: Creates chunks of exact size with overlap")
    print("   • Sentence/Paragraph: Automatically splits at natural boundaries")
    print("   • Sentence/Paragraph: No size limits - completely natural chunks")
    
    strategy_choice = input("🎯 Choose strategy (1-3): ").strip()
    
    chunk_strategy = None
    chunk_size = None
    overlap = None
    
    if strategy_choice == "1":
        chunk_strategy = "fixed-size"
        # Get chunk size and overlap
        chunk_size_input = input("📐 Enter chunk size (default: 1000): ").strip()
        chunk_size = int(chunk_size_input) if chunk_size_input else 1000
        
        overlap_input = input("🔄 Enter overlap size (default: 200): ").strip()
        overlap = int(overlap_input) if overlap_input else 200
        
    elif strategy_choice == "2":
        chunk_strategy = "sentence"
        print("📝 **Note**: Chunks will be automatically sized by sentence boundaries.")
        print("   No size limits - each sentence becomes a natural chunk.")
        chunk_size = None  # No size limit for sentence-based
        
    elif strategy_choice == "3":
        chunk_strategy = "paragraph"
        print("📝 **Note**: Chunks will be automatically sized by paragraph boundaries.")
        print("   No size limits - each paragraph becomes a natural chunk.")
        chunk_size = None  # No size limit for paragraph-based
        
    else:
        print("❌ Invalid choice. Using default strategy.")
        chunk_strategy = "fixed-size"
        chunk_size = 1000
        overlap = 200
    
    # Ask about force update
    force_update = input("🔄 Force re-index? (y/N): ").strip().lower() == 'y'
    
    print(f"\n📄 Indexing document with strategy: {chunk_strategy}")
    if chunk_size:
        print(f"   📐 Chunk size: {chunk_size}")
    if overlap:
        print(f"   🔄 Overlap: {overlap}")
    
    # Index the document
    success = manager.index_document(
        file_path,
        force_update=force_update,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    if success:
        print("✅ Document indexed successfully!")
    else:
        print("❌ Failed to index document.")

def handle_search_documents(manager: RAGSystemManager):
    """Handle document search in interactive mode."""
    print("\n🔍 **Search Documents**")
    print("-" * 30)
    
    # Get search query
    query = input("🔍 Enter your search query: ").strip()
    
    if not query:
        print("❌ No search query provided.")
        return
    
    # Get number of results
    n_results_input = input("📊 Number of results (default: 5): ").strip()
    n_results = int(n_results_input) if n_results_input else 5
    
    print(f"\n🔍 Searching for: '{query}'")
    print(f"📊 Requesting {n_results} results...")
    
    # Perform search
    success = manager.search_documents(query, n_results=n_results)
    
    if not success:
        print("❌ Search failed.")

def handle_delete_file(manager: RAGSystemManager):
    """Handle file deletion in interactive mode."""
    print("\n🗑️  **Delete File**")
    print("-" * 30)
    
    # Get file name
    file_name = input("📁 Enter the name of the file to delete: ").strip()
    
    if not file_name:
        print("❌ No file name provided.")
        return
    
    # Confirm deletion
    confirm = input(f"⚠️  Are you sure you want to delete '{file_name}' and all its embeddings? (y/N): ").strip().lower()
    
    if confirm == 'y':
        success = manager.delete_file(file_name)
        if success:
            print("✅ File deleted successfully!")
        else:
            print("❌ Failed to delete file.")
    else:
        print("❌ Deletion cancelled.")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Python Module - Main CLI Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a document
  python main.py document.pdf
  
  # Force re-index a document
  python main.py --force-update document.pdf
  
  # Index with custom chunking strategy
  python main.py --chunk-strategy sentence document.pdf
  python main.py --chunk-strategy paragraph --chunk-size 500 document.pdf
  python main.py --chunk-strategy fixed-size --chunk-size 800 --overlap 100 document.pdf
  
  # Search documents
  python main.py --search "artificial intelligence"
  
  # Check system requirements
  python main.py --check-system
  
  # List indexed files
  python main.py --list-files
  
  # Delete a file
  python main.py --delete document.pdf
  
  # Show system info
  python main.py --info
  
  # Interactive mode
  python main.py --interactive
        """
    )
    
    parser.add_argument("file_path", nargs='?', help="Path to document file to index")
    parser.add_argument("--check-system", action="store_true", help="Check system requirements and dependencies")
    parser.add_argument("--search", metavar="QUERY", help="Search documents with query")
    parser.add_argument("--list-files", action="store_true", help="List all indexed files")
    parser.add_argument("--delete", metavar="FILE", help="Delete file and all its embeddings")
    parser.add_argument("--force-update", action="store_true", help="Force re-indexing of existing file")
    parser.add_argument("--chunk-strategy", choices=['fixed-size', 'sentence', 'paragraph'], 
                       help="Text chunking strategy: fixed-size (with overlaps), sentence-based, or paragraph-based")
    parser.add_argument("--chunk-size", type=int, help="Maximum chunk size (uses config default if not specified)")
    parser.add_argument("--overlap", type=int, help="Overlap between chunks (uses config default if not specified)")
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize system manager
    manager = RAGSystemManager()
    
    try:
        # Handle different operations
        if args.check_system:
            # System check mode
            success = manager.check_system_requirements()
            sys.exit(0 if success else 1)
        
        elif args.search:
            # Search mode
            if not manager.check_system_requirements():
                print("❌ System requirements not met. Run with --check-system for details.")
                sys.exit(1)
            
            if not manager.initialize_system():
                sys.exit(1)
            
            manager.search_documents(args.search)
        
        elif args.list_files:
            # List files mode
            if not manager.check_system_requirements():
                print("❌ System requirements not met. Run with --check-system for details.")
                sys.exit(1)
            
            if not manager.initialize_system():
                sys.exit(1)
            
            manager.list_files()
        
        elif args.delete:
            # Delete mode
            if not manager.check_system_requirements():
                print("❌ System requirements not met. Run with --check-system for details.")
                sys.exit(1)
            
            if not manager.initialize_system():
                sys.exit(1)
            
            manager.delete_file(args.delete)
        
        elif args.info:
            # Info mode
            if not manager.check_system_requirements():
                print("❌ System requirements not met. Run with --check-system for details.")
                sys.exit(1)
            
            if not manager.initialize_system():
                sys.exit(1)
            
            manager.show_system_info()
        
        elif args.file_path:
            # Index document mode
            if not manager.check_system_requirements():
                print("❌ System requirements not met. Run with --check-system for details.")
                sys.exit(1)
            
            if not manager.initialize_system():
                sys.exit(1)
            
            manager.index_document(
                args.file_path, 
                force_update=args.force_update,
                chunk_strategy=args.chunk_strategy,
                chunk_size=args.chunk_size,
                overlap=args.overlap
            )
        
        elif args.interactive:
            # Interactive mode
            run_interactive_mode(manager)
        
        else:
            # No arguments provided
            print("RAG Python Module - Main CLI Application")
            print("\nUse --help for usage information")
            print("\nQuick start:")
            print("  python main.py --check-system    # Check system requirements")
            print("  python main.py document.pdf      # Index a document")
            print("  python main.py --chunk-strategy sentence document.pdf  # Index with sentence-based chunking")
            print("  python main.py --search 'query'  # Search documents")
            print("  python main.py --interactive     # Run in interactive mode")
            print("\n💡 **Try interactive mode:** python main.py --interactive")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
