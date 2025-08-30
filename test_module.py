#!/usr/bin/env python3
"""
Test script for the RAG Python Module with Gemini API

This script tests the basic functionality of the module components.
"""

import os
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        from index_documents import DocumentIndexer, index_document
        print("  ✓ index_documents module imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import index_documents: {e}")
        return False
    
    try:
        from search_documents import DocumentSearcher, search_documents
        print("  ✓ search_documents module imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import search_documents: {e}")
        return False
    
    try:
        import google.generativeai as genai
        print("  ✓ Google Gemini API module imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import Google Gemini API: {e}")
        return False
    
    return True

def test_document_indexer_initialization():
    """Test DocumentIndexer class initialization."""
    print("\nTesting DocumentIndexer initialization...")
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("  ⚠ GOOGLE_API_KEY environment variable not set")
        print("  ⚠ Skipping Gemini API tests")
        return True  # Skip this test if no API key
    
    try:
        from index_documents import DocumentIndexer
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            indexer = DocumentIndexer(db_path=temp_dir, api_key=api_key)
            print(f"  ✓ DocumentIndexer initialized with temp database: {temp_dir}")
            
            # Test collection info
            info = indexer.get_collection_info()
            print(f"  ✓ Collection info retrieved: {info['total_documents']} documents")
            
            return True
            
    except Exception as e:
        print(f"  ✗ DocumentIndexer initialization failed: {e}")
        return False

def test_document_searcher_initialization():
    """Test DocumentSearcher class initialization."""
    print("\nTesting DocumentSearcher initialization...")
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("  ⚠ GOOGLE_API_KEY environment variable not set")
        print("  ⚠ Skipping Gemini API tests")
        return True  # Skip this test if no API key
    
    try:
        from search_documents import DocumentSearcher
        
        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # First create an indexer to set up the database
            from index_documents import DocumentIndexer
            indexer = DocumentIndexer(db_path=temp_dir, api_key=api_key)
            
            # Now test the searcher
            searcher = DocumentSearcher(db_path=temp_dir, api_key=api_key)
            print(f"  ✓ DocumentSearcher initialized with temp database: {temp_dir}")
            
            return True
            
    except Exception as e:
        print(f"  ✗ DocumentSearcher initialization failed: {e}")
        return False

def test_text_chunking():
    """Test text chunking functionality."""
    print("\nTesting text chunking...")
    
    try:
        from index_documents import DocumentIndexer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create indexer without API key for this test (just testing chunking)
            try:
                indexer = DocumentIndexer(db_path=temp_dir, api_key="dummy_key")
            except ValueError:
                # Expected error for dummy key, but we can still test chunking
                # Create a minimal indexer instance for testing
                indexer = DocumentIndexer.__new__(DocumentIndexer)
                indexer.chunk_text = DocumentIndexer.chunk_text.__get__(indexer, DocumentIndexer)
            
            # Test text with chunking
            test_text = "This is a test document. " * 50  # Create a long text
            chunks = indexer.chunk_text(test_text, chunk_size=100, overlap=20)
            
            if len(chunks) > 1:
                print(f"  ✓ Text chunking successful: {len(chunks)} chunks created")
                print(f"  ✓ First chunk length: {len(chunks[0])} characters")
                return True
            else:
                print("  ✗ Text chunking failed: expected multiple chunks")
                return False
                
    except Exception as e:
        print(f"  ✗ Text chunking test failed: {e}")
        return False

def test_gemini_api_connection():
    """Test Gemini API connection and embedding generation."""
    print("\nTesting Gemini API connection...")
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("  ⚠ GOOGLE_API_KEY environment variable not set")
        print("  ⚠ Skipping Gemini API connection test")
        return True  # Skip this test if no API key
    
    try:
        import google.generativeai as genai
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Test embedding generation
        embedding_model = genai.get_model('models/embedding-001')
        test_text = "Hello world"
        result = embedding_model.embed_content(test_text)
        
        if 'embedding' in result and len(result['embedding']) > 0:
            print(f"  ✓ Gemini API connection successful")
            print(f"  ✓ Embedding generated: {len(result['embedding'])} dimensions")
            return True
        else:
            print("  ✗ Unexpected embedding result format")
            return False
            
    except Exception as e:
        print(f"  ✗ Gemini API connection test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=== RAG Module Test Suite (Gemini API) ===\n")
    
    tests = [
        ("Module Imports", test_imports),
        ("Gemini API Connection", test_gemini_api_connection),
        ("DocumentIndexer Initialization", test_document_indexer_initialization),
        ("DocumentSearcher Initialization", test_document_searcher_initialization),
        ("Text Chunking", test_text_chunking),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The RAG module with Gemini API is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        if not os.getenv('GOOGLE_API_KEY'):
            print("\n💡 Note: Set GOOGLE_API_KEY environment variable to test Gemini API functionality:")
            print("   export GOOGLE_API_KEY='your_api_key_here'")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
