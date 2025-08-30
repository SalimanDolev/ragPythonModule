#!/usr/bin/env python3
"""
Example usage of the RAG Python Module with Gemini API

This script demonstrates how to use the module to index documents
and perform semantic search using Google Gemini embeddings.
"""

import os
from pathlib import Path
from index_documents import DocumentIndexer
from search_documents import DocumentSearcher

def main():
    """Demonstrate the RAG module functionality with Gemini API."""
    
    print("=== RAG Python Module Example (Gemini API) ===\n")
    
    # Configuration
    db_path = "./example_chroma_db"
    pdf_file = "???? ??? Jeen  AI Solution-1.pdf"  # Using the existing PDF
    
    # Get API key from environment or user input
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  GOOGLE_API_KEY environment variable not set.")
        print("   You can set it with: export GOOGLE_API_KEY='your_api_key_here'")
        print("   Or pass it directly in the code below.\n")
        
        # For demo purposes, you can uncomment and set your API key here:
        # api_key = "your_gemini_api_key_here"
        
        if not api_key:
            print("❌ Cannot proceed without Gemini API key.")
            print("   Please get your API key from: https://makersuite.google.com/app/apikey")
            return
    
    # Check if the PDF file exists
    if not os.path.exists(pdf_file):
        print(f"PDF file '{pdf_file}' not found. Please ensure the file exists in the current directory.")
        return
    
    print("1. Initializing Document Indexer with Gemini API...")
    try:
        indexer = DocumentIndexer(db_path=db_path, api_key=api_key)
        print(f"   ✓ Indexer initialized with database: {db_path}")
        print(f"   ✓ Using Gemini embedding model")
    except Exception as e:
        print(f"   ✗ Failed to initialize indexer: {e}")
        return
    
    print("\n2. Indexing PDF document...")
    try:
        success = indexer.index_document(pdf_file, chunk_size=800, overlap=150)
        if success:
            print(f"   ✓ Successfully indexed: {pdf_file}")
            
            # Get collection info
            info = indexer.get_collection_info()
            print(f"   ✓ Total chunks indexed: {info['total_documents']}")
        else:
            print(f"   ✗ Failed to index: {pdf_file}")
            return
    except Exception as e:
        print(f"   ✗ Error during indexing: {e}")
        return
    
    print("\n3. Initializing Document Searcher with Gemini API...")
    try:
        searcher = DocumentSearcher(db_path=db_path, api_key=api_key)
        print(f"   ✓ Searcher initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize searcher: {e}")
        return
    
    print("\n4. Performing sample searches...")
    
    # Sample queries to test
    sample_queries = [
        "What is artificial intelligence?",
        "machine learning",
        "data processing",
        "algorithm",
        "technology solution"
    ]
    
    for i, query in enumerate(sample_queries, 1):
        print(f"\n   Query {i}: '{query}'")
        try:
            results = searcher.search_documents(query, n_results=3, threshold=0.3)
            
            if results:
                print(f"   ✓ Found {len(results)} relevant results:")
                for j, result in enumerate(results[:2], 1):  # Show top 2 results
                    score = result['similarity_score']
                    content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                    print(f"      Result {j} (Score: {score:.3f}): {content_preview}")
            else:
                print(f"   ⚠ No results found (threshold too high)")
                
        except Exception as e:
            print(f"   ✗ Search error: {e}")
    
    print("\n5. Collection Statistics...")
    try:
        stats = searcher.get_collection_stats()
        print("   Collection Information:")
        for key, value in stats.items():
            print(f"      {key}: {value}")
    except Exception as e:
        print(f"   ✗ Error getting stats: {e}")
    
    print("\n=== Example completed successfully! ===")
    print(f"\nYou can now use the command line tools:")
    print(f"  • Index documents: python index_documents.py {pdf_file} --api-key YOUR_API_KEY")
    print(f"  • Search documents: python search_documents.py 'your query here' --api-key YOUR_API_KEY")
    print(f"  • Database location: {db_path}")
    print(f"\nOr set the GOOGLE_API_KEY environment variable for convenience.")

if __name__ == "__main__":
    main()
