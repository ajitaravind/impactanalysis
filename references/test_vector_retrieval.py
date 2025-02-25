#!/usr/bin/env python
"""
Test script for vector retrieval functionality.
This script helps diagnose issues with the vector retrieval process.
"""

import os
import logging
from dotenv import load_dotenv, find_dotenv
from vector_retriever import VectorRetriever, test_pinecone_connection
from retriever import LLMRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def test_standalone_vector_retriever():
    """Test the VectorRetriever class directly."""
    print("\n" + "=" * 50)
    print("TESTING STANDALONE VECTOR RETRIEVER")
    print("-" * 50)
    
    # First, test the Pinecone connection
    test_pinecone_connection()
    
    # If the connection test passes, try a direct query
    try:
        # Load environment variables
        load_dotenv(find_dotenv(), override=True)
        
        # Get environment variables
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
        
        if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
            print("Missing required environment variables. Skipping direct query test.")
            return
        
        # Initialize the vector retriever
        retriever = VectorRetriever(
            openai_api_key=OPENAI_API_KEY,
            pinecone_api_key=PINECONE_API_KEY,
            pinecone_environment=PINECONE_ENVIRONMENT,
            relevance_threshold=0.5
        )
        
        # Test a query
        query = "How does error handling work?"
        print(f"\nTesting query: '{query}'")
        print(f"Using relevance threshold: {retriever.relevance_threshold}")
        
        results = retriever.retrieve(query)
        
        print(f"Query returned {len(results.get('raw_results', []))} results")
        if results.get('raw_results'):
            print("\nFirst result:")
            first_result = results['raw_results'][0]
            print(f"- Content: {first_result['content'][:100]}...")
            print(f"- Metadata: {first_result['metadata']}")
            print(f"- Relevance: {first_result['relevance_score']}")
        
        # Close the retriever
        retriever.close()
        
    except Exception as e:
        print(f"Error in standalone vector retriever test: {str(e)}")

def test_integrated_retriever():
    """Test the LLMRetriever with integrated vector search."""
    print("\n" + "=" * 50)
    print("TESTING INTEGRATED RETRIEVER")
    print("-" * 50)
    
    try:
        # Load environment variables
        load_dotenv(find_dotenv(), override=True)
        
        # Get environment variables
        NEO4J_URI = os.environ.get("NEO4J_URI")
        NEO4J_USER = os.environ.get("NEO4J_USERNAME")
        NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
        
        if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY]):
            print("Missing required Neo4j environment variables. Skipping integrated retriever test.")
            return
        
        # Initialize vector retriever
        vector_retriever = None
        if all([PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
            try:
                vector_retriever = VectorRetriever(
                    openai_api_key=OPENAI_API_KEY,
                    pinecone_api_key=PINECONE_API_KEY,
                    pinecone_environment=PINECONE_ENVIRONMENT,
                    relevance_threshold=0.5
                )
                print("Vector retriever initialized successfully")
                print(f"Using relevance threshold: {vector_retriever.relevance_threshold}")
            except Exception as e:
                print(f"Failed to initialize vector retriever: {e}")
        else:
            print("Missing Pinecone environment variables. Vector retrieval will be skipped.")
        
        # Initialize the LLM retriever
        retriever = LLMRetriever(
            NEO4J_URI, 
            NEO4J_USER, 
            NEO4J_PASSWORD, 
            OPENAI_API_KEY,
            vector_retriever=vector_retriever
        )
        
        # Test a query
        query = "How does error handling work?"
        print(f"\nTesting query: '{query}'")
        
        results = retriever.retrieve(query)
        
        print(f"Query returned {len(results.get('raw_results', []))} results")
        print(f"Vector results included: {results.get('has_vector_results', False)}")
        
        if results.get('raw_results'):
            print("\nFirst result:")
            first_result = results['raw_results'][0]
            if 'is_vector_result' in first_result:
                print(f"- Is vector result: {first_result.get('is_vector_result', False)}")
            print(f"- Name: {first_result.get('name', 'N/A')}")
            print(f"- Type: {first_result.get('type', 'N/A')}")
            if 'content' in first_result:
                print(f"- Content: {first_result['content'][:100]}...")
        
        # Close the retriever
        retriever.close()
        
    except Exception as e:
        print(f"Error in integrated retriever test: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test vector retrieval functionality")
    parser.add_argument("--standalone", action="store_true", help="Test standalone vector retriever")
    parser.add_argument("--integrated", action="store_true", help="Test integrated retriever")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.standalone or args.all:
        test_standalone_vector_retriever()
    
    if args.integrated or args.all:
        test_integrated_retriever()
    
    if not (args.standalone or args.integrated or args.all):
        print("Please specify a test to run:")
        print("  --standalone: Test standalone vector retriever")
        print("  --integrated: Test integrated retriever")
        print("  --all: Run all tests")
        print("\nExample: python test_vector_retrieval.py --all") 