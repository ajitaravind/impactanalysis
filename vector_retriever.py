import logging
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv, find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class VectorRetriever:
    """Retrieves code information from a vector database."""
    
    def __init__(
        self, 
        index_name: str = "code-repository",
        namespace: str = "code-analysis",
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
        relevance_threshold: float = 0.5
    ):
        """Initialize the vector retriever.
        
        Args:
            index_name: Name of the Pinecone index
            namespace: Namespace within the Pinecone index
            openai_api_key: OpenAI API key (will use env var if not provided)
            pinecone_api_key: Pinecone API key (will use env var if not provided)
            pinecone_environment: Pinecone environment (will use env var if not provided)
            embedding_model: Name of the embedding model to use
            top_k: Number of results to return
            relevance_threshold: Minimum relevance score (0-1) for results to be included
        """
        # Load environment variables if not already loaded
        load_dotenv(find_dotenv(), override=True)
        
        logger.info(f"Initializing VectorRetriever with index={index_name}, namespace={namespace}, relevance_threshold={relevance_threshold}")
        
        # Set relevance threshold
        self.relevance_threshold = relevance_threshold
        
        # Set up OpenAI API key
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        else:
            logger.info("OpenAI API key found")
        
        # Set up Pinecone API key and environment
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            logger.error("Pinecone API key not found")
            raise ValueError("Pinecone API key not found. Please provide it or set PINECONE_API_KEY environment variable.")
        else:
            logger.info("Pinecone API key found")
            
        self.pinecone_environment = pinecone_environment or os.environ.get("PINECONE_ENVIRONMENT")
        if not self.pinecone_environment:
            logger.error("Pinecone environment not found")
            raise ValueError("Pinecone environment not found. Please provide it or set PINECONE_ENVIRONMENT environment variable.")
        else:
            logger.info(f"Pinecone environment found: {self.pinecone_environment}")
        
        # Initialize embeddings
        logger.info(f"Initializing embeddings with model: {embedding_model}")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self.openai_api_key
        )
        
        # Set up Pinecone parameters
        self.index_name = index_name
        self.namespace = namespace
        self.top_k = top_k
        
        # Initialize Pinecone and load vector store
        self._init_pinecone()
        self._load_vector_store()
    
    def _init_pinecone(self):
        """Initialize Pinecone connection."""
        try:
            logger.info(f"VectorRetriever: Initializing Pinecone with environment: {self.pinecone_environment}")
            # Initialize the Pinecone client using the Pinecone class
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            logger.info("VectorRetriever: Pinecone client initialized")
            
            # Check if index exists
            logger.info(f"VectorRetriever: Checking if index exists: {self.index_name}")
            index_names = self.pc.list_indexes().names()
            logger.info(f"VectorRetriever: Available indexes: {index_names}")
            
            if self.index_name not in index_names:
                logger.warning(f"VectorRetriever: Pinecone index not found: {self.index_name}")
                self.index_exists = False
            else:
                logger.info(f"VectorRetriever: Found Pinecone index: {self.index_name}")
                self.index_exists = True
                
        except Exception as e:
            logger.error(f"VectorRetriever: Error initializing Pinecone: {str(e)}", exc_info=True)
            self.index_exists = False
    
    def _load_vector_store(self):
        """Load the vector store."""
        try:
            if self.index_exists:
                logger.info(f"VectorRetriever: Loading vector store from Pinecone index: {self.index_name}")
                # Use from_existing_index without passing client directly
                self.vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=self.namespace
                )
                logger.info("VectorRetriever: Vector store loaded successfully")
            else:
                logger.warning(f"VectorRetriever: Vector store not available: Pinecone index {self.index_name} not found")
                self.vector_store = None
        except Exception as e:
            logger.error(f"VectorRetriever: Error loading vector store: {str(e)}", exc_info=True)
            self.vector_store = None
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant code snippets based on the query.
        
        Args:
            query: The natural language query
            
        Returns:
            Dictionary containing retrieval results
        """
        logger.info(f"VectorRetriever: Retrieving information for query: {query}")
        
        if self.vector_store is None:
            logger.warning("VectorRetriever: Vector store not available")
            return {
                "query": query,
                "raw_results": [],
                "response": "Vector database not available or empty."
            }
        
        try:
            # Perform similarity search
            logger.info(f"VectorRetriever: Performing similarity search with k={self.top_k}")
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=self.top_k
            )
            
            logger.info(f"VectorRetriever: Similarity search returned {len(results)} results")
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # Convert score to similarity score (Pinecone returns distance)
                # For cosine distance, similarity = 1 - distance
                similarity_score = 1 - score
                
                # Log all scores for better debugging
                logger.info(f"VectorRetriever: Result score: {similarity_score:.4f} for content: {doc.page_content[:50]}...")
                
                # Skip results with low relevance
                if similarity_score < self.relevance_threshold:
                    logger.info(f"VectorRetriever: Skipping result with low relevance score: {similarity_score:.4f} (threshold: {self.relevance_threshold})")
                    continue
                    
                formatted_result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": similarity_score
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"VectorRetriever: Formatted {len(formatted_results)} results after relevance filtering (threshold: {self.relevance_threshold})")
            
            # Generate a summary response
            response = self._generate_response(query, formatted_results)
            
            return {
                "query": query,
                "raw_results": formatted_results,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"VectorRetriever: Error retrieving from vector store: {str(e)}", exc_info=True)
            return {
                "query": query,
                "raw_results": [],
                "response": f"Error retrieving information: {str(e)}"
            }
    
    def _generate_response(self, query: str, results: List[Dict]) -> str:
        """Generate a human-readable response from the retrieval results.
        
        Args:
            query: The original query
            results: List of retrieval results
            
        Returns:
            Formatted response string
        """
        if not results:
            return "No relevant code snippets found for your query."
        
        response_parts = [f"Found {len(results)} relevant code snippets:"]
        
        for i, result in enumerate(results, 1):
            metadata = result["metadata"]
            file_path = metadata.get("source", "Unknown file")
            code_type = metadata.get("code_type", "code snippet")
            code_name = metadata.get("code_name", "")
            
            # Format the code snippet
            content = result["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            
            # Add to response
            response_parts.append(f"\n{i}. {code_type.capitalize()}: {code_name}")
            response_parts.append(f"   File: {file_path}")
            response_parts.append(f"   Relevance: {result['relevance_score']:.2f}")
            response_parts.append(f"   ```python\n   {content}\n   ```")
        
        return "\n".join(response_parts)
    
    def close(self):
        """Close the vector store connection."""
        # Clean up Pinecone client
        if hasattr(self, 'pc') and self.pc is not None:
            del self.pc
        logger.info("Vector store connection closed")

def test_pinecone_connection():
    """Test function to diagnose Pinecone connection issues."""
    import os
    from dotenv import load_dotenv, find_dotenv
    
    # Load environment variables
    load_dotenv(find_dotenv(), override=True)
    
    # Get environment variables
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    
    print("\n" + "=" * 50)
    print("PINECONE CONNECTION TEST")
    print("-" * 50)
    
    # Check environment variables
    print("Checking environment variables:")
    print(f"- OPENAI_API_KEY: {'Found' if OPENAI_API_KEY else 'Not found'}")
    print(f"- PINECONE_API_KEY: {'Found' if PINECONE_API_KEY else 'Not found'}")
    print(f"- PINECONE_ENVIRONMENT: {'Found' if PINECONE_ENVIRONMENT else 'Not found'}")
    
    if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT]):
        print("\nMissing required environment variables. Please check your .env file.")
        return
    
    # Test Pinecone connection
    print("\nTesting Pinecone connection...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print("✅ Successfully connected to Pinecone")
        
        # List indexes
        indexes = pc.list_indexes().names()
        print(f"\nAvailable indexes: {indexes}")
        
        if not indexes:
            print("\n⚠️ No indexes found in your Pinecone account.")
            print("You need to create an index before using the vector retriever.")
        else:
            print("\nTesting vector retriever initialization...")
            try:
                retriever = VectorRetriever(
                    index_name=indexes[0],  # Use the first available index
                    openai_api_key=OPENAI_API_KEY,
                    pinecone_api_key=PINECONE_API_KEY,
                    pinecone_environment=PINECONE_ENVIRONMENT
                )
                print(f"✅ Successfully initialized vector retriever with index: {indexes[0]}")
                
                # Test a simple query
                print("\nTesting a simple query...")
                results = retriever.retrieve("test query")
                print(f"Query returned {len(results.get('raw_results', []))} results")
                
                retriever.close()
            except Exception as e:
                print(f"❌ Error initializing vector retriever: {str(e)}")
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {str(e)}")
    
    print("=" * 50)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vector retriever utilities")
    parser.add_argument("--test", action="store_true", help="Run Pinecone connection test")
    parser.add_argument("--query", help="The query to search for")
    parser.add_argument("--index_name", default="code-repository", help="Name of the Pinecone index")
    parser.add_argument("--namespace", default="code-analysis", help="Namespace within the Pinecone index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.test:
        test_pinecone_connection()
    elif args.query:
        # Initialize retriever
        retriever = VectorRetriever(
            index_name=args.index_name,
            namespace=args.namespace,
            top_k=args.top_k
        )
        
        # Retrieve information
        results = retriever.retrieve(args.query)
        
        # Print results
        print("\n" + "=" * 50)
        print("QUERY:", args.query)
        print("-" * 50)
        print(results["response"])
        print("=" * 50)
        
        # Close retriever
        retriever.close()
    else:
        print("Please specify either --test or --query")
        print("Example: python vector_retriever.py --test")
        print("Example: python vector_retriever.py --query 'How does error handling work?'") 