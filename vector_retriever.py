import logging
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv, find_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

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
        top_k: int = 5
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
        """
        # Load environment variables if not already loaded
        load_dotenv(find_dotenv(), override=True)
        
        # Set up OpenAI API key
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        # Set up Pinecone API key and environment
        self.pinecone_api_key = pinecone_api_key or os.environ.get("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found. Please provide it or set PINECONE_API_KEY environment variable.")
            
        self.pinecone_environment = pinecone_environment or os.environ.get("PINECONE_ENVIRONMENT")
        if not self.pinecone_environment:
            raise ValueError("Pinecone environment not found. Please provide it or set PINECONE_ENVIRONMENT environment variable.")
        
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
            logger.info(f"Initializing Pinecone with environment: {self.pinecone_environment}")
            # Initialize the Pinecone client using the Pinecone class
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.warning(f"Pinecone index not found: {self.index_name}")
                self.index_exists = False
            else:
                logger.info(f"Found Pinecone index: {self.index_name}")
                self.index_exists = True
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            self.index_exists = False
    
    def _load_vector_store(self):
        """Load the vector store."""
        try:
            if self.index_exists:
                logger.info(f"Loading vector store from Pinecone index: {self.index_name}")
                # Use from_existing_index without passing client directly
                self.vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings,
                    namespace=self.namespace
                )
                logger.info("Vector store loaded successfully")
            else:
                logger.warning(f"Vector store not available: Pinecone index {self.index_name} not found")
                self.vector_store = None
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            self.vector_store = None
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant code snippets based on the query.
        
        Args:
            query: The natural language query
            
        Returns:
            Dictionary containing retrieval results
        """
        logger.info(f"Retrieving information for query: {query}")
        
        if self.vector_store is None:
            logger.warning("Vector store not available")
            return {
                "query": query,
                "raw_results": [],
                "response": "Vector database not available or empty."
            }
        
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=self.top_k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # Convert score to similarity score (Pinecone returns distance)
                # For cosine distance, similarity = 1 - distance
                similarity_score = 1 - score
                
                # Skip results with low relevance
                if similarity_score < 0.7:  # Adjust threshold as needed
                    continue
                    
                formatted_result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": similarity_score
                }
                formatted_results.append(formatted_result)
            
            # Generate a summary response
            response = self._generate_response(query, formatted_results)
            
            return {
                "query": query,
                "raw_results": formatted_results,
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Error retrieving from vector store: {str(e)}")
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

def main():
    """Command line interface for the vector retriever."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve code information from a vector database")
    parser.add_argument("--query", required=True, help="The query to search for")
    parser.add_argument("--index_name", default="code-repository", help="Name of the Pinecone index")
    parser.add_argument("--namespace", default="code-analysis", help="Namespace within the Pinecone index")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main() 