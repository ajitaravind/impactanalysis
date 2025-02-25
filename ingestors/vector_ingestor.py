from dotenv import load_dotenv, find_dotenv
import os
import logging
from typing import List, Optional

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class VectorDBIngestor:
    """Ingests code into a vector database for semantic retrieval."""
    
    def __init__(
        self, 
        index_name: str = "code-repository",
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        pinecone_environment: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize the vector database ingestor.
        
        Args:
            index_name: Name of the Pinecone index
            openai_api_key: OpenAI API key (will use env var if not provided)
            pinecone_api_key: Pinecone API key (will use env var if not provided)
            pinecone_environment: Pinecone environment (will use env var if not provided)
            embedding_model: Name of the embedding model to use
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
        
        # Set up Pinecone index name
        self.index_name = index_name
        
        # Initialize Pinecone
        self._init_pinecone()
        
        # Initialize vector store
        self.vector_store = None
    
    def _init_pinecone(self):
        """Initialize Pinecone connection."""
        try:
            logger.info(f"Initializing Pinecone with environment: {self.pinecone_environment}")
            # Initialize Pinecone client using the Pinecone class instead of pinecone.init()
            self.pc = Pinecone(api_key=self.pinecone_api_key)

            # Check if index exists, create if it doesn't
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.pinecone_environment)
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise
    
    def load_code_from_repo(
        self, 
        repo_path: str, 
        file_glob: str = "**/*",
        file_suffixes: List[str] = [".py"],
        exclude_patterns: List[str] = ["**/venv/**", "**/.git/**", "**/__pycache__/**"]
    ) -> List:
        """Load code files from a repository.
        
        Args:
            repo_path: Path to the code repository
            file_glob: Glob pattern for finding files
            file_suffixes: List of file suffixes to include
            exclude_patterns: List of patterns to exclude
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading code from repository: {repo_path}")
        
        try:
            loader = GenericLoader.from_filesystem(
                repo_path,
                glob=file_glob,
                suffixes=file_suffixes,
                exclude=exclude_patterns,
                parser=LanguageParser(language=Language.PYTHON)
            )
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from repository")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading code from repository: {str(e)}")
            raise
    
    def split_documents(
        self, 
        documents: List, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200
    ) -> List:
        """Split documents into chunks suitable for embedding.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split documents
        """
        logger.info(f"Splitting documents with chunk size {chunk_size} and overlap {chunk_overlap}")
        
        try:
            # Use Python-aware splitter
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            split_docs = splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            
            # Enhance metadata
            for doc in split_docs:
                # Extract function/class names if possible
                self._enhance_metadata(doc)
            
            return split_docs
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def _enhance_metadata(self, doc):
        """Enhance document metadata with code-specific information."""
        content = doc.page_content.strip()
        
        # Simple heuristics to identify code type
        if content.startswith("def "):
            doc.metadata["code_type"] = "function"
            # Extract function name
            try:
                func_name = content.split("def ")[1].split("(")[0].strip()
                doc.metadata["code_name"] = func_name
            except:
                pass
                
        elif content.startswith("class "):
            doc.metadata["code_type"] = "class"
            # Extract class name
            try:
                class_name = content.split("class ")[1].split("(")[0].split(":")[0].strip()
                doc.metadata["code_name"] = class_name
            except:
                pass
                
        elif "import " in content.split("\n")[0]:
            doc.metadata["code_type"] = "import"
            
        # Add source code language
        doc.metadata["language"] = "python"
    
    def ingest_to_vectordb(self, documents: List):
        """Ingest documents into the vector database.
        
        Args:
            documents: List of documents to ingest
        """
        logger.info(f"Ingesting {len(documents)} documents to Pinecone index: {self.index_name}")
        
        try:
            # Create vector store without passing client directly
            # Extract texts and metadatas from documents
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create vector store
            self.vector_store = PineconeVectorStore.from_texts(
                texts=texts,
                embedding=self.embeddings,
                index_name=self.index_name,
                namespace="code-analysis",
                metadatas=metadatas
            )
            
            logger.info(f"Successfully ingested documents to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error ingesting to Pinecone: {str(e)}")
            raise
    
    def process_repository(
        self, 
        repo_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Process a repository and ingest it into the vector database.
        
        Args:
            repo_path: Path to the code repository
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        """
        logger.info(f"Processing repository: {repo_path}")
        
        # Load documents
        documents = self.load_code_from_repo(repo_path)
        
        # Split documents
        split_docs = self.split_documents(documents, chunk_size, chunk_overlap)
        
        # Ingest to vector database
        self.ingest_to_vectordb(split_docs)
        
        logger.info(f"Repository processing complete: {repo_path}")
        return self.vector_store

def main():
    """Command line interface for the vector database ingestor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest code into a vector database")
    parser.add_argument("--repo_path", required=True, help="Path to the code repository")
    parser.add_argument("--index_name", default="code-repository", help="Name of the Pinecone index")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of each chunk")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between chunks")
    
    args = parser.parse_args()
    
    # Initialize ingestor
    ingestor = VectorDBIngestor(index_name=args.index_name)
    
    # Process repository
    ingestor.process_repository(
        repo_path=args.repo_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

if __name__ == "__main__":
    main() 