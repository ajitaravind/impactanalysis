import logging
from typing import Dict, Sequence, TypedDict, Annotated, List
import operator
import os
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from retriever import LLMRetriever
from vector_retriever import VectorRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

###############################################################################
# State Management
###############################################################################
class CodeAnalysisState(TypedDict):
    """Represents the state of our code analysis graph."""
    user_question: str
    retrieval_results: Dict
    vector_results: Dict
    analysis_result: str
    messages: Annotated[Sequence[BaseMessage], operator.add]

###############################################################################
# Retrieval Nodes
###############################################################################
def structured_retrieval_node(state: CodeAnalysisState, retriever: LLMRetriever) -> Dict:
    """Retrieves information from the code database using the LLMRetriever."""
    try:
        logger.info(f"Retrieving structured information for: {state['user_question']}")
        
        # Use the existing retriever to get information
        results = retriever.retrieve(state['user_question'])
        
        # Check if we got any results
        if not results.get("raw_results"):
            logger.info("No results found, trying alternative query formats")
            
            # Try alternative formats (replace spaces with underscores)
            alt_question = state['user_question'].replace(" ", "_")
            if alt_question != state['user_question']:
                logger.info(f"Trying alternative query: {alt_question}")
                alt_results = retriever.retrieve(alt_question)
                
                # If alternative query found results, use those instead
                if alt_results.get("raw_results"):
                    logger.info("Alternative query found results")
                    results = alt_results
                    
                    # Add a note about the query transformation
                    results["response"] = (
                        f"Note: I searched for '{alt_question}' instead of '{state['user_question']}' "
                        f"and found the following results:\n\n{results['response']}"
                    )
        
        return {
            "retrieval_results": results,
            "messages": [AIMessage(content="Retrieved structured code information")]
        }
        
    except Exception as e:
        logger.error(f"Error in structured retrieval node: {str(e)}", exc_info=True)
        return {
            "retrieval_results": {
                "query": state['user_question'],
                "cypher_query": "",
                "raw_results": [],
                "response": f"Error retrieving information: {str(e)}"
            },
            "messages": [AIMessage(content=f"Error in retrieval: {str(e)}")]
        }

###############################################################################
# Analysis Node
###############################################################################
def create_analysis_agent():
    """Creates an agent specialized in analyzing code information."""
    
    system_msg = """You are an expert code analyst. Your role is to:
    
    1. Analyze the retrieved information about code structure and relationships
    2. Provide clear, technical explanations about how the code works
    3. Focus on implementation details, design patterns, and code organization
    4. Highlight important relationships between components
    5. Be precise and accurate in your technical descriptions
    
    When answering questions about code:
    - Cite specific functions, classes, and files from the retrieved data
    - Explain the purpose and implementation of code components
    - Describe how different parts of the code interact
    - Use technical language appropriate for software developers
    
    If the user is asking about impact analysis or how to implement a new feature:
    - Identify specific files, functions, and classes that would need to be modified
    - Suggest new components that might need to be created
    - Prioritize changes based on their importance and complexity
    - Consider architectural patterns and existing code organization
    - Suggest a step-by-step implementation plan
    - Identify potential risks and mitigations
    
    Impact analysis questions typically include phrases like:
    - "What changes would be needed to..."
    - "How to implement..."
    - "What would it take to add..."
    - "Impact of adding..."
    - "How to introduce..."
    """
    
    human_msg = """Question: {question}
    
    Cypher Query: {cypher_query}
    
    Graph Results: {graph_results}
    
    Graph Response: {graph_response}
    
    Vector Results Available: {has_vector_results}
    
    Suggestions: {suggestions}
    
    Please provide a detailed technical analysis. If this is an impact analysis question about implementing a new feature, structure your response as an implementation plan with affected components and required changes:
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg)
    ])
    
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    
    return prompt | model

def analysis_node(state: CodeAnalysisState) -> Dict:
    """Analyzes the retrieved information to answer the user's question."""
    try:
        logger.info("Analyzing retrieved information")
        
        # Get the retrieval results
        retrieval_results = state.get("retrieval_results", {})
        
        # Extract the raw results and response
        graph_results = retrieval_results.get("raw_results", [])
        graph_response = retrieval_results.get("response", "No response available")
        
        # Check if we have vector results integrated in the retrieval results
        has_vector_results = retrieval_results.get("has_vector_results", False)
        
        # Check if we have any results
        if not graph_results:
            logger.warning("No results found in either retrieval method")
            
            # Generate suggestions for alternative queries
            suggestions = []
            query = state["user_question"]
            
            # Suggest camelCase
            if " " in query:
                camel_case = query.split(" ")[0] + "".join(word.capitalize() for word in query.split(" ")[1:])
                suggestions.append(f"Try '{camel_case}' (camelCase format)")
            
            # Suggest snake_case
            if " " in query:
                snake_case = re.sub(r'([a-z])([A-Z])', r'\1_\2', query).lower()
                suggestions.append(f"Try '{snake_case}' (snake_case format)")
            
            # Add these suggestions to the retrieval results
            retrieval_results["suggestions"] = suggestions
        
        analyzer = create_analysis_agent()
        response = analyzer.invoke({
            "question": state["user_question"],
            "cypher_query": retrieval_results.get("cypher_query", "No query available"),
            "graph_results": graph_results,
            "graph_response": graph_response,
            "has_vector_results": has_vector_results,
            "suggestions": retrieval_results.get("suggestions", [])
        })
        
        analysis_result = response.content
        logger.info("Analysis complete")
        
        return {
            "analysis_result": analysis_result,
            "messages": [AIMessage(content=analysis_result)]
        }
        
    except Exception as e:
        logger.error(f"Error in analysis node: {str(e)}", exc_info=True)
        return {
            "analysis_result": f"Error analyzing code: {str(e)}",
            "messages": [AIMessage(content=f"Error in analysis: {str(e)}")]
        }

###############################################################################
# Graph Construction
###############################################################################
def create_code_analysis_graph(retriever: LLMRetriever) -> StateGraph:
    """Creates and configures the code analysis graph."""
    
    graph = StateGraph(CodeAnalysisState)
    
    # Add nodes - vector retrieval is now integrated into the LLMRetriever
    graph.add_node("structured_retrieve", lambda state: structured_retrieval_node(state, retriever))
    graph.add_node("analyze", analysis_node)
    
    # Configure edges - simplified flow since vector retrieval is integrated
    graph.set_entry_point("structured_retrieve")
    graph.add_edge("structured_retrieve", "analyze")
    graph.add_edge("analyze", END)
    
    return graph.compile()

###############################################################################
# Main Interface
###############################################################################
class CodeAnalysisAgent:
    """Agent for analyzing code using natural language queries."""
    
    def __init__(
        self, 
        neo4j_uri, 
        neo4j_user, 
        neo4j_password, 
        pinecone_index_name="code-repository",
        pinecone_namespace="code-analysis",
        openai_api_key=None,
        pinecone_api_key=None,
        pinecone_environment=None,
        relevance_threshold=0.5
    ):
        """Initialize the agent with database connection parameters."""
        # Set up vector retriever first
        self.vector_retriever = None
        
        # Ensure pinecone_index_name is not accidentally set to an API key
        if pinecone_index_name and (
            pinecone_index_name.startswith("sk-") or 
            len(pinecone_index_name) > 50
        ):
            logger.warning(f"Invalid Pinecone index name detected (looks like an API key). Using default 'code-repository' instead.")
            pinecone_index_name = "code-repository"
            
        try:
            self.vector_retriever = VectorRetriever(
                index_name=pinecone_index_name,
                namespace=pinecone_namespace,
                openai_api_key=openai_api_key,
                pinecone_api_key=pinecone_api_key,
                pinecone_environment=pinecone_environment,
                relevance_threshold=relevance_threshold
            )
            logger.info(f"Vector retriever initialized with Pinecone index: {pinecone_index_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize vector retriever: {str(e)}")
        
        # Set up Neo4j retriever with vector retriever
        self.retriever = LLMRetriever(
            neo4j_uri, 
            neo4j_user, 
            neo4j_password, 
            openai_api_key, 
            vector_retriever=self.vector_retriever
        )
        
        # Create the graph
        self.graph_executor = create_code_analysis_graph(self.retriever)
    
    def analyze(self, question: str) -> str:
        """Analyze code based on a natural language question."""
        try:
            logger.info(f"Starting analysis for question: {question}")
            
            initial_state = {
                "user_question": question,
                "retrieval_results": {},
                "vector_results": {},
                "analysis_result": "",
                "messages": [HumanMessage(content=question)]
            }
            
            final_state = self.graph_executor.invoke(initial_state)
            
            if "analysis_result" in final_state:
                return final_state["analysis_result"]
            
            raise RuntimeError("No analysis result generated")
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            return f"Failed to analyze question: {str(e)}"
    
    def close(self):
        """Close the database connections."""
        self.retriever.close()
        if self.vector_retriever:
            self.vector_retriever.close()

###############################################################################
# Command Line Interface
###############################################################################
def main():
    """Command line interface for the code analysis agent."""
    import os
    from dotenv import load_dotenv, find_dotenv
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Code Analysis Agent")
    parser.add_argument("--pinecone_index", default="code-repository", help="Name of the Pinecone index")
    parser.add_argument("--pinecone_namespace", default="code-analysis", help="Namespace within the Pinecone index")
    parser.add_argument("--relevance_threshold", type=float, default=0.5, help="Minimum relevance score (0-1) for vector search results")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(find_dotenv(), override=True)
    
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
    
    # Initialize the agent
    agent = CodeAnalysisAgent(
        NEO4J_URI, 
        NEO4J_USER, 
        NEO4J_PASSWORD, 
        pinecone_index_name=args.pinecone_index,
        pinecone_namespace=args.pinecone_namespace,
        openai_api_key=OPENAI_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_environment=PINECONE_ENVIRONMENT,
        relevance_threshold=args.relevance_threshold
    )
    
    print("Code Analysis Agent")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        question = input("\nEnter your question: ")
        if question.lower() == "exit":
            break
        
        print("\nAnalyzing...")
        result = agent.analyze(question)
        
        print("\n" + "=" * 50)
        print("ANALYSIS RESULT:")
        print("-" * 40)
        print(result)
    
    agent.close()

if __name__ == "__main__":
    main() 