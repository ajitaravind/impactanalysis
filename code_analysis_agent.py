import logging
from typing import Dict, Sequence, TypedDict, Annotated
import operator
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from retriever import LLMRetriever

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
    
    IMPORTANT: If no relevant information is found in the database results, clearly state that 
    the requested code element was not found in the codebase. Do NOT generate hypothetical 
    explanations about what the code might do based on the name alone. Instead, suggest 
    alternative search terms or approaches that might yield better results, including any 
    provided suggestions.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", """Question: {question}
        
        Structured Information:
        - Cypher Query: {cypher_query}
        - Database Results: {raw_results}
        - Initial Analysis: {initial_response}
        
        Suggested Alternative Queries: {suggestions}
        
        Please provide a comprehensive analysis of this code. If no relevant information was found,
        clearly indicate this and suggest alternative approaches:""")
    ])
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    return prompt | llm

def analysis_node(state: CodeAnalysisState) -> Dict:
    """Analyzes the retrieved information to provide a comprehensive answer."""
    try:
        logger.info("Starting code analysis")
        
        retrieval_results = state["retrieval_results"]
        raw_results = retrieval_results.get("raw_results", [])
        
        # Check if we have any meaningful results
        if not raw_results:
            logger.warning("No results found in database for the query")
            
            # Generate suggestions based on the query
            query = state["user_question"]
            suggestions = []
            
            # Check for common naming patterns
            if " " in query:
                suggestions.append(f"Try '{query.replace(' ', '_')}' (with underscores)")
            if "_" in query:
                suggestions.append(f"Try '{query.replace('_', ' ')}' (without underscores)")
            
            # Check for camelCase vs snake_case
            import re
            if re.search(r'[a-z][A-Z]', query):  # camelCase detected
                # Convert camelCase to snake_case
                snake_case = re.sub(r'([a-z])([A-Z])', r'\1_\2', query).lower()
                suggestions.append(f"Try '{snake_case}' (snake_case format)")
            
            # Add these suggestions to the retrieval results
            retrieval_results["suggestions"] = suggestions
        
        analyzer = create_analysis_agent()
        response = analyzer.invoke({
            "question": state["user_question"],
            "cypher_query": retrieval_results.get("cypher_query", "No query available"),
            "raw_results": raw_results,
            "initial_response": retrieval_results.get("response", "No initial analysis available"),
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
    
    # Add nodes
    graph.add_node("structured_retrieve", lambda state: structured_retrieval_node(state, retriever))
    graph.add_node("analyze", analysis_node)
    
    # Configure edges
    graph.set_entry_point("structured_retrieve")
    graph.add_edge("structured_retrieve", "analyze")
    graph.add_edge("analyze", END)
    
    return graph.compile()

###############################################################################
# Main Interface
###############################################################################
class CodeAnalysisAgent:
    """Agent for analyzing code using natural language queries."""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, openai_api_key=None):
        """Initialize the agent with database connection parameters."""
        self.retriever = LLMRetriever(neo4j_uri, neo4j_user, neo4j_password, openai_api_key)
        
        # Create the graph
        self.graph_executor = create_code_analysis_graph(self.retriever)
    
    def analyze(self, question: str) -> str:
        """Analyze code based on a natural language question."""
        try:
            logger.info(f"Starting analysis for question: {question}")
            
            initial_state = {
                "user_question": question,
                "retrieval_results": {},
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
        """Close the database connection."""
        self.retriever.close()

###############################################################################
# Command Line Interface
###############################################################################
def main():
    """Command line interface for the code analysis agent."""
    import os
    from dotenv import load_dotenv, find_dotenv
    
    # Load environment variables
    load_dotenv(find_dotenv(), override=True)
    
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    # Initialize the agent
    agent = CodeAnalysisAgent(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)
    
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