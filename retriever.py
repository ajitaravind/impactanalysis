import os
import logging
from typing import List, Dict, Any
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # Make sure this is set in your .env file


class CodeElements(BaseModel):
    """Identifying information about code elements."""
    
    elements: List[str] = Field(
        ...,
        description="All the code elements (functions, classes, variables, or modules) "
        "that appear in the text",
    )


class LLMRetriever:
    """
    An LLM-enhanced retriever for a Neo4j code database.
    """

    def __init__(self, uri, user, password, openai_api_key=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize LLM components
        self.llm = ChatOpenAI(
            temperature=0, 
            model_name="gpt-4o-mini",  # You can use a smaller model to reduce costs
            api_key=openai_api_key
        )
        
        # Entity extraction prompt
        self.entity_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting code elements like functions, classes, variables, and modules from the text. "
                "Focus on technical terms and code-related concepts.",
            ),
            (
                "human",
                "Use the given format to extract code elements from the following input: {question}",
            ),
        ])
        
        # Create entity extraction chain
        self.entity_chain = self.entity_prompt | self.llm.with_structured_output(CodeElements)
        
        # Query planning prompt
        self.query_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert at converting natural language questions about code into Neo4j Cypher queries.
                Based on the user's question and the extracted code elements, generate the appropriate Cypher query.
                
                Here are the node types and relationships in our database:
                - File nodes with properties: path
                - Function nodes with properties: name, start_line, end_line, docstring
                - Class nodes with properties: name, start_line, end_line, docstring
                - Comment nodes with properties: text
                
                Relationships:
                - (File)-[:CONTAINS]->(Function)
                - (File)-[:CONTAINS]->(Class)
                - (File)-[:HAS_COMMENT]->(Comment)
                - (Class)-[:DEFINES]->(Function) (for methods)
                - (Function)-[:CALLS]->(Function)
                - (Class)-[:CALLS]->(Function)
                
                IMPORTANT: Return only the raw Cypher query without any markdown formatting, code blocks, or explanations.
                Do not include ```cypher or ``` in your response.
                """
            ),
            (
                "human",
                """Question: {question}
                
                Extracted code elements: {elements}
                
                Generate a Cypher query to answer this question:"""
            ),
        ])
        
        # Create query planning chain
        self.query_chain = self.query_prompt | self.llm | StrOutputParser()
        
        # Response generation prompt
        self.response_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a code analysis expert. Using the provided database results:
                1. Focus on technical accuracy and implementation details
                2. Explain code relationships and structure clearly
                3. Cite specific examples from the retrieved data
                4. Keep explanations concise but complete
                """
            ),
            (
                "human",
                """Question: {question}
                
                Database results: {results}
                
                Please provide a clear, technical answer to the question:"""
            ),
        ])
        
        # Create response generation chain
        self.response_chain = self.response_prompt | self.llm | StrOutputParser()

    def retrieve(self, query_string: str) -> Dict[str, Any]:
        """
        Retrieves information from the Neo4j database based on a natural language query.
        
        Args:
            query_string: The natural language query.
            
        Returns:
            A dictionary containing the original query, the generated Cypher query,
            the raw database results, and a natural language response.
        """
        logger.info(f"Processing query: {query_string}")
        
        # Step 1: Extract code elements from the query
        try:
            code_elements = self.entity_chain.invoke({"question": query_string})
            logger.info(f"Extracted code elements: {code_elements.elements}")
        except Exception as e:
            logger.error(f"Error extracting code elements: {e}")
            code_elements = CodeElements(elements=[])
        
        # Step 2: Generate a Cypher query
        try:
            raw_cypher = self.query_chain.invoke({
                "question": query_string,
                "elements": code_elements.elements
            })
            
            # Clean up the Cypher query by removing markdown formatting
            cypher_query = raw_cypher
            # Remove markdown code block syntax if present
            if "```" in cypher_query:
                # Extract content between triple backticks
                import re
                match = re.search(r'```(?:cypher)?\s*([\s\S]*?)```', cypher_query)
                if match:
                    cypher_query = match.group(1).strip()
                else:
                    # If regex fails, just remove the backticks lines
                    lines = cypher_query.split('\n')
                    filtered_lines = [line for line in lines if not line.strip().startswith('```')]
                    cypher_query = '\n'.join(filtered_lines).strip()
            
            logger.info(f"Generated Cypher query: {cypher_query}")
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            # Fallback to a simple query if generation fails
            cypher_query = "MATCH (n) RETURN n LIMIT 10"
        
        # Step 3: Execute the Cypher query
        raw_results = []
        try:
            with self.driver.session() as session:
                records = session.run(cypher_query)
                raw_results = [dict(record) for record in records]
            logger.info(f"Query returned {len(raw_results)} results")
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
        
        # Step 4: Generate a natural language response
        try:
            response = self.response_chain.invoke({
                "question": query_string,
                "results": raw_results
            })
            logger.info("Generated response successfully")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = "I encountered an error while analyzing the code database. Please try a different query."
        
        # Return the complete result
        return {
            "query": query_string,
            "cypher_query": cypher_query,
            "raw_results": raw_results,
            "response": response
        }

    def close(self):
        self.driver.close()


def main():
    # Initialize the retriever
    retriever = LLMRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)

    while True:
        query = input("Enter your query (or 'exit'): ")
        if query.lower() == "exit":
            break

        result = retriever.retrieve(query)

        print("\n" + "="*50)
        print("NATURAL LANGUAGE RESPONSE:")
        print("-"*40)
        print(result["response"])
        
        print("\n" + "="*50)
        print("CYPHER QUERY USED:")
        print("-"*40)
        print(result["cypher_query"])
        
        print("\n" + "="*50)
        print("RAW DATABASE RESULTS:")
        print("-"*40)
        print(f"Found {len(result['raw_results'])} results")
        if result['raw_results']:
            for i, item in enumerate(result['raw_results'][:5]):  # Show first 5 results
                print(f"\nResult {i+1}:")
                print(item)
            if len(result['raw_results']) > 5:
                print(f"\n... and {len(result['raw_results']) - 5} more results")
    
    retriever.close()

if __name__ == "__main__":
    main()