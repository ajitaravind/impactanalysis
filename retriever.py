import os
import logging
import re
from typing import List, Dict, Any, Tuple, Set
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
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
                "Focus on technical terms and code-related concepts. "
                "Be comprehensive and include variations of terms (e.g., both camelCase and snake_case versions). "
                "Include both singular and plural forms where appropriate. "
                "Break down compound terms into their components."
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
                
                Use case-insensitive matching with toLower() for more flexible matching.
                Consider both exact matches and partial matches using CONTAINS.
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

    def expand_query_terms(self, elements: List[str]) -> List[str]:
        """
        Expand query terms with synonyms and related terms to improve search coverage.
        Uses an LLM to dynamically generate expansions based on the context.
        
        Args:
            elements: List of extracted code elements
            
        Returns:
            Expanded list of terms
        """
        if not elements:
            return []
            
        expanded_terms = set(elements)
        
        # Create a simple prompt for the LLM to generate term expansions
        class TermExpansions(BaseModel):
            """Expanded programming terms and related concepts."""
            
            expanded_terms: List[str] = Field(
                description="List of expanded programming terms and related concepts that are similar to or related to the input terms"
            )
        
        try:
            # Create a simple prompt
            expansion_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are an expert in programming languages and code analysis.
                    Your task is to expand the given code-related terms with synonyms, related concepts,
                    different naming conventions, and variations that might be used in code.
                    
                    For each term, consider:
                    1. Common synonyms in programming contexts
                    2. Related programming concepts
                    3. Different naming conventions (camelCase, snake_case, PascalCase)
                    4. Singular/plural forms
                    5. Verb/noun variations
                    
                    Return ONLY a list of expanded terms without any explanations or categorization.
                    """
                ),
                (
                    "human",
                    "Generate expanded terms for these code elements: {elements}"
                ),
            ])
            
            # Create the chain with structured output - just one LLM call
            expansion_chain = expansion_prompt | self.llm.with_structured_output(TermExpansions, method="function_calling")
            
            # Get expansions from the LLM
            result = expansion_chain.invoke({"elements": elements})
            
            # Simply add all the expanded terms to our set
            expanded_terms.update(result.expanded_terms)
            
            logger.info(f"LLM expanded terms: {result.expanded_terms}")
        
        except Exception as e:
            logger.error(f"Error in LLM term expansion: {str(e)}")
            logger.info("Falling back to original terms only")
        
        logger.info(f"Final expanded terms: {expanded_terms}")
        return list(expanded_terms)

    def explore_relationships(self, initial_results: List[Dict]) -> List[Tuple[str, Dict]]:
        """
        Generate queries to explore relationships between found components.
        
        Args:
            initial_results: Results from the initial queries
            
        Returns:
            List of (query, params) tuples for relationship exploration
        """
        relationship_queries = []
        
        # For each function found
        for result in initial_results:
            if 'name' in result and result.get('type') == 'Function':
                function_name = result['name']
                
                # 1. Find functions that call this function
                callers_query = """
                MATCH (caller:Function)-[:CALLS]->(callee:Function {name: $function_name})
                RETURN caller.name as name, caller.docstring as docstring,
                       'Function' as type, caller.start_line as start_line,
                       caller.end_line as end_line, 'CALLS' as relationship_type,
                       $function_name as target_name, 'caller' as role
                """
                relationship_queries.append((callers_query, {"function_name": function_name}))
                
                # 2. Find functions that this function calls
                callees_query = """
                MATCH (caller:Function {name: $function_name})-[:CALLS]->(callee:Function)
                RETURN callee.name as name, callee.docstring as docstring,
                       'Function' as type, callee.start_line as start_line,
                       callee.end_line as end_line, 'IS_CALLED_BY' as relationship_type,
                       $function_name as source_name, 'callee' as role
                """
                relationship_queries.append((callees_query, {"function_name": function_name}))
                
                # 3. Find the class that contains this function (if it's a method)
                container_query = """
                MATCH (c:Class)-[:DEFINES]->(f:Function {name: $function_name})
                RETURN c.name as name, c.docstring as docstring,
                       'Class' as type, c.start_line as start_line,
                       c.end_line as end_line, 'CONTAINS_METHOD' as relationship_type,
                       $function_name as method_name, 'container' as role
                """
                relationship_queries.append((container_query, {"function_name": function_name}))
            
            # For each class found
            elif 'name' in result and result.get('type') == 'Class':
                class_name = result['name']
                
                # 1. Find methods defined by this class
                methods_query = """
                MATCH (c:Class {name: $class_name})-[:DEFINES]->(f:Function)
                RETURN f.name as name, f.docstring as docstring,
                       'Function' as type, f.start_line as start_line,
                       f.end_line as end_line, 'DEFINED_BY' as relationship_type,
                       $class_name as class_name, 'method' as role
                """
                relationship_queries.append((methods_query, {"class_name": class_name}))
                
                # 2. Find parent classes (inheritance)
                parent_query = """
                MATCH (c:Class {name: $class_name})-[:INHERITS_FROM]->(parent:Class)
                RETURN parent.name as name, parent.docstring as docstring,
                       'Class' as type, parent.start_line as start_line,
                       parent.end_line as end_line, 'INHERITS_FROM' as relationship_type,
                       $class_name as child_name, 'parent' as role
                """
                relationship_queries.append((parent_query, {"class_name": class_name}))
                
                # 3. Find child classes (inheritance)
                child_query = """
                MATCH (child:Class)-[:INHERITS_FROM]->(c:Class {name: $class_name})
                RETURN child.name as name, child.docstring as docstring,
                       'Class' as type, child.start_line as start_line,
                       child.end_line as end_line, 'INHERITED_BY' as relationship_type,
                       $class_name as parent_name, 'child' as role
                """
                relationship_queries.append((child_query, {"class_name": class_name}))
            
            # For class-method pairs
            elif 'class_name' in result and 'method_name' in result:
                class_name = result['class_name']
                method_name = result['method_name']
                
                # Find other methods in the same class
                related_methods_query = """
                MATCH (c:Class {name: $class_name})-[:DEFINES]->(f:Function)
                WHERE f.name <> $method_name
                RETURN f.name as name, f.docstring as docstring,
                       'Function' as type, f.start_line as start_line,
                       f.end_line as end_line, 'RELATED_METHOD' as relationship_type,
                       $class_name as class_name, $method_name as related_to_method, 'related_method' as role
                """
                relationship_queries.append((related_methods_query, {
                    "class_name": class_name,
                    "method_name": method_name
                }))
                
                # Find functions called by this method
                method_calls_query = """
                MATCH (c:Class {name: $class_name})-[:DEFINES]->(f:Function {name: $method_name})-[:CALLS]->(called:Function)
                RETURN called.name as name, called.docstring as docstring,
                       'Function' as type, called.start_line as start_line,
                       called.end_line as end_line, 'CALLED_BY_METHOD' as relationship_type,
                       $class_name as class_name, $method_name as method_name, 'called_by_method' as role
                """
                relationship_queries.append((method_calls_query, {
                    "class_name": class_name,
                    "method_name": method_name
                }))
        
        return relationship_queries

    def generate_multi_stage_queries(self, query_string: str, elements: List[str]) -> List[Tuple[str, Dict]]:
        """
        Generate multiple Cypher queries with different strategies.
        
        Args:
            query_string: The original query string
            elements: Extracted and expanded code elements
            
        Returns:
            List of (query, params) tuples
        """
        queries = []
        
        # 1. Direct name matches for functions
        function_name_query = """
        MATCH (f:Function)
        WHERE toLower(f.name) CONTAINS toLower($term)
        RETURN f.name as name, f.docstring as docstring, 
               'Function' as type, f.start_line as start_line, f.end_line as end_line
        """
        
        # 2. Direct name matches for classes
        class_name_query = """
        MATCH (c:Class)
        WHERE toLower(c.name) CONTAINS toLower($term)
        RETURN c.name as name, c.docstring as docstring, 
               'Class' as type, c.start_line as start_line, c.end_line as end_line
        """
        
        # 3. Docstring content matches
        docstring_query = """
        MATCH (n)
        WHERE (n:Function OR n:Class) AND toLower(n.docstring) CONTAINS toLower($term)
        RETURN n.name as name, n.docstring as docstring, 
               CASE WHEN n:Function THEN 'Function' ELSE 'Class' END as type,
               n.start_line as start_line, n.end_line as end_line
        """
        
        # 4. Class methods (for more specific queries)
        class_method_query = """
        MATCH (c:Class)-[:DEFINES]->(f:Function)
        WHERE toLower(c.name) CONTAINS toLower($class_term) AND toLower(f.name) CONTAINS toLower($method_term)
        RETURN c.name as class_name, f.name as method_name, f.docstring as docstring,
               f.start_line as start_line, f.end_line as end_line
        """
        
        # Add single-term queries
        for term in elements:
            queries.append((function_name_query, {"term": term}))
            queries.append((class_name_query, {"term": term}))
            queries.append((docstring_query, {"term": term}))
        
        # Add class-method combinations
        for i, term1 in enumerate(elements):
            for term2 in elements[i+1:]:
                # Try both combinations of class/method
                queries.append((class_method_query, {"class_term": term1, "method_term": term2}))
                queries.append((class_method_query, {"class_term": term2, "method_term": term1}))
        
        # Also include the original LLM-generated query as a fallback
        try:
            original_query = self.query_chain.invoke({
                "question": query_string,
                "elements": elements
            })
            
            # Clean up the query
            if "```" in original_query:
                match = re.search(r'```(?:cypher)?\s*([\s\S]*?)```', original_query)
                if match:
                    original_query = match.group(1).strip()
                else:
                    lines = original_query.split('\n')
                    filtered_lines = [line for line in lines if not line.strip().startswith('```')]
                    original_query = '\n'.join(filtered_lines).strip()
            
            queries.append((original_query, {}))
            
        except Exception as e:
            logger.error(f"Error generating original query: {e}")
        
        return queries

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
        
        # Step 2: Expand query terms
        expanded_elements = self.expand_query_terms(code_elements.elements)
        
        # Step 3: Generate multiple Cypher queries
        query_candidates = self.generate_multi_stage_queries(query_string, expanded_elements)
        
        # Step 4: Execute queries until we find results
        all_results = []
        successful_query = None
        
        for query, params in query_candidates:
            try:
                with self.driver.session() as session:
                    if params:  # Parameterized query
                        records = session.run(query, params)
                    else:  # Direct query
                        records = session.run(query)
                    
                    results = [dict(record) for record in records]
                    
                    if results:
                        all_results.extend(results)
                        if not successful_query:
                            successful_query = query
                            logger.info(f"Found results with query: {query}")
                            
                            # If we have enough results, we can stop
                            if len(all_results) >= 10:
                                break
            except Exception as e:
                logger.error(f"Error executing query {query}: {e}")
        
        # Step 5: If we found some results, explore relationships
        if all_results:
            logger.info("Exploring relationships between found components")
            relationship_queries = self.explore_relationships(all_results)
            
            # Execute relationship queries
            for query, params in relationship_queries:
                try:
                    with self.driver.session() as session:
                        records = session.run(query, params)
                        results = [dict(record) for record in records]
                        
                        if results:
                            logger.info(f"Found {len(results)} relationship results")
                            all_results.extend(results)
                except Exception as e:
                    logger.error(f"Error executing relationship query: {e}")
        
        # Step 6: Deduplicate results based on name and type
        unique_results = []
        seen = set()
        
        for result in all_results:
            # Create a key for deduplication
            if 'name' in result and 'type' in result:
                key = (result['name'], result['type'])
            elif 'class_name' in result and 'method_name' in result:
                key = (result['class_name'], result['method_name'])
            else:
                # If we can't create a key, just add the result
                unique_results.append(result)
                continue
                
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        logger.info(f"Found {len(unique_results)} unique results after relationship exploration")
        
        # Step 7: Generate a natural language response
        try:
            # Enhance the response prompt to include relationship information
            enhanced_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are a code analysis expert. Using the provided database results:
                    1. Focus on technical accuracy and implementation details
                    2. Explain code relationships and structure clearly
                    3. Cite specific examples from the retrieved data
                    4. Keep explanations concise but complete
                    5. Pay special attention to relationships between components (calls, contains, inherits)
                    6. Organize your response to show how components interact in the system
                    """
                ),
                (
                    "human",
                    """Question: {question}
                    
                    Database results: {results}
                    
                    Please provide a comprehensive analysis of this code, focusing on both individual components
                    and how they relate to each other:"""
                ),
            ])
            
            # Create enhanced response chain
            enhanced_response_chain = enhanced_prompt | self.llm | StrOutputParser()
            
            response = enhanced_response_chain.invoke({
                "question": query_string,
                "results": unique_results
            })
            logger.info("Generated response successfully")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response = "I encountered an error while analyzing the code database. Please try a different query."
        
        # Return the complete result
        return {
            "query": query_string,
            "cypher_query": successful_query or "No successful query",
            "raw_results": unique_results,
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