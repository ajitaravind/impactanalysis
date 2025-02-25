import os
import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from neo4j import GraphDatabase
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel, Field

# Import VectorRetriever
from vector_retriever import VectorRetriever

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

    def __init__(self, uri, user, password, openai_api_key=None, 
                 vector_retriever: Optional[VectorRetriever] = None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize vector retriever
        self.vector_retriever = vector_retriever
        
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
        Expand query terms with a small number of high-quality synonyms.
        Limits total expansion to at most 5 terms regardless of input size.
        
        Args:
            elements: List of extracted code elements
            
        Returns:
            Expanded list of terms with original elements prioritized
        """
        if not elements:
            return []
        
        # If we already have 5 or more terms, just use those (prioritize original terms)
        if len(elements) >= 5:
            logger.info(f"Using only original terms (already have {len(elements)} terms)")
            return elements[:5]  # Take at most 5 original terms
        
        # We need to expand, but only enough to get to 5 total terms
        num_expansions_needed = min(5 - len(elements), 5)  # At most 5 expansions
        expanded_terms = set(elements)  # Start with original terms
        
        if num_expansions_needed <= 0:
            return list(expanded_terms)
        
        # Create a simple prompt for the LLM to generate term expansions
        class TermExpansions(BaseModel):
            """Expanded programming terms and related concepts."""
            
            expanded_terms: List[str] = Field(
                description=f"List of exactly {num_expansions_needed} expanded programming terms"
            )
        
        try:
            # Create a simple prompt
            expansion_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are an expert in programming languages and code analysis.
                    Your task is to expand the given code-related terms with the MOST RELEVANT synonyms
                    or related concepts that would be useful for searching code.
                    
                    Return EXACTLY {num_expansions_needed} terms total - not per input term, but in total.
                    Focus on terms that would most likely appear in actual code or documentation.
                    """
                ),
                (
                    "human",
                    "Generate exactly {num_expansions} expanded terms for these code elements: {elements}"
                ),
            ])
            
            # Create the chain with structured output
            expansion_chain = expansion_prompt | self.llm.with_structured_output(TermExpansions, method="function_calling")
            
            # Get expansions from the LLM
            result = expansion_chain.invoke({
                "elements": elements,
                "num_expansions": num_expansions_needed
            })
            
            # Add the expanded terms to our set
            expanded_terms.update(result.expanded_terms[:num_expansions_needed])
            
            logger.info(f"LLM expanded terms: {result.expanded_terms[:num_expansions_needed]}")
        
        except Exception as e:
            logger.error(f"Error in LLM term expansion: {str(e)}")
            logger.info("Falling back to original terms only")
        
        logger.info(f"Final expanded terms: {expanded_terms}")
        return list(expanded_terms)[:5]  # Final safety check - never return more than 5 terms

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
        Generate a limited set of targeted Cypher queries.
        
        Args:
            query_string: The original query string
            elements: Extracted and expanded code elements (limited to 5 max)
            
        Returns:
            List of (query, params) tuples
        """
        queries = []
        
        # 1. Combined query for functions and classes (more efficient)
        combined_query = """
        MATCH (n)
        WHERE (n:Function OR n:Class) AND toLower(n.name) CONTAINS toLower($term)
        RETURN n.name as name, n.docstring as docstring, 
               CASE WHEN n:Function THEN 'Function' ELSE 'Class' END as type,
               n.start_line as start_line, n.end_line as end_line
        """
        
        # 2. Docstring content query (for semantic matches)
        docstring_query = """
        MATCH (n)
        WHERE (n:Function OR n:Class) AND toLower(n.docstring) CONTAINS toLower($term)
        RETURN n.name as name, n.docstring as docstring, 
               CASE WHEN n:Function THEN 'Function' ELSE 'Class' END as type,
               n.start_line as start_line, n.end_line as end_line
        """
        
        # Add queries for each term (but keep it limited)
        for term in elements:
            queries.append((combined_query, {"term": term}))
            
            # Only do docstring queries for key terms to reduce query count
            if len(term) > 3:  # Only for non-trivial terms
                queries.append((docstring_query, {"term": term}))
        
        # Only generate class-method combinations if we have few terms
        if len(elements) <= 3:
            # 3. Class methods (for more specific queries)
            class_method_query = """
            MATCH (c:Class)-[:DEFINES]->(f:Function)
            WHERE toLower(c.name) CONTAINS toLower($class_term) AND toLower(f.name) CONTAINS toLower($method_term)
            RETURN c.name as class_name, f.name as method_name, f.docstring as docstring,
                   f.start_line as start_line, f.end_line as end_line
            """
            
            # Add class-method combinations (but only for the first few terms)
            for i, term1 in enumerate(elements[:2]):  # Only use first 2 terms
                for term2 in elements[i+1:3]:  # Only combine with next term up to 3rd term
                    queries.append((class_method_query, {"class_term": term1, "method_term": term2}))
        
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
        return queries

    def retrieve(self, query_string: str) -> Dict[str, Any]:
        """
        Retrieves information from the Neo4j database based on a natural language query.
        Also performs vector search if a vector retriever is available.
        
        Args:
            query_string: The natural language query.
            
        Returns:
            A dictionary containing the original query, the generated Cypher query,
            the raw database results, and a natural language response.
        """
        logger.info(f"Processing query: {query_string}")
        
        # Step 1: Perform vector search if available
        vector_results = []
        if self.vector_retriever:
            try:
                logger.info("Performing vector search with VectorRetriever")
                # Add more detailed logging
                logger.info(f"Vector retriever configuration: index={self.vector_retriever.index_name}, namespace={self.vector_retriever.namespace}")
                
                vector_search_results = self.vector_retriever.retrieve(query_string)
                logger.info(f"Vector search completed with status: {'success' if vector_search_results else 'no results'}")
                
                # Extract the raw results from the vector search
                if vector_search_results and "raw_results" in vector_search_results:
                    logger.info(f"Vector search returned {len(vector_search_results['raw_results'])} raw results")
                    for result in vector_search_results["raw_results"]:
                        # Convert vector results to a format compatible with Neo4j results
                        formatted_result = {
                            "name": result["metadata"].get("code_name", "Unknown"),
                            "type": result["metadata"].get("code_type", "Code"),
                            "content": result["content"],
                            "source": result["metadata"].get("source", "Unknown"),
                            "relevance_score": result.get("relevance_score", 0),
                            "is_vector_result": True  # Flag to identify vector results
                        }
                        
                        # Add line numbers if available
                        if "start_line" in result["metadata"]:
                            formatted_result["start_line"] = result["metadata"]["start_line"]
                        if "end_line" in result["metadata"]:
                            formatted_result["end_line"] = result["metadata"]["end_line"]
                            
                        vector_results.append(formatted_result)
                    
                    logger.info(f"Processed {len(vector_results)} vector search results")
                else:
                    logger.warning("Vector search returned no usable results")
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}", exc_info=True)  # Add full traceback
        else:
            logger.warning("Vector retriever not available, skipping vector search")
        
        # Step 2: Extract code elements from the query
        try:
            code_elements = self.entity_chain.invoke({"question": query_string})
            logger.info(f"Extracted code elements: {code_elements.elements}")
        except Exception as e:
            logger.error(f"Error extracting code elements: {e}")
            code_elements = CodeElements(elements=[])
        
        # Step 3: Expand query terms
        expanded_elements = self.expand_query_terms(code_elements.elements)
        
        # Step 4: Generate multiple Cypher queries
        query_candidates = self.generate_multi_stage_queries(query_string, expanded_elements)
        
        # Step 5: Execute queries until we find results
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
        
        # Step 6: If we found some results, explore relationships
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
        
        # Step 7: Add vector search results to all_results
        if vector_results:
            all_results.extend(vector_results)
            logger.info(f"Added {len(vector_results)} vector search results to all results")
        
        # Step 8: Deduplicate results based on name and type
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
        
        logger.info(f"Found {len(unique_results)} unique results after deduplication")
        
        # Step 9: Generate a natural language response
        try:
            # Enhance the response prompt to include relationship information and vector results
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
                    7. Incorporate both structured database results and vector search results in your analysis
                    8. For vector search results, focus on the code content and its relevance to the query
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
            "response": response,
            "has_vector_results": len(vector_results) > 0
        }

    def close(self):
        self.driver.close()


def main():
    # Initialize the retriever
    retriever = LLMRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)

    # Optionally initialize vector retriever
    vector_retriever = None
    try:
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
        
        if PINECONE_API_KEY and PINECONE_ENVIRONMENT:
            vector_retriever = VectorRetriever(
                openai_api_key=OPENAI_API_KEY,
                pinecone_api_key=PINECONE_API_KEY,
                pinecone_environment=PINECONE_ENVIRONMENT
            )
            retriever.vector_retriever = vector_retriever
            print("Vector retriever initialized successfully")
    except Exception as e:
        print(f"Failed to initialize vector retriever: {e}")

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
    if vector_retriever:
        vector_retriever.close()

if __name__ == "__main__":
    main()