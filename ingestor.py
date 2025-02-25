import os
import logging
from tree_sitter import Language, Parser, Query
import tree_sitter_python
from neo4j import GraphDatabase

from dotenv import load_dotenv, find_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)

NEO4J_URI=os.environ.get("NEO4J_URI")
NEO4J_USER=os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD=os.environ.get("NEO4J_PASSWORD")
# Use raw string (r prefix) to prevent escape sequence interpretation
CODE_DIRECTORY = r"C:\Users\HP\Documents\Personal Projects and Learnings\AI Related\treesitterproject\sample\sample.py"

logger.info(f"Neo4j URI: {NEO4J_URI}")
logger.info(f"Neo4j User: {NEO4J_USER}")
logger.info(f"Code Directory: {CODE_DIRECTORY}")

# --- Configuration ---
# NEO4J_URI = "bolt://localhost:7687"  # Replace with your Neo4j URI
# NEO4J_USER = "neo4j"  # Replace with your Neo4j username
# NEO4J_PASSWORD = "your_password"  # Replace with your Neo4j password
# CODE_DIRECTORY = "your_code_directory"  # Replace with the path to your code

# --- Tree-sitter Setup ---
PY_LANGUAGE = Language(tree_sitter_python.language())
parser = Parser(PY_LANGUAGE)

# --- Tree-sitter Queries ---
# We define queries to extract the necessary information
QUERIES = {
    "file": """
        (module) @file
    """,
    "functions": """
        (function_definition
            name: (identifier) @function_name
            body: (block) @function_body
            .
            (string)? @docstring  
        ) @function
    """,
    "classes": """
        (class_definition
            name: (identifier) @class_name
            body: (block) @class_body
            .
            (string)? @docstring
        ) @class
    """,
        "comments": """
        (comment) @comment_text
    """,
    "calls": """
        (call
            function: (identifier) @called_function_name
        ) @function_call
    """,
}

# --- Helper Functions ---

def execute_query(tree, query_str):
    """Executes a Tree-sitter query and returns the captures."""
    query = PY_LANGUAGE.query(query_str)
    
    # Get matches
    matches = query.matches(tree.root_node)
    
    # Debug the structure
    if matches:
        logger.info(f"Match example: {matches[0]}")
    
    # Process matches to extract node-capture pairs
    result = []
    for match in matches:
        # Each match is a tuple (index, capture_dict)
        # The capture_dict maps capture names to lists of nodes
        _, capture_dict = match
        
        for capture_name, nodes in capture_dict.items():
            for node in nodes:
                result.append((node, capture_name))
    
    return result

def get_node_text(node):
    """Gets the text content of a Tree-sitter node."""
    return node.text.decode()

def get_docstring(captures, body_node):
  for node, type in captures:
    if node.parent == body_node.parent:
      if type == "docstring":
        return node.text.decode()
  return ""

# --- Neo4j Interaction ---

def create_file_node(tx, file_path):
    """Creates a File node in Neo4j."""
    logger.info(f"Creating File node for: {file_path}")
    result = tx.run("CREATE (f:File {path: $path}) RETURN f", path=file_path)
    summary = result.consume()
    logger.info(f"Created {summary.counters.nodes_created} file node(s)")

def create_function_node(tx, file_path, function_name, start_line, end_line, docstring):
    """Creates a Function node and connects it to its File."""
    logger.info(f"Creating Function node: {function_name} in {file_path}")
    result = tx.run("""
        MERGE (f:File {path: $file_path})
        CREATE (func:Function {name: $name, start_line: $start_line, end_line: $end_line, docstring: $docstring})
        CREATE (f)-[:CONTAINS]->(func)
        RETURN func
    """, file_path=file_path, name=function_name, start_line=start_line,
           end_line=end_line, docstring=docstring)
    summary = result.consume()
    logger.info(f"Created {summary.counters.nodes_created} function node(s) and {summary.counters.relationships_created} relationship(s)")

def create_class_node(tx, file_path, class_name, start_line, end_line, docstring):
    """Creates a Class node and connects it to its File."""
    logger.info(f"Creating Class node: {class_name} in {file_path}")
    result = tx.run("""
        MERGE (f:File {path: $file_path})
        CREATE (cls:Class {name: $name, start_line: $start_line, end_line: $end_line, docstring: $docstring})
        CREATE (f)-[:CONTAINS]->(cls)
        RETURN cls
    """, file_path=file_path, name=class_name, start_line=start_line,
           end_line=end_line, docstring=docstring)
    summary = result.consume()
    logger.info(f"Created {summary.counters.nodes_created} class node(s) and {summary.counters.relationships_created} relationship(s)")

def create_comment_node(tx, file_path, comment_text):
    logger.info(f"Creating Comment node in {file_path}")
    result = tx.run("""
        MERGE (f:File {path: $file_path})
        CREATE (c:Comment {text: $text})
        CREATE (f)-[:HAS_COMMENT]->(c)
        RETURN c
    """, file_path=file_path, text=comment_text)
    summary = result.consume()
    logger.info(f"Created {summary.counters.nodes_created} comment node(s) and {summary.counters.relationships_created} relationship(s)")

def create_method_node(tx, file_path, class_name, method_name, start_line, end_line, docstring):
    """Creates a Method node, connects it to its Class, and its File"""
    logger.info(f"Creating Method node: {method_name} in class {class_name} in {file_path}")
    result = tx.run("""
        MERGE (f:File {path: $file_path})
        MERGE (c:Class {name: $class_name})
        CREATE (m:Function {name: $method_name, start_line: $start_line, end_line: $end_line, docstring: $docstring})
        CREATE (f)-[:CONTAINS]->(m)
        CREATE (c)-[:DEFINES]->(m)
        RETURN m
    """, file_path=file_path, class_name=class_name, method_name=method_name, start_line=start_line,
           end_line=end_line, docstring=docstring)
    summary = result.consume()
    logger.info(f"Created {summary.counters.nodes_created} method node(s) and {summary.counters.relationships_created} relationship(s)")
    return result

def create_call_relationship(tx, caller_start, caller_end, called_function_name):
    """
        Create call relationship between functions
    """
    logger.info(f"Creating CALLS relationship to function: {called_function_name}")
    
    # First, check if the callee function exists
    result_check = tx.run("""
        MATCH (callee:Function {name: $callee_name})
        RETURN count(callee) as count
    """, callee_name=called_function_name)
    
    callee_count = result_check.single()["count"]
    if callee_count == 0:
        logger.warning(f"No function named '{called_function_name}' found in database. Creating placeholder.")
        # Create a placeholder function node
        tx.run("""
            CREATE (f:Function {name: $name, is_placeholder: true})
        """, name=called_function_name)
    
    # Now create the relationship
    result1 = tx.run("""
        MATCH (caller:Function) WHERE $caller_start >= caller.start_line AND $caller_start <= caller.end_line
        MERGE (callee:Function {name: $callee_name})
        MERGE (caller)-[:CALLS]->(callee)
        RETURN caller, callee
    """, caller_start=caller_start, caller_end=caller_end, callee_name=called_function_name)
    summary1 = result1.consume()
    
    result2 = tx.run("""
        MATCH (caller:Class) WHERE $caller_start >= caller.start_line AND $caller_start <= caller.end_line
        MERGE (callee:Function {name: $callee_name})
        MERGE (caller)-[:CALLS]->(callee)
        RETURN caller, callee
    """, caller_start=caller_start, caller_end=caller_end, callee_name=called_function_name)
    summary2 = result2.consume()
    
    logger.info(f"Created {summary1.counters.relationships_created + summary2.counters.relationships_created} CALLS relationship(s)")

def process_file(tx, file_path):
    """Parses a file, extracts information, and creates nodes/relationships."""
    logger.info(f"Processing file: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            logger.info(f"Successfully read file: {file_path}, size: {len(code)} bytes")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return

    tree = parser.parse(bytes(code, "utf8"))
    logger.info(f"Successfully parsed file: {file_path}")

    # Create File Node
    create_file_node(tx, file_path)

    # Extract and Create Functions
    function_captures = execute_query(tree, QUERIES["functions"])
    logger.info(f"Found {len(function_captures)} function-related captures in {file_path}")
    
    # Process function captures to extract actual functions
    function_nodes = {}
    for capture in function_captures:
        logger.info(f"Function capture: {type(capture)}, {capture}")
        if isinstance(capture, tuple) and len(capture) >= 2:
            node, capture_type = capture
            if capture_type == "function":
                function_nodes[node] = {"node": node, "name": "", "body": None, "docstring": ""}
    
    logger.info(f"Found {len(function_nodes)} function nodes")
    
    # Extract function details
    for capture in function_captures:
        if isinstance(capture, tuple) and len(capture) >= 2:
            node, capture_type = capture
            parent = node.parent
            if parent in function_nodes:
                if capture_type == "function_name":
                    function_nodes[parent]["name"] = get_node_text(node)
                elif capture_type == "function_body":
                    function_nodes[parent]["body"] = node
                elif capture_type == "docstring":
                    function_nodes[parent]["docstring"] = get_node_text(node)
    
    # Create function nodes
    functions_processed = 0
    for func_node, func_data in function_nodes.items():
        # Skip methods (will be processed later)
        if func_node.parent.type == "block" and func_node.parent.parent.type == "class_definition":
            continue
            
        start_line = func_node.start_point[0] + 1
        end_line = func_node.end_point[0] + 1
        create_function_node(tx, file_path, func_data["name"], start_line, end_line, func_data["docstring"])
        functions_processed += 1
    
    logger.info(f"Processed {functions_processed} standalone functions")

    # Extract and Create Classes
    class_captures = execute_query(tree, QUERIES["classes"])
    logger.info(f"Found {len(class_captures)} class-related captures in {file_path}")
    
    # Process class captures to extract actual classes
    class_nodes = {}
    for capture in class_captures:
        if isinstance(capture, tuple) and len(capture) >= 2:
            node, capture_type = capture
            if capture_type == "class":
                class_nodes[node] = {"node": node, "name": "", "body": None, "docstring": ""}
    
    logger.info(f"Found {len(class_nodes)} class nodes")
    
    # Extract class details
    for capture in class_captures:
        if isinstance(capture, tuple) and len(capture) >= 2:
            node, capture_type = capture
            parent = node.parent
            if parent in class_nodes:
                if capture_type == "class_name":
                    class_nodes[parent]["name"] = get_node_text(node)
                elif capture_type == "class_body":
                    class_nodes[parent]["body"] = node
                elif capture_type == "docstring":
                    class_nodes[parent]["docstring"] = get_node_text(node)
    
    # Create class nodes
    classes_processed = 0
    for class_node, class_data in class_nodes.items():
        start_line = class_node.start_point[0] + 1
        end_line = class_node.end_point[0] + 1
        create_class_node(tx, file_path, class_data["name"], start_line, end_line, class_data["docstring"])
        classes_processed += 1
    
    logger.info(f"Processed {classes_processed} classes")

    # Process methods (functions inside classes)
    methods_processed = 0
    for func_node, func_data in function_nodes.items():
        if func_node.parent.type == "block" and func_node.parent.parent.type == "class_definition":
            class_name = get_node_text(func_node.parent.parent.child_by_field_name("name"))
            start_line = func_node.start_point[0] + 1
            end_line = func_node.end_point[0] + 1
            create_method_node(tx, file_path, class_name, func_data["name"], start_line, end_line, func_data["docstring"])
            methods_processed += 1
    
    logger.info(f"Processed {methods_processed} methods")

    # Extract and Create Comments
    comment_captures = execute_query(tree, QUERIES["comments"])
    logger.info(f"Found {len(comment_captures)} comments in {file_path}")
    
    comments_processed = 0
    for capture in comment_captures:
        logger.info(f"Comment capture structure: {type(capture)}, {capture}")
        
        try:
            if isinstance(capture, tuple) and len(capture) >= 1:
                comment_node = capture[0]
                comment_text = get_node_text(comment_node)
                create_comment_node(tx, file_path, comment_text)
                comments_processed += 1
            else:
                logger.warning(f"Unexpected comment capture format: {capture}")
        except Exception as e:
            logger.error(f"Error processing comment: {str(e)}")
    
    logger.info(f"Processed {comments_processed} comments")

    # Extract and Create Call Relationships
    call_captures = execute_query(tree, QUERIES["calls"])
    logger.info(f"Found {len(call_captures)} function calls in {file_path}")
    
    calls_processed = 0
    for capture in call_captures:
        logger.info(f"Call capture structure: {type(capture)}, {capture}")
        
        try:
            if isinstance(capture, tuple) and len(capture) >= 1:
                call_node = capture[0]
                caller_start = call_node.start_point[0] + 1
                caller_end = call_node.end_point[0] + 1
                called_function_name = get_node_text(call_node.child_by_field_name("function"))
                create_call_relationship(tx, caller_start, caller_end, called_function_name)
                calls_processed += 1
            else:
                logger.warning(f"Unexpected call capture format: {capture}")
        except Exception as e:
            logger.error(f"Error processing function call: {str(e)}")
    
    logger.info(f"Processed {calls_processed} function calls")
    logger.info(f"Completed processing file: {file_path}")

def main():
    """Main function to connect to Neo4j and process files."""
    logger.info("Starting code analysis and Neo4j loading process")

    # Connect to Neo4j
    try:
        logger.info(f"Connecting to Neo4j at {NEO4J_URI}")
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Test the connection
        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            test_value = result.single()["test"]
            if test_value == 1:
                logger.info("Successfully connected to Neo4j")
            else:
                logger.error("Connection test failed")
                return
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        return

    # Check if CODE_DIRECTORY is a file or directory
    if os.path.isfile(CODE_DIRECTORY):
        logger.info(f"CODE_DIRECTORY is a file: {CODE_DIRECTORY}")
        with driver.session() as session:
            session.execute_write(process_file, CODE_DIRECTORY)
    else:
        # Process each file in the directory
        logger.info(f"CODE_DIRECTORY is a directory: {CODE_DIRECTORY}")
        with driver.session() as session:
            for root, _, files in os.walk(CODE_DIRECTORY):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        session.execute_write(process_file, file_path)

    driver.close()
    logger.info("Codebase processed and loaded into Neo4j.")

if __name__ == "__main__":
    main()