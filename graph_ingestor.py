import os
import logging
from tree_sitter import Language, Parser, Query
import tree_sitter_python
from neo4j import GraphDatabase
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from dotenv import load_dotenv, find_dotenv

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
# Use raw string (r prefix) to prevent escape sequence interpretation
CODE_DIRECTORY = os.environ.get("CODE_DIRECTORY", r"./sample")

logger.info(f"Neo4j URI: {NEO4J_URI}")
logger.info(f"Neo4j User: {NEO4J_USER}")
logger.info(f"Code Directory: {CODE_DIRECTORY}")

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
        ) @function
    """,
    "classes": """
        (class_definition
            name: (identifier) @class_name
            body: (block) @class_body
            superclasses: (argument_list)? @superclasses
        ) @class
    """,
    "docstrings": """
        (function_definition
            name: (identifier) @function_name
            body: (block
                (expression_statement
                    (string) @docstring
                ) .
            )
        )
        
        (class_definition
            name: (identifier) @class_name
            body: (block
                (expression_statement
                    (string) @docstring
                ) .
            )
        )
    """,
    "comments": """
        (comment) @comment_text
    """,
    "calls": """
        (call
            function: [(identifier) @called_function_name
                      (attribute
                        object: (identifier) @object_name
                        attribute: (identifier) @method_name)]
        ) @function_call
    """,
    "imports": """
        (import_statement
            name: (dotted_name) @import_name
        ) @import
        
        (import_from_statement
            module_name: (dotted_name) @module_name
            name: (dotted_name) @import_name
        ) @import_from
    """,
    "variables": """
        (assignment
            left: [(identifier) @variable_name
                  (tuple_pattern
                    (identifier) @variable_name)]
            right: (expression_list) @variable_value
        ) @assignment
    """,
    "decorators": """
        (decorator
            name: [(identifier) @decorator_name
                  (attribute
                    object: (identifier) @decorator_object
                    attribute: (identifier) @decorator_attribute)]
        ) @decorator_def
    """,
    "conditionals": """
        (if_statement) @if_statement
        (for_statement) @for_statement
        (while_statement) @while_statement
        (try_statement) @try_statement
    """,
}

# --- Helper Functions ---

def execute_query(tree, query_str):
    """Executes a Tree-sitter query and returns the captures."""
    try:
        query = PY_LANGUAGE.query(query_str)
        
        # Get matches
        matches = query.matches(tree.root_node)
        
        # Process matches to extract node-capture pairs
        result = []
        for match in matches:
            capture_dict = {}
            
            # Check the structure of match
            if isinstance(match, tuple) and len(match) >= 2:
                # match[0] is typically the pattern index
                # match[1] is typically the captures dictionary
                captures = match[1]
                
                for name, node in captures.items():
                    capture_dict[name] = node
            else:
                logger.warning(f"Unexpected match structure: {match}")
            
            result.append(capture_dict)
        
        return result
    except Exception as e:
        logger.error(f"Error in execute_query: {str(e)}")
        return []

def get_node_text(node, source_code):
    """Gets the text content of a Tree-sitter node."""
    if node is None:
        return ""
    
    # Handle case where node is a list (as seen in the debug output)
    if isinstance(node, list):
        if not node:  # Empty list
            return ""
        node = node[0]  # Take the first node in the list
    
    start_byte = node.start_byte
    end_byte = node.end_byte
    return source_code[start_byte:end_byte].decode('utf-8')

def extract_docstring(function_node, source_code):
    """Extract docstring from a function or class node."""
    if function_node is None:
        return ""
    
    # Handle case where function_node is a list
    if isinstance(function_node, list):
        if not function_node:
            return ""
        function_node = function_node[0]
    
    # Look for the first expression statement in the body that is a string
    body_node = function_node.child_by_field_name('body')
    if body_node is None:
        return ""
    
    for child in body_node.children:
        if child.type == 'expression_statement':
            expr_child = child.child(0)
            if expr_child and expr_child.type == 'string':
                docstring = get_node_text(expr_child, source_code)
                # Clean up the docstring (remove quotes, normalize whitespace)
                docstring = re.sub(r'^["\']|["\']$', '', docstring)
                docstring = re.sub(r'^["\'\s]+|["\'\s]+$', '', docstring)
                return docstring
    
    return ""

def get_parent_function_or_class(node):
    """Find the parent function or class of a node."""
    current = node.parent
    while current:
        if current.type in ('function_definition', 'class_definition'):
            return current
        current = current.parent
    return None

def get_function_parameters(params_node, source_code):
    """Extract function parameters as a list."""
    if params_node is None:
        return []
    
    parameters = []
    for child in params_node.children:
        if child.type == 'identifier':
            param_name = get_node_text(child, source_code)
            parameters.append(param_name)
    
    return parameters

def get_class_superclasses(superclasses_node, source_code):
    """Extract superclasses from a class definition."""
    if superclasses_node is None:
        return []
    
    superclasses = []
    for child in superclasses_node.children:
        if child.type == 'identifier':
            superclass_name = get_node_text(child, source_code)
            superclasses.append(superclass_name)
    
    return superclasses

# --- Neo4j Interaction ---

def clear_database(tx):
    """Clear all nodes and relationships in the database."""
    logger.info("Clearing existing database")
    tx.run("MATCH (n) DETACH DELETE n")

def create_indexes(tx):
    """Create necessary indexes for performance."""
    logger.info("Creating indexes")
    tx.run("CREATE INDEX file_path IF NOT EXISTS FOR (f:File) ON (f.path)")
    tx.run("CREATE INDEX function_name IF NOT EXISTS FOR (f:Function) ON (f.name)")
    tx.run("CREATE INDEX class_name IF NOT EXISTS FOR (c:Class) ON (c.name)")
    tx.run("CREATE INDEX variable_name IF NOT EXISTS FOR (v:Variable) ON (v.name)")
    tx.run("CREATE INDEX import_name IF NOT EXISTS FOR (i:Import) ON (i.name)")

def create_file_node(tx, file_path):
    """Creates a File node in Neo4j."""
    logger.info(f"Creating File node for: {file_path}")
    result = tx.run("""
        MERGE (f:File {path: $path})
        RETURN f
    """, path=file_path)
    return result.single()["f"]

def create_function_node(tx, file_path, function_name, start_line, end_line, docstring="", parameters=None):
    """Create a Function node in Neo4j."""
    logger.info(f"Creating Function node: {function_name} in {file_path}")
    
    # Convert parameters list to string if provided
    params_str = ", ".join(parameters) if parameters else ""
    
    # Use a simpler query with explicit parameters
    query = """
        MATCH (f:File {path: $file_path})
        MERGE (func:Function {name: $function_name, start_line: $start_line, end_line: $end_line})
        SET func.docstring = $docstring,
            func.parameters = $params_str
        MERGE (f)-[:CONTAINS]->(func)
        RETURN func
    """
    
    try:
        result = tx.run(query, 
                       file_path=file_path, 
                       function_name=function_name, 
                       start_line=start_line, 
                       end_line=end_line, 
                       docstring=docstring, 
                       params_str=params_str)
        
        return result.single()
    except Exception as e:
        logger.error(f"Error creating function node: {str(e)}")
        return None

def create_class_node(tx, file_path, class_name, start_line, end_line, docstring, superclasses=None):
    """Creates a Class node and connects it to its File."""
    logger.info(f"Creating Class node: {class_name} in {file_path}")
    
    if superclasses is None:
        superclasses = []
    
    result = tx.run("""
        MATCH (f:File {path: $file_path})
        MERGE (cls:Class {name: $name, file_path: $file_path})
        SET cls.start_line = $start_line,
            cls.end_line = $end_line,
            cls.docstring = $docstring,
            cls.superclasses = $superclasses
        MERGE (f)-[:CONTAINS]->(cls)
        RETURN cls
    """, file_path=file_path, name=class_name, start_line=start_line,
           end_line=end_line, docstring=docstring, superclasses=superclasses)
    
    return result.single()["cls"]

def create_comment_node(tx, file_path, comment_text, line_number):
    """Create a Comment node in Neo4j."""
    logger.info(f"Creating Comment node in {file_path} at line {line_number}")
    
    # Create a dictionary of parameters to pass to the query
    query_params = {
        "file_path": file_path,
        "comment_text": comment_text,
        "line_number": line_number
    }
    
    try:
        result = tx.run("""
            MATCH (f:File {path: $file_path})
            MERGE (c:Comment {text: $comment_text, line: $line_number, file_path: $file_path})
            MERGE (f)-[:CONTAINS]->(c)
            RETURN c
        """, **query_params)
        
        return result.single()
    except Exception as e:
        logger.error(f"Error creating comment node: {str(e)}")
        return None

def create_method_node(tx, file_path, class_name, method_name, start_line, end_line, docstring="", parameters=None):
    """Create a Method node (Function node connected to a Class) in Neo4j."""
    logger.info(f"Creating Method node: {method_name} in class {class_name} in {file_path}")
    
    # Convert parameters list to string if provided
    params_str = ", ".join(parameters) if parameters else ""
    
    # Use a simpler query with explicit parameters
    query = """
        MATCH (f:File {path: $file_path})
        MATCH (c:Class {name: $class_name})
        MERGE (func:Function {name: $method_name, start_line: $start_line, end_line: $end_line})
        SET func.docstring = $docstring,
            func.parameters = $params_str,
            func.is_method = true
        MERGE (f)-[:CONTAINS]->(func)
        MERGE (c)-[:DEFINES]->(func)
        RETURN func
    """
    
    try:
        result = tx.run(query, 
                       file_path=file_path, 
                       class_name=class_name, 
                       method_name=method_name, 
                       start_line=start_line, 
                       end_line=end_line, 
                       docstring=docstring, 
                       params_str=params_str)
        
        return result.single()
    except Exception as e:
        logger.error(f"Error creating method node: {str(e)}")
        return None

def create_call_relationship(tx, file_path, caller_start, caller_end, called_function_name, object_name=None):
    """Create a CALLS relationship between a caller location and a function."""
    if object_name:
        logger.info(f"Creating CALLS relationship from {file_path}:{caller_start}-{caller_end} to {object_name}.{called_function_name}")
    else:
        logger.info(f"Creating CALLS relationship from {file_path}:{caller_start}-{caller_end} to {called_function_name}")
    
    try:
        if object_name:
            # This is a method call
            query = """
                MATCH (f:File {path: $file_path})
                MERGE (c:CallSite {file_path: $file_path, start_line: $caller_start, end_line: $caller_end})
                MERGE (f)-[:CONTAINS]->(c)
                WITH c
                MATCH (func:Function {name: $function_name})
                WHERE EXISTS((func)<-[:DEFINES]-(:Class {name: $object_name}))
                MERGE (c)-[:CALLS]->(func)
                RETURN c, func
            """
            
            result = tx.run(query, 
                           file_path=file_path, 
                           caller_start=caller_start, 
                           caller_end=caller_end, 
                           function_name=called_function_name, 
                           object_name=object_name)
        else:
            # This is a direct function call
            query = """
                MATCH (f:File {path: $file_path})
                MERGE (c:CallSite {file_path: $file_path, start_line: $caller_start, end_line: $caller_end})
                MERGE (f)-[:CONTAINS]->(c)
                WITH c
                MATCH (func:Function {name: $function_name})
                WHERE NOT EXISTS((func)<-[:DEFINES]-(:Class))
                MERGE (c)-[:CALLS]->(func)
                RETURN c, func
            """
            
            result = tx.run(query, 
                           file_path=file_path, 
                           caller_start=caller_start, 
                           caller_end=caller_end, 
                           function_name=called_function_name)
        
        return result.single()
    except Exception as e:
        logger.error(f"Error creating call relationship: {str(e)}")
        return None

def create_import_node(tx, file_path, import_name, is_from_import=False, module_name=None):
    """Create an Import node in Neo4j."""
    if is_from_import:
        logger.info(f"Creating Import node: from {module_name} import {import_name} in {file_path}")
    else:
        logger.info(f"Creating Import node: import {import_name} in {file_path}")
    
    try:
        if is_from_import:
            query = """
                MATCH (f:File {path: $file_path})
                MERGE (i:Import {name: $import_name, module: $module_name, file_path: $file_path})
                SET i.is_from_import = true
                MERGE (f)-[:IMPORTS]->(i)
                RETURN i
            """
            
            result = tx.run(query, 
                           file_path=file_path, 
                           import_name=import_name, 
                           module_name=module_name if module_name else "")
        else:
            query = """
                MATCH (f:File {path: $file_path})
                MERGE (i:Import {name: $import_name, file_path: $file_path})
                SET i.is_from_import = false
                MERGE (f)-[:IMPORTS]->(i)
                RETURN i
            """
            
            result = tx.run(query, 
                           file_path=file_path, 
                           import_name=import_name)
        
        return result.single()
    except Exception as e:
        logger.error(f"Error creating import node: {str(e)}")
        return None

def create_variable_node(tx, file_path, variable_name, value, line_number):
    """Create a Variable node in Neo4j."""
    logger.info(f"Creating Variable node: {variable_name} in {file_path} at line {line_number}")
    
    # Create a dictionary of parameters to pass to the query
    query_params = {
        "file_path": file_path,
        "variable_name": variable_name,
        "value": value,
        "line_number": line_number
    }
    
    try:
        result = tx.run("""
            MATCH (f:File {path: $file_path})
            MERGE (v:Variable {name: $variable_name, file_path: $file_path, line: $line_number})
            SET v.value = $value
            MERGE (f)-[:CONTAINS]->(v)
            RETURN v
        """, **query_params)
        
        return result.single()
    except Exception as e:
        logger.error(f"Error creating variable node: {str(e)}")
        return None

def create_decorator_relationship(tx, file_path, decorator_name, target_start_line, target_end_line, decorator_object=None):
    """Creates a decorator relationship between a decorator and its target."""
    logger.info(f"Creating Decorator relationship: {decorator_name} in {file_path}")
    
    # Find the target (function or class)
    result = tx.run("""
        MATCH (target)
        WHERE (target:Function OR target:Class) AND
              $file_path = target.file_path AND
              $target_start = target.start_line AND
              $target_end = target.end_line
        RETURN target
    """, file_path=file_path, target_start=target_start_line, target_end=target_end_line)
    
    target_record = result.single()
    if not target_record:
        logger.warning(f"No target found for decorator {decorator_name} at lines {target_start_line}-{target_end_line}")
        return
    
    target = target_record["target"]
    
    # Handle decorator with object (e.g., @app.route)
    if decorator_object:
        full_name = f"{decorator_object}.{decorator_name}"
        tx.run("""
            MATCH (target) WHERE id(target) = $target_id
            MERGE (d:Decorator {name: $full_name, object: $object, method: $method})
            MERGE (target)-[:DECORATED_BY]->(d)
        """, target_id=target.id, full_name=full_name, object=decorator_object, method=decorator_name)
    else:
        # Simple decorator (e.g., @staticmethod)
        tx.run("""
            MATCH (target) WHERE id(target) = $target_id
            MERGE (d:Decorator {name: $name})
            MERGE (target)-[:DECORATED_BY]->(d)
        """, target_id=target.id, name=decorator_name)

def create_inheritance_relationships(tx, class_name, superclasses):
    """Creates INHERITS_FROM relationships between a class and its superclasses."""
    if not superclasses:
        return
    
    logger.info(f"Creating inheritance relationships for class {class_name} to superclasses: {superclasses}")
    
    for superclass in superclasses:
        # Remove any whitespace or special characters from superclass name
        clean_superclass = superclass.strip()
        if not clean_superclass:
            continue
            
        result = tx.run("""
            MATCH (cls:Class {name: $class_name})
            MATCH (parent:Class {name: $superclass})
            MERGE (cls)-[:INHERITS_FROM]->(parent)
            RETURN parent
        """, class_name=class_name, superclass=clean_superclass)
        
        # Log the result for debugging
        if result.peek() is None:
            logger.warning(f"Could not create inheritance relationship: {class_name} -> {clean_superclass}. Parent class might not exist.")

def process_file(tx, file_path):
    """Parses a file, extracts information, and creates nodes/relationships."""
    logger.info(f"Processing file: {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            source_code = f.read()
            logger.info(f"Successfully read file: {file_path}, size: {len(source_code)} bytes")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return

    try:
        tree = parser.parse(source_code)
        logger.info(f"Successfully parsed file: {file_path}")
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")
        return

    # Create File Node
    file_node = create_file_node(tx, file_path)
    
    # Track processed classes to handle methods correctly
    processed_classes = {}
    
    # Process Classes First
    class_captures = execute_query(tree, QUERIES["classes"])
    for capture in class_captures:
        if 'class_name' in capture and 'class' in capture:
            # Extract nodes from lists
            class_node = capture['class'][0] if isinstance(capture['class'], list) else capture['class']
            class_name_node = capture['class_name'][0] if isinstance(capture['class_name'], list) else capture['class_name']
            class_body = capture.get('class_body')
            if isinstance(class_body, list) and class_body:
                class_body = class_body[0]
            
            class_name = get_node_text(class_name_node, source_code)
            
            start_line = class_node.start_point[0] + 1
            end_line = class_node.end_point[0] + 1
            
            # Extract docstring
            docstring = extract_docstring(class_node, source_code)
            
            # Extract superclasses if available
            superclasses = []
            if 'superclasses' in capture and capture['superclasses']:
                superclasses_node = capture['superclasses'][0] if isinstance(capture['superclasses'], list) else capture['superclasses']
                superclasses = get_class_superclasses(superclasses_node, source_code)
            
            # Create class node
            class_db_node = create_class_node(tx, file_path, class_name, start_line, end_line, docstring, superclasses)
            
            # Create inheritance relationships if superclasses exist
            if superclasses:
                create_inheritance_relationships(tx, class_name, superclasses)
            
            # Store for later reference
            processed_classes[class_node.id] = {
                'name': class_name,
                'node': class_node,
                'body': class_body
            }
    
    # Process Functions and Methods
    function_captures = execute_query(tree, QUERIES["functions"])
    for capture in function_captures:
        if 'function_name' in capture and 'function' in capture:
            # Extract nodes from lists
            function_node = capture['function'][0] if isinstance(capture['function'], list) else capture['function']
            function_name_node = capture['function_name'][0] if isinstance(capture['function_name'], list) else capture['function_name']
            
            function_name = get_node_text(function_name_node, source_code)
            
            start_line = function_node.start_point[0] + 1
            end_line = function_node.end_point[0] + 1
            
            # Extract docstring
            docstring = extract_docstring(function_node, source_code)
            
            # Extract parameters
            parameters = []
            if 'function_params' in capture and capture['function_params']:
                params_node = capture['function_params'][0] if isinstance(capture['function_params'], list) else capture['function_params']
                parameters = get_function_parameters(params_node, source_code)
            
            # Check if this is a method (function inside a class)
            parent_class = None
            for class_id, class_info in processed_classes.items():
                class_body = class_info.get('body')
                if class_body and function_node.start_byte >= class_body.start_byte and function_node.end_byte <= class_body.end_byte:
                    parent_class = class_info
                    break
            
            if parent_class:
                # This is a method
                create_method_node(tx, file_path, parent_class['name'], function_name, 
                                  start_line, end_line, docstring, parameters)
            else:
                # This is a standalone function
                create_function_node(tx, file_path, function_name, start_line, end_line, 
                                    docstring, parameters)
    
    # Process Comments
    comment_captures = execute_query(tree, QUERIES["comments"])
    for capture in comment_captures:
        if 'comment_text' in capture:
            comment_node = capture['comment_text'][0] if isinstance(capture['comment_text'], list) else capture['comment_text']
            comment_text = get_node_text(comment_node, source_code)
            line_number = comment_node.start_point[0] + 1
            
            create_comment_node(tx, file_path, comment_text, line_number)
    
    # Process Function Calls
    call_captures = execute_query(tree, QUERIES["calls"])
    for capture in call_captures:
        if 'function_call' in capture:
            call_node = capture['function_call'][0] if isinstance(capture['function_call'], list) else capture['function_call']
            caller_start = call_node.start_point[0] + 1
            caller_end = call_node.end_point[0] + 1
            
            # Handle both direct function calls and method calls
            if 'called_function_name' in capture:
                # Direct function call
                function_name = get_node_text(capture['called_function_name'], source_code)
                create_call_relationship(tx, file_path, caller_start, caller_end, function_name)
            elif 'method_name' in capture and 'object_name' in capture:
                # Method call
                method_name = get_node_text(capture['method_name'], source_code)
                object_name = get_node_text(capture['object_name'], source_code)
                create_call_relationship(tx, file_path, caller_start, caller_end, method_name, object_name)
    
    # Process Imports
    import_captures = execute_query(tree, QUERIES["imports"])
    for capture in import_captures:
        if 'import' in capture and 'import_name' in capture:
            import_name = get_node_text(capture['import_name'], source_code)
            create_import_node(tx, file_path, import_name)
        elif 'import_from' in capture and 'import_name' in capture and 'module_name' in capture:
            import_name = get_node_text(capture['import_name'], source_code)
            module_name = get_node_text(capture['module_name'], source_code)
            create_import_node(tx, file_path, import_name, True, module_name)
    
    # Process Variables
    variable_captures = execute_query(tree, QUERIES["variables"])
    for capture in variable_captures:
        if 'assignment' in capture and 'variable_name' in capture:
            variable_node = capture['variable_name']
            variable_name = get_node_text(variable_node, source_code)
            value_node = capture.get('variable_value')
            value = get_node_text(value_node, source_code) if value_node else ""
            line_number = variable_node.start_point[0] + 1
            
            create_variable_node(tx, file_path, variable_name, value, line_number)
    
    # Process Decorators
    decorator_captures = execute_query(tree, QUERIES["decorators"])
    for capture in decorator_captures:
        if 'decorator_def' in capture:
            decorator_node = capture['decorator_def']
            
            # Find the target (function or class) that follows this decorator
            target_node = decorator_node.next_sibling
            while target_node and target_node.type in ('comment', 'decorator'):
                target_node = target_node.next_sibling
            
            if target_node and target_node.type in ('function_definition', 'class_definition'):
                target_start = target_node.start_point[0] + 1
                target_end = target_node.end_point[0] + 1
                
                # Handle both simple decorators and object.method decorators
                if 'decorator_name' in capture:
                    decorator_name = get_node_text(capture['decorator_name'], source_code)
                    create_decorator_relationship(tx, file_path, decorator_name, target_start, target_end)
                elif 'decorator_object' in capture and 'decorator_attribute' in capture:
                    decorator_object = get_node_text(capture['decorator_object'], source_code)
                    decorator_attr = get_node_text(capture['decorator_attribute'], source_code)
                    create_decorator_relationship(tx, file_path, decorator_attr, target_start, target_end, decorator_object)
    
    logger.info(f"Completed processing file: {file_path}")

def process_directory(tx, directory_path):
    """Process all Python files in a directory and its subdirectories."""
    logger.info(f"Processing directory: {directory_path}")
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                process_file(tx, file_path)

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

    # Clear database and create indexes
    with driver.session() as session:
        session.execute_write(clear_database)
        session.execute_write(create_indexes)

    # Check if CODE_DIRECTORY is a file or directory
    if os.path.isfile(CODE_DIRECTORY):
        logger.info(f"CODE_DIRECTORY is a file: {CODE_DIRECTORY}")
        with driver.session() as session:
            session.execute_write(process_file, CODE_DIRECTORY)
    else:
        # Process each file in the directory
        logger.info(f"CODE_DIRECTORY is a directory: {CODE_DIRECTORY}")
        with driver.session() as session:
            session.execute_write(process_directory, CODE_DIRECTORY)

    driver.close()
    logger.info("Codebase processed and loaded into Neo4j.")

if __name__ == "__main__":
    main()