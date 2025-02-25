# Third-party Library Imports
from tree_sitter import Language, Parser, Query

# Language-specific Tree-sitter Bindings
import tree_sitter_python


def main():
    """Main function to demonstrate basic Tree-sitter usage."""

    # 1. Load the Python Language
    PY_LANGUAGE = Language(tree_sitter_python.language())

    # 2. Create a Parser
    parser = Parser(PY_LANGUAGE)

    # 3. Parse a String of Code
    code_string = """
import os
import sys

# This is a module-level comment.

def add(x, y: int = 1) -> int:
    \"\"\"Adds two numbers.\"\"\"
    if not isinstance(x, (int, float)):
        raise TypeError("x must be a number")
    if not isinstance(y, (int, float)):
        raise TypeError("y must be a number")
    return x + y

class DataProcessor:
    \"\"\"Processes data from a file.\"\"\"

    def __init__(self, filename: str):
        \"\"\"Initializes the DataProcessor.\"\"\"
        self.filename = filename
        self.data = None

    def load_data(self):
        \"\"\"Loads data from the file.\"\"\"
        try:
            with open(self.filename, "r") as f:
                self.data = f.readlines()
        except FileNotFoundError:
            print(f"Error: File not found: {self.filename}")
            self.data = []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.data = []


    def process_data(self):
        \"\"\"Processes the loaded data (simple example).\"\"\"
        if self.data:
            for line in self.data:
                # Remove leading/trailing whitespace and print
                print(line.strip())

def main():
    \"\"\"Entry point of the program.\"\"\"
    processor = DataProcessor("my_data.txt")
    processor.load_data()
    processor.process_data()

if __name__ == "__main__":
    main()
"""
    tree = parser.parse(bytes(code_string, "utf8"))

    # 4. Define and Execute Queries
    print("-" * 20, "Query Results", "-" * 20)

    queries = {
        "function_names": """
            (function_definition name: (identifier) @function_name)
        """,
        "class_names": """
            (class_definition name: (identifier) @class_name)
        """,
        "function_calls": """
            (call function: (identifier) @called_function)
        """,
        "comments": """
            (comment) @comment_text
        """,
        "add_function_calls": """
            (call
                function: (identifier) @function_name
                (#eq? @function_name "add")
            )
        """,
    }

    for query_name, query_str in queries.items():
        print(f"--- Query: {query_name} ---")
        query = PY_LANGUAGE.query(query_str)
        matches = query.matches(tree.root_node)  # Use .matches() instead of .captures()

        for pattern_index, captures in matches: # Iterate through matches
            for capture_name, node_list in captures.items(): #captures is dict
                for node in node_list: # node_list is list of nodes
                    print(f"  Capture: {capture_name}, Text: {node.text.decode()}")
        print()

if __name__ == "__main__":
    main()