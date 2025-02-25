# Third-party Library Imports
from tree_sitter import Language, Parser

# Language-specific Tree-sitter Bindings
import tree_sitter_python


def print_tree(node, depth=0):
    """Recursively prints the tree structure with indentation."""
    indent = "  " * depth
    print(f"{indent}{node.type}: {node.text.decode()}")
    for child in node.children:
        print_tree(child, depth + 1)


def main():
    """Main function to demonstrate basic Tree-sitter usage."""

    # 1. Load the Python Language
    PY_LANGUAGE = Language(tree_sitter_python.language())

    # 2. Create a Parser
    parser = Parser(PY_LANGUAGE)

    # 3. Parse a String of Code (More Complex Example)
    code_string = """
import os
import sys

# This is a module-level comment.

def add(x, y: int = 1) -> int:
    \"\"\"Adds two numbers.

    Args:
        x: The first number.
        y: The second number (optional, defaults to 1).

    Returns:
        The sum of x and y.
    \"\"\"
    if not isinstance(x, (int, float)):
        raise TypeError("x must be a number")
    if not isinstance(y, (int, float)):
        raise TypeError("y must be a number")
    return x + y

class DataProcessor:
    \"\"\"Processes data from a file.\"\"\"

    def __init__(self, filename: str):
        \"\"\"Initializes the DataProcessor.

        Args:
            filename: The name of the file to process.
        \"\"\"
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

    # 4. Print the Tree
    print("-" * 20, "Parsed Tree", "-" * 20)
    print_tree(tree.root_node)  # Use our helper function
    # print(tree.root_node.sexp()) # Or, use the built-in .sexp() method


if __name__ == "__main__":
    main()