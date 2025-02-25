# Third-party Library Imports
from tree_sitter import Language, Parser

# Language-specific Tree-sitter Bindings
import tree_sitter_python  # We still need this for the Python grammar


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

    # 2. Create a Parser, passing the language object *directly*
    parser = Parser(PY_LANGUAGE)

    # 3. Parse a String of Code
    code_string = """
def my_function(a, b):
    \"\"\"This is a docstring.\"\"\"
    if a > b:
        return a
    else:
        return b

class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
"""
    tree = parser.parse(bytes(code_string, "utf8"))

    # 4. Print the Tree
    print("-" * 20, "Parsed Tree", "-" * 20)
    print_tree(tree.root_node)  # Use our helper function
    # print(tree.root_node.sexp()) # Or, use the built-in .sexp() method


if __name__ == "__main__":
    main()