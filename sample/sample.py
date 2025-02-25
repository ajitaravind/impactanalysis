# This is a sample Python file for testing code structure extraction
# It contains functions, classes, methods, and function calls

def calculate_sum(a, b):
    """
    Calculate the sum of two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    return a + b

def calculate_product(a, b):
    """Multiply two numbers together"""
    # Call another function from within this function
    if a == 0 or b == 0:
        return 0
    return a * b

# A simple class with methods
class Calculator:
    """A simple calculator class that performs basic operations"""
    
    def __init__(self, initial_value=0):
        """Initialize the calculator with a value"""
        self.value = initial_value
    
    def add(self, x):
        """Add a number to the current value"""
        # Call a standalone function from a method
        self.value = calculate_sum(self.value, x)
        return self.value
    
    def multiply(self, x):
        """Multiply the current value by a number"""
        self.value = calculate_product(self.value, x)
        return self.value
    
    def reset(self):
        """Reset the calculator to zero"""
        self.value = 0
        return self.value

# Create an instance of the Calculator class
calc = Calculator(10)

# Call some methods on the instance
calc.add(5)      # Should call the add method
result = calc.multiply(2)  # Should call the multiply method

# Call standalone functions
total = calculate_sum(15, 27)
product = calculate_product(5, 6)

print(f"Calculator value: {calc.value}")
print(f"Sum: {total}")
print(f"Product: {product}")