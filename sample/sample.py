# sample.py - A more complex example simulating a basic e-commerce system.

class Product:
    """Represents a product in the store."""

    def __init__(self, product_id: int, name: str, price: float, stock: int):
        self.product_id = product_id
        self.name = name
        self.price = price
        self.stock = stock

    def __str__(self):
        return f"{self.name} (ID: {self.product_id}, Price: ${self.price:.2f}, Stock: {self.stock})"

    def update_stock(self, quantity: int):
        """Updates the stock level of the product."""
        if self.stock + quantity < 0:
            raise ValueError("Not enough stock available.")
        self.stock += quantity

class Customer:
    """Represents a customer."""

    def __init__(self, customer_id: int, name: str, email: str):
        self.customer_id = customer_id
        self.name = name
        self.email = email
        self.cart = {}  # product_id: quantity

    def __str__(self):
        return f"{self.name} ({self.email})"

    def add_to_cart(self, product: Product, quantity: int):
        """Adds a product to the customer's cart."""
        if product.product_id not in self.cart:
            self.cart[product.product_id] = 0
        if self.cart[product.product_id] + quantity > product.stock:
            raise ValueError(f"Not enough stock of {product.name} available.")
        self.cart[product.product_id] += quantity

    def remove_from_cart(self, product: Product, quantity: int):
        """Removes a product from the customer's cart."""
        if product.product_id not in self.cart:
            return  # Nothing to remove

        self.cart[product.product_id] -= quantity
        if self.cart[product.product_id] <= 0:
            del self.cart[product.product_id]

    def view_cart(self, products: dict):
        """Displays the contents of the customer's cart."""
        if not self.cart:
            print("Your cart is empty.")
            return

        print("Items in your cart:")
        total_cost = 0
        for product_id, quantity in self.cart.items():
            product = products.get(product_id)
            if product:
                print(f"- {product.name}: {quantity} x ${product.price:.2f} = ${product.price * quantity:.2f}")
                total_cost += product.price * quantity
        print(f"Total cost: ${total_cost:.2f}")

    def checkout(self, products: dict):
        """Completes the purchase and updates stock levels."""
        if not self.cart:
            print("Your cart is empty. Nothing to checkout.")
            return

        try:
            for product_id, quantity in self.cart.items():
                product = products.get(product_id)
                if product:
                    product.update_stock(-quantity)  # Reduce stock
            print("Checkout successful! Thank you for your purchase.")
            self.cart = {}  # Empty the cart
        except ValueError as e:
            print(f"Checkout failed: {e}")


class Order:
    """Represents an order."""
    _next_order_id = 1

    def __init__(self, customer: Customer, products_dict: dict):
        self.order_id = Order._next_order_id
        Order._next_order_id += 1
        self.customer = customer
        self.items = {}  # product_id: quantity
        self.total_amount = 0.0
        self.status = "Pending"

        for product_id, quantity in customer.cart.items():
            product = products_dict.get(product_id)
            if product:
                self.items[product_id] = quantity
                self.total_amount += product.price * quantity

    def __str__(self):
        return (f"Order ID: {self.order_id}, Customer: {self.customer.name}, "
                f"Total: ${self.total_amount:.2f}, Status: {self.status}")

    def display_order_details(self, products_dict: dict):
        """Displays detailed information about the order."""
        print(f"Order ID: {self.order_id}")
        print(f"Customer: {self.customer.name} ({self.customer.email})")
        print("Items:")
        for product_id, quantity in self.items.items():
            product = products_dict.get(product_id)
            if product:
                print(f"  - {product.name}: {quantity} x ${product.price:.2f}")
        print(f"Total Amount: ${self.total_amount:.2f}")
        print(f"Status: {self.status}")

def create_sample_products():
    """Creates a dictionary of sample products."""
    products = {
        1: Product(1, "Laptop", 1200.00, 10),
        2: Product(2, "Mouse", 25.00, 50),
        3: Product(3, "Keyboard", 75.00, 30),
        4: Product(4, "Monitor", 300.00, 15),
        5: Product(5, "Headphones", 100.00, 20),
    }
    return products

def main():
    # Create some sample products
    products = create_sample_products()

    # Create a customer
    customer1 = Customer(101, "Alice Smith", "alice.smith@example.com")

    # Add some products to the cart
    customer1.add_to_cart(products[1], 2)  # 2 Laptops
    customer1.add_to_cart(products[3], 1)  # 1 Keyboard

    # View the cart
    print("Customer's Cart:")
    customer1.view_cart(products)

    # Create an order
    order1 = Order(customer1, products)
    print("\nOrder Details:")
    order1.display_order_details(products)

    # Checkout
    customer1.checkout(products)

    # Show updated stock levels
    print("\nUpdated Stock Levels:")
    for product_id, product in products.items():
        print(product)

    # Try to add more to cart than available
    try:
        customer1.add_to_cart(products[1], 15)  # Try to add 15 laptops
    except ValueError as e:
        print(f"\nError: {e}")

    # Create another customer and order
    customer2 = Customer(102, "Bob Johnson", "bob.johnson@example.com")
    customer2.add_to_cart(products[2], 5) # 5 mice
    customer2.add_to_cart(products[5], 2) # 2 headphones
    order2 = Order(customer2, products)
    print("\nOrder Details (Customer 2):")
    order2.display_order_details(products)

if __name__ == "__main__":
    main()