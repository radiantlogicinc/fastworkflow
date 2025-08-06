import json
import os
from typing import Dict, Any


def load_data() -> Dict[str, Any]:
    """Load retail data from JSON files."""
    current_dir = os.path.dirname(__file__)
    
    # Load users data
    users_path = os.path.join(current_dir, "users.json")
    with open(users_path, 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    # Load products data
    products_path = os.path.join(current_dir, "products.json")
    with open(products_path, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    # Load orders data
    orders_path = os.path.join(current_dir, "orders.json")
    with open(orders_path, 'r', encoding='utf-8') as f:
        orders = json.load(f)
    
    return {
        "users": users,
        "products": products,
        "orders": orders
    }