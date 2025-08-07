import json
import os
from typing import Any, Dict


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_data() -> Dict[str, Any]:
    """Load retail workflow data from JSON files.

    Returns a dictionary with keys: 'users', 'products', 'orders'.
    """
    base_dir = os.path.dirname(__file__)
    users = _read_json(os.path.join(base_dir, "users.json"))
    products = _read_json(os.path.join(base_dir, "products.json"))
    orders = _read_json(os.path.join(base_dir, "orders.json"))
    return {"users": users, "products": products, "orders": orders}


