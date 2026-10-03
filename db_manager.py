import json
import os
import uuid
from datetime import datetime
from threading import Lock

INVENTORY_FILE = "inventory.json"
BILLS_FILE = "bills.json"
file_lock = Lock()

# Default Seed Inventory if inventory.json does not exist
DEFAULT_INVENTORY = [
    {"id": "INV-001", "name": "Organic Milk 1L", "price": 3.50, "unit": "liter"},
    {"id": "INV-002", "name": "Amul Salted Butter 500g", "price": 4.20, "unit": "pack"},
    {"id": "INV-003", "name": "Whole Wheat Bread 400g", "price": 2.10, "unit": "pack"},
    {"id": "INV-004", "name": "Coca-Cola Original 500ml", "price": 1.50, "unit": "bottle"},
]

def _init_files():
    if not os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, "w") as f:
            json.dump(DEFAULT_INVENTORY, f, indent=2)
    if not os.path.exists(BILLS_FILE):
        with open(BILLS_FILE, "w") as f:
            json.dump([], f, indent=2)

_init_files()

def load_inventory():
    with file_lock:
        with open(INVENTORY_FILE, "r") as f:
            return json.load(f)

def add_inventory_item(name: str, price: float, unit: str = "item"):
    with file_lock:
        inventory = []
        if os.path.exists(INVENTORY_FILE):
            with open(INVENTORY_FILE, "r") as f:
                inventory = json.load(f)
        
        new_item = {
            "id": f"INV-{uuid.uuid4().hex[:6].upper()}",
            "name": name.strip(),
            "price": round(float(price), 2),
            "unit": unit
        }
        inventory.append(new_item)
        
        with open(INVENTORY_FILE, "w") as f:
            json.dump(inventory, f, indent=2)
        return new_item

def save_bill(items: list, grand_total: float, status: str = "completed"):
    with file_lock:
        bills = []
        if os.path.exists(BILLS_FILE):
            with open(BILLS_FILE, "r") as f:
                bills = json.load(f)
        
        bill_record = {
            "bill_id": f"BILL-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "items": items,
            "grand_total": round(grand_total, 2)
        }
        bills.append(bill_record)
        
        with open(BILLS_FILE, "w") as f:
            json.dump(bills, f, indent=2)
        return bill_record