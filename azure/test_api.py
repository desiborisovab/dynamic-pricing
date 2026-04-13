"""
azure/test_api.py

Tests the deployed API endpoints.

Usage:
    # Against deployed Azure API:
    python azure/test_api.py

    # Against local API (run api/app.py first):
    API_URL=http://localhost:8000 python azure/test_api.py
"""

import os
import json
import sys

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

SAMPLE_PRODUCT = {
    "product_id":        "P0042",
    "current_price":     33.50,
    "competitor_price":  29.69,
    "inventory_level":   231,
    "discount":          20,
    "holiday_promotion": 0,
    "date":              "2026-04-13",
    "category":          "Toys",
    "region":            "North",
    "weather":           "Rainy",
    "season":            "Spring"
}


def test(label, method, path, **kwargs):
    url = f"{API_URL}{path}"
    resp = getattr(requests, method)(url, **kwargs)
    status = "PASS" if resp.status_code < 400 else "FAIL"
    print(f"  [{status}]  {method.upper()} {path}  →  {resp.status_code}")
    if resp.status_code >= 400:
        print(f"         {resp.text[:200]}")
    return resp


print(f"\nTesting API at: {API_URL}")
print("=" * 50)

# Health check
r = test("GET /health", "get", "/health")
if r.status_code == 200:
    d = r.json()
    print(f"         version={d.get('model_version')}  loaded_at={d.get('loaded_at')}")

# Info
r = test("GET /info", "get", "/info")
if r.status_code == 200:
    d = r.json()
    print(f"         categories={d.get('valid_categories')}")
    print(f"         regions={d.get('valid_regions')}")

# Single price
r = test("POST /price", "post", "/price",
         json=SAMPLE_PRODUCT,
         headers={"Content-Type": "application/json"})
if r.status_code == 200:
    d = r.json()
    print(f"         recommended_price=${d.get('recommended_price')}")
    print(f"         effective_price=${d.get('effective_price')}")
    print(f"         multiplier={d.get('multiplier')}  action={d.get('action')}")

# Batch price
batch_payload = {"products": [SAMPLE_PRODUCT, {**SAMPLE_PRODUCT, "product_id": "P0043", "discount": 0}]}
r = test("POST /price/batch", "post", "/price/batch",
         json=batch_payload,
         headers={"Content-Type": "application/json"})
if r.status_code == 200:
    d = r.json()
    print(f"         succeeded={d.get('succeeded')}  failed={d.get('failed')}")
    for res in d.get("results", []):
        print(f"         {res['product_id']}  →  ${res['recommended_price']}")

# Validation error test
r = test("POST /price (bad category)", "post", "/price",
         json={**SAMPLE_PRODUCT, "category": "InvalidCategory"},
         headers={"Content-Type": "application/json"})

# Missing fields test
r = test("POST /price (missing fields)", "post", "/price",
         json={"current_price": 33.50},
         headers={"Content-Type": "application/json"})

print("=" * 50)
print()