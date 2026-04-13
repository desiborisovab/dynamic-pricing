"""
api/app.py

Flask API that serves real-time price recommendations from the trained DQN agent.

Endpoints:
    GET  /health          — liveness check
    GET  /info            — model version + feature info
    POST /price           — get a price recommendation for one product
    POST /price/batch     — get recommendations for multiple products

Environment variables:
    AZURE_STORAGE_CONNECTION_STRING   — Azure Blob Storage connection string
    MODEL_VERSION                     — which version to load (default: "latest")
    PORT                              — port to listen on (default: 8000)
"""

import os
import io
import json
import logging
import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── Global model state ────────────────────────────────────────────────────────
MODEL = {
    "agent":          None,
    "scaler":         None,
    "encoders":       None,
    "multipliers":    None,
    "version":        None,
    "loaded_at":      None,
}

CONTAINER_NAME = "dynamic-pricing-model"
N_ACTIONS      = 10
STATE_DIM      = 14
MULTIPLIERS    = [1.000, 1.025, 1.050, 1.075, 1.100,
                  1.125, 1.150, 1.175, 1.200, 1.225]


# ── Model loading ─────────────────────────────────────────────────────────────
def _download_blob(blob_service, version: str, filename: str) -> bytes:
    blob_client = blob_service.get_blob_client(
        container=CONTAINER_NAME,
        blob=f"{version}/{filename}"
    )
    return blob_client.download_blob().readall()


def load_model(version: str = None):
    """Download artefacts from Blob Storage and load into memory."""
    import torch
    from azure.storage.blob import BlobServiceClient
    # import DQNAgent from the same package
    from dqn_agent_pytorch import DQNAgent

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING not set")

    version = version or os.getenv("MODEL_VERSION", "latest")
    log.info(f"Loading model version: {version}")

    blob_service = BlobServiceClient.from_connection_string(conn_str)

    # Download checkpoint
    log.info("  Downloading dqn_checkpoint.pt ...")
    ckpt_bytes = _download_blob(blob_service, version, "dqn_checkpoint.pt")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(ckpt_bytes)
        tmp_path = tmp.name

    agent = DQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    agent.load(tmp_path)
    Path(tmp_path).unlink()

    # Download scaler
    log.info("  Downloading sim_scaler.pkl ...")
    scaler_bytes = _download_blob(blob_service, version, "sim_scaler.pkl")
    scaler = pickle.loads(scaler_bytes)

    # Download label encoders
    log.info("  Downloading label_encoders.pkl ...")
    enc_bytes = _download_blob(blob_service, version, "label_encoders.pkl")
    encoders  = pickle.loads(enc_bytes)

    MODEL["agent"]       = agent
    MODEL["scaler"]      = scaler
    MODEL["encoders"]    = encoders
    MODEL["multipliers"] = MULTIPLIERS
    MODEL["version"]     = version
    MODEL["loaded_at"]   = datetime.utcnow().isoformat()

    log.info(f"Model loaded successfully (version={version})")


# ── Feature builder ───────────────────────────────────────────────────────────
def build_state(product: dict) -> np.ndarray:
    """Convert a raw product dict into the 14-feature state vector."""
    from datetime import datetime as dt

    enc      = MODEL["encoders"]
    scaler   = MODEL["scaler"]

    date     = dt.strptime(product["date"], "%Y-%m-%d")
    price    = float(product["current_price"])
    comp     = float(product["competitor_price"])
    discount = float(product["discount"])

    cat     = product["category"]
    region  = product["region"]
    weather = product["weather"]
    season  = product["season"]

    # Validate categorical values
    for key, val, enc_key in [
        ("category", cat, "category"),
        ("region", region, "region"),
        ("weather", weather, "weather_condition"),
        ("season", season, "seasonality"),
    ]:
        if val not in enc[enc_key]:
            valid = list(enc[enc_key].keys())
            raise ValueError(f"Unknown {key} '{val}'. Valid values: {valid}")

    features = np.array([[
        price,
        comp,
        float(product["inventory_level"]),
        discount,
        float(product["holiday_promotion"]),
        date.weekday(),
        date.month,
        (date.month - 1) // 3 + 1,
        price / (comp + 1e-9),               # price_ratio
        price * (1 - discount / 100),        # effective_price
        enc["category"][cat],
        enc["region"][region],
        enc["weather_condition"][weather],
        enc["seasonality"][season],
    ]])

    return scaler.transform(features)[0]


# ── Pricing logic ─────────────────────────────────────────────────────────────
def price_product(product: dict) -> dict:
    agent      = MODEL["agent"]
    multipliers = MODEL["multipliers"]

    state      = build_state(product)
    action     = agent.act(state, training=False)
    multiplier = multipliers[action]

    base_cost  = float(product["current_price"]) * 0.60
    new_price  = base_cost * multiplier
    discount   = float(product["discount"])
    eff_price  = new_price * (1 - discount / 100)

    return {
        "product_id":          product.get("product_id", "unknown"),
        "recommended_price":   round(new_price, 2),
        "effective_price":     round(eff_price, 2),
        "base_cost":           round(base_cost, 2),
        "action":              int(action),
        "multiplier":          multiplier,
        "model_version":       MODEL["version"],
        "priced_at":           datetime.utcnow().isoformat(),
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if MODEL["agent"] is None:
        return jsonify({"status": "unavailable", "reason": "model not loaded"}), 503
    return jsonify({
        "status":       "ok",
        "model_version": MODEL["version"],
        "loaded_at":    MODEL["loaded_at"],
    })


@app.get("/info")
def info():
    if MODEL["agent"] is None:
        return jsonify({"error": "model not loaded"}), 503

    enc = MODEL["encoders"]
    return jsonify({
        "model_version":  MODEL["version"],
        "loaded_at":      MODEL["loaded_at"],
        "n_actions":      N_ACTIONS,
        "price_multipliers": MULTIPLIERS,
        "valid_categories": list(enc["category"].keys()),
        "valid_regions":    list(enc["region"].keys()),
        "valid_weather":    list(enc["weather_condition"].keys()),
        "valid_seasons":    list(enc["seasonality"].keys()),
    })


@app.post("/price")
def price():
    """
    Get a price recommendation for a single product.

    Request body (JSON):
    {
        "product_id":       "P0042",          // optional
        "current_price":    33.50,
        "competitor_price": 29.69,
        "inventory_level":  231,
        "discount":         20,
        "holiday_promotion": 0,
        "date":             "2026-04-13",
        "category":         "Toys",
        "region":           "North",
        "weather":          "Rainy",
        "season":           "Spring"
    }
    """
    if MODEL["agent"] is None:
        return jsonify({"error": "model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "request body must be JSON"}), 400

    required = ["current_price", "competitor_price", "inventory_level",
                "discount", "holiday_promotion", "date",
                "category", "region", "weather", "season"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    try:
        result = price_product(data)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        log.exception("Unexpected error in /price")
        return jsonify({"error": "internal server error"}), 500


@app.post("/price/batch")
def price_batch():
    """
    Get price recommendations for multiple products.

    Request body (JSON):
    {
        "products": [ { ...same as /price... }, ... ]
    }
    """
    if MODEL["agent"] is None:
        return jsonify({"error": "model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "products" not in data:
        return jsonify({"error": "request body must have a 'products' array"}), 400

    products = data["products"]
    if not isinstance(products, list) or len(products) == 0:
        return jsonify({"error": "'products' must be a non-empty array"}), 400
    if len(products) > 500:
        return jsonify({"error": "batch size limit is 500"}), 400

    results = []
    errors  = []
    for i, product in enumerate(products):
        try:
            results.append(price_product(product))
        except Exception as e:
            errors.append({"index": i, "product_id": product.get("product_id"), "error": str(e)})

    return jsonify({
        "results":       results,
        "errors":        errors,
        "total":         len(products),
        "succeeded":     len(results),
        "failed":        len(errors),
        "model_version": MODEL["version"],
    })


# ── Startup ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    version = os.getenv("MODEL_VERSION", "latest")
    try:
        load_model(version)
    except Exception as e:
        log.error(f"Failed to load model at startup: {e}")
        log.error("API will return 503 until model is loaded.")

    port = int(os.getenv("PORT", 8000))
    log.info(f"Starting API on port {port}")
    app.run(host="0.0.0.0", port=port)