"""
azure/upload_artefacts.py

Run this after train.py to push all model artefacts to Azure Blob Storage.

Usage:
    python azure/upload_artefacts.py

Requires:
    AZURE_STORAGE_CONNECTION_STRING set in .env or environment
"""

import os
import sys
from pathlib import Path
from datetime import datetime

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    print("ERROR: azure-storage-blob not installed. Run: pip install azure-storage-blob")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env optional — can use real env vars

# ── Config ────────────────────────────────────────────────────────────────────
CONTAINER_NAME = "dynamic-pricing-model"

ARTEFACTS = [
    "dqn_checkpoint.pt",
    "sim_scaler.pkl",
    "label_encoders.pkl",
    "results.json",
]

# ── Upload ────────────────────────────────────────────────────────────────────
def upload_artefacts(version: str = None):
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        print("ERROR: AZURE_STORAGE_CONNECTION_STRING not set.")
        print("  Add it to your .env file or export it in your shell.")
        sys.exit(1)

    version = version or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    client  = BlobServiceClient.from_connection_string(conn_str)

    # Create container if it doesn't exist
    container = client.get_container_client(CONTAINER_NAME)
    try:
        container.create_container()
        print(f"  Created container: {CONTAINER_NAME}")
    except Exception:
        pass  # already exists

    print(f"\nUploading artefacts (version: {version})")
    print("-" * 45)

    uploaded = []
    for filename in ARTEFACTS:
        path = Path(filename)
        if not path.exists():
            print(f"  SKIP  {filename}  (not found — run train.py first)")
            continue

        # Upload to both versioned path and 'latest'
        for blob_name in [f"{version}/{filename}", f"latest/{filename}"]:
            blob_client = client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
            with open(path, "rb") as f:
                blob_client.upload_blob(f, overwrite=True)

        size_kb = path.stat().st_size / 1024
        print(f"  OK    {filename}  ({size_kb:.1f} KB)")
        uploaded.append(filename)

    print("-" * 45)
    print(f"  {len(uploaded)}/{len(ARTEFACTS)} files uploaded")
    print(f"  Versioned path : {CONTAINER_NAME}/{version}/")
    print(f"  Latest path    : {CONTAINER_NAME}/latest/")
    print(f"\n  Set this in your API environment:")
    print(f"  MODEL_VERSION={version}")

    return version


if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else None
    upload_artefacts(version)