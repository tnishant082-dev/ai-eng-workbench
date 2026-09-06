from __future__ import annotations

import os
import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("AIWB_API_KEY", "dev-workbench-key")


def api(method: str, path: str, json_body: dict | None = None, json: dict | None = None):
    body = json_body if json_body is not None else json
    url = f"{API_URL}{path}"
    headers = {"X-API-Key": API_KEY}
    with httpx.Client(timeout=180.0, headers=headers) as client:
        r = client.request(method, url, json=body)
        r.raise_for_status()
        return r.json()
