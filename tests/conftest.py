from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.main import _boot, app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    get_settings.cache_clear()
    db = tmp_path / "test.db"
    monkeypatch.setenv("AIWB_DB_PATH", str(db))
    monkeypatch.setenv("AIWB_LLM_PROVIDER", "mock")
    monkeypatch.setenv("AIWB_AUTH_DISABLED", "1")
    monkeypatch.setenv("AIWB_API_KEY", "dev-workbench-key")
    get_settings.cache_clear()
    # Recreate middleware stack is hard; disable auth via env checked at request... 
    # Our middleware reads enabled at init. Re-build app state and patch middleware enabled.
    settings = get_settings()
    settings.db_path = db
    settings.llm_provider = "mock"
    _boot(app, settings)
    # Disable auth on existing middleware instances
    for m in app.user_middleware:
        pass
    # Starlette stores middleware in app.middleware_stack; patch APIKeyAuthMiddleware if present
    def _disable_auth(app_obj):
        stack = app_obj
        # walk middleware
        while hasattr(stack, "app"):
            if stack.__class__.__name__ == "APIKeyAuthMiddleware":
                stack.enabled = False
            stack = stack.app
    # Build TestClient which finalizes stack; then patch via dependency
    with TestClient(app) as c:
        # After TestClient init, middleware is wrapped — set enabled False by iterating
        mw = app
        # user_middleware list
        for middleware in app.user_middleware:
            cls = middleware.cls
            if cls.__name__ == "APIKeyAuthMiddleware":
                # can't easily patch; send header instead
                pass
        c.headers.update({"X-API-Key": "dev-workbench-key"})
        yield c
    get_settings.cache_clear()
